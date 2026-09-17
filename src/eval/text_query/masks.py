"""
masks.py

Phase 2 (INSTRUCTIONS_text_query.md, Entscheidung 2026-09-17): baut pro Kachel und Backbone
GENAU EINE bool-Maske im Kachelraster (gleiche Shape/Transform/CRS wie das zugehoerige Arm-C-
Feld aus Phase 1) - Reaktion auf die in Phase 1 gefundene Kachel-Ueberlappung. Kombiniert zwei
Bedingungen (UND):

  1. Eindeutigkeitsregel: ein gueltiges Pixel gehoert GENAU der Kachel, deren Kachelmittelpunkt
     (Bounding-Box-Zentrum der ROHEN, unbeschnittenen Kachel, in Weltkoordinaten, bei
     unterschiedlicher CRS gemeinsam in COMMON_CRS verglichen) seinem Pixelzentrum am naechsten
     liegt. Bei exaktem Gleichstand gewinnt die Kachel mit dem lexikographisch kleineren
     Dateinamen. Nur fuer Pixel relevant, die auch in einer ueberlappenden Nachbarkachel gueltig
     sind - alle anderen gueltigen Pixel behalten die Kachel automatisch.
  2. AOI-Polygon-Maske: Pixelzentrum liegt innerhalb des Greater-London-Polygons
     (data/aoi/london.gpkg), NICHT nur innerhalb der Bounding Box, mit der
     build_sdm_projected_field.py die Arm-C-Felder zugeschnitten hat (siehe check_aoi_is_bbox_crop()).

Die Arm-C-Embeddings selbst werden NICHT veraendert - nur eine zusaetzliche Maske gespeichert.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Repo-Root

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import box

import config as cfg
from tiles import list_tile_pairs

AOI_PATH = cfg.AOI_DIR / "london.gpkg"
MASK_ROOT = cfg.DATA / "interim" / "text_query_masks"
COMMON_CRS = "EPSG:32630"
PIXEL_AREA_M2 = 100.0  # 10m x 10m, beide Backbones (Phase 0 verifiziert)


def check_aoi_is_bbox_crop(backbone: str = "tessera") -> dict:
    """Vergleicht 'gueltig laut Arm-C' gegen 'Pixelzentrum im AOI-Polygon' auf einer Kachel -
    wenn Arm-C nur auf die Bounding Box zugeschnitten waere (nicht das Polygon), muss die
    Polygon-Maske STRIKT WENIGER Pixel markieren als das Arm-C-Feld gueltig hat (die Ecken
    zwischen Bounding Box und der unregelmaessigen Verwaltungsgrenze sind dann in Arm-C
    faelschlich 'gueltig')."""
    pairs = list_tile_pairs(backbone)
    p = pairs[0]
    aoi = gpd.read_file(AOI_PATH)
    with rasterio.open(p.proj_path) as src:
        band1 = src.read(1)
        transform, crs, shape = src.transform, src.crs, src.shape
        valid = np.isfinite(band1)
    aoi_in_crs = aoi.to_crs(crs)
    poly_mask = rasterize([(aoi_in_crs.geometry.iloc[0], 1)], out_shape=shape,
                           transform=transform, fill=0, dtype="uint8").astype(bool)
    n_valid = int(valid.sum())
    n_in_poly = int(poly_mask.sum())
    n_valid_outside_poly = int((valid & ~poly_mask).sum())
    return {"tile": p.tile_name, "n_valid_armc": n_valid, "n_pixel_center_in_polygon": n_in_poly,
            "n_valid_but_outside_polygon": n_valid_outside_poly,
            "is_bbox_crop": n_valid_outside_poly > 0}


def _reprojected_bbox_polygon(bounds, src_crs, dst_crs):
    """Bounding Box, segmentiert (Stuetzpunkt alle 500m) VOR der Reprojektion - vermeidet die
    Linearisierung einer gekruemmten Zonengrenze auf nur 4 Eckpunkten (siehe Phase-1-Check)."""
    poly = box(*bounds).segmentize(500)
    return gpd.GeoDataFrame({"geometry": [poly]}, crs=src_crs).to_crs(dst_crs).geometry.iloc[0]


class _TileData:
    """Haelt die pro Kachel wiederverwendeten Arrays/Metadaten im Speicher - vermeidet
    wiederholte rasterio-Fensterlesevorgaenge beim paarweisen Ueberlappungs-Check."""

    def __init__(self, pair):
        self.pair = pair
        with rasterio.open(pair.raw_path) as src:
            self.raw_bounds = src.bounds
        with rasterio.open(pair.proj_path) as src:
            self.band1 = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.shape = src.shape
        self.valid = np.isfinite(self.band1)

        cx = (self.raw_bounds.left + self.raw_bounds.right) / 2
        cy = (self.raw_bounds.bottom + self.raw_bounds.top) / 2
        to_common = Transformer.from_crs(str(self.crs), COMMON_CRS, always_xy=True)
        self.center_common = to_common.transform(cx, cy)
        self.footprint_common = _reprojected_bbox_polygon(self.raw_bounds, str(self.crs), COMMON_CRS)


def _find_overlap_candidates(tile_datas: list) -> list:
    """Grobe Kandidatenliste (Bounding-Box-Ueberlappung der ROHEN Kacheln in COMMON_CRS) -
    bewusst grosszuegig (kann Paare enthalten, die sich im AOI-beschnittenen Arm-C-Ausschnitt
    gar nicht mehr ueberlappen); das ist unschaedlich, das Pixel-Level-Retten in
    build_tile_mask() filtert exakt ueber die tatsaechliche Arm-C-Gueltigkeit."""
    pairs = []
    n = len(tile_datas)
    for i in range(n):
        for j in range(i + 1, n):
            if tile_datas[i].footprint_common.intersects(tile_datas[j].footprint_common):
                pairs.append((i, j))
    return pairs


def build_tile_mask(td, rivals: list) -> np.ndarray:
    """keep: bool-Array im Kachelraster von td - True wo td die Eindeutigkeitsregel gewinnt
    (oder gar kein Rivale dort gueltig ist) UND td selbst dort gueltig ist."""
    keep = td.valid.copy()
    if not rivals:
        return keep

    h, w = td.shape
    to_common = Transformer.from_crs(str(td.crs), COMMON_CRS, always_xy=True)

    for rd in rivals:
        # Grobes Suchfenster in td's Grid: Bounding Box von rd's Roh-Kachel, reprojiziert in
        # td's CRS, geschnitten mit td's eigenen Bounds.
        rd_bbox_in_td_crs = _reprojected_bbox_polygon(rd.raw_bounds, str(rd.crs), str(td.crs))
        minx, miny, maxx, maxy = rd_bbox_in_td_crs.bounds
        r0, c0 = rasterio.transform.rowcol(td.transform, minx, maxy)
        r1, c1 = rasterio.transform.rowcol(td.transform, maxx, miny)
        r0, r1 = sorted((max(r0, 0), min(r1, h)))
        c0, c1 = sorted((max(c0, 0), min(c1, w)))
        if r0 >= r1 or c0 >= c1:
            continue

        sub_valid = td.valid[r0:r1, c0:c1]
        rows_local, cols_local = np.where(sub_valid)
        if len(rows_local) == 0:
            continue
        rows = rows_local + r0
        cols = cols_local + c0

        xs, ys = rasterio.transform.xy(td.transform, rows, cols)  # Pixelzentren, td-CRS
        xs, ys = np.asarray(xs), np.asarray(ys)

        to_rival = Transformer.from_crs(str(td.crs), str(rd.crs), always_xy=True)
        xr, yr = to_rival.transform(xs, ys)
        rr, rc = rasterio.transform.rowcol(rd.transform, xr, yr)
        rr, rc = np.asarray(rr), np.asarray(rc)
        in_bounds = (rr >= 0) & (rr < rd.shape[0]) & (rc >= 0) & (rc < rd.shape[1])

        valid_in_rival = np.zeros(len(rows), dtype=bool)
        if in_bounds.any():
            valid_in_rival[in_bounds] = rd.valid[rr[in_bounds], rc[in_bounds]]
        contested = in_bounds & valid_in_rival
        if not contested.any():
            continue

        xc, yc = to_common.transform(xs[contested], ys[contested])
        dist_td = np.hypot(np.asarray(xc) - td.center_common[0], np.asarray(yc) - td.center_common[1])
        dist_rd = np.hypot(np.asarray(xc) - rd.center_common[0], np.asarray(yc) - rd.center_common[1])

        td_name, rd_name = td.pair.tile_name, rd.pair.tile_name
        td_loses = (dist_td > dist_rd) | ((dist_td == dist_rd) & (td_name > rd_name))

        lose_rows = rows[contested][td_loses]
        lose_cols = cols[contested][td_loses]
        keep[lose_rows, lose_cols] = False

    return keep


def build_backbone_masks(backbone: str, region: str = "london") -> dict:
    pairs = list_tile_pairs(backbone, region)
    print(f"  Lade {len(pairs)} Kacheln ({backbone})...")
    tile_datas = [_TileData(p) for p in pairs]
    candidates = _find_overlap_candidates(tile_datas)
    rivals_by_idx = {i: [] for i in range(len(tile_datas))}
    for i, j in candidates:
        rivals_by_idx[i].append(j)
        rivals_by_idx[j].append(i)
    print(f"  {len(candidates)} Kandidatenpaare (Bounding-Box-Ueberlappung, Roh-Kacheln)")

    aoi = gpd.read_file(AOI_PATH)
    out_dir = MASK_ROOT / backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    n_valid_before = 0
    n_valid_after = 0
    for i, td in enumerate(tile_datas):
        rivals = [tile_datas[j] for j in rivals_by_idx[i]]
        keep = build_tile_mask(td, rivals)

        aoi_in_crs = aoi.to_crs(td.crs)
        poly_mask = rasterize([(aoi_in_crs.geometry.iloc[0], 1)], out_shape=td.shape,
                               transform=td.transform, fill=0, dtype="uint8").astype(bool)
        final_mask = (keep & poly_mask).astype("uint8")

        n_valid_before += int(td.valid.sum())
        n_valid_after += int(final_mask.sum())

        out_path = out_dir / f"{Path(td.pair.tile_name).stem}_mask.tif"
        profile = dict(driver="GTiff", height=td.shape[0], width=td.shape[1], count=1,
                       dtype="uint8", crs=td.crs, transform=td.transform, compress="lzw",
                       tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(final_mask, 1)

    area_before_km2 = n_valid_before * PIXEL_AREA_M2 / 1e6
    area_after_km2 = n_valid_after * PIXEL_AREA_M2 / 1e6
    return {"backbone": backbone, "n_tiles": len(pairs),
            "n_valid_before": n_valid_before, "n_valid_after": n_valid_after,
            "area_before_km2": area_before_km2, "area_after_km2": area_after_km2}


def verify_union_area(backbone: str, n_valid_after: int, region: str = "london") -> dict:
    """Unabhaengige Flaechenschaetzung: Vereinigung ALLER Kachel-Footprints (Arm-C-Bounds, nicht
    Roh-Kachel-Bounds) in COMMON_CRS, geschnitten mit dem AOI-Polygon - verglichen mit der
    Pixelflaeche der maskierten Pixel. Kleine Abweichungen (Rasterdiskretisierung,
    Randeffekte) sind erwartbar, keine exakte Uebereinstimmung."""
    pairs = list_tile_pairs(backbone, region)
    aoi = gpd.read_file(AOI_PATH).to_crs(COMMON_CRS)
    footprints = []
    for p in pairs:
        with rasterio.open(p.proj_path) as src:
            b, crs = src.bounds, src.crs
        footprints.append(_reprojected_bbox_polygon(b, str(crs), COMMON_CRS))
    union_bbox = footprints[0]
    for f in footprints[1:]:
        union_bbox = union_bbox.union(f)
    union_in_aoi = union_bbox.intersection(aoi.geometry.iloc[0])
    union_area_km2 = union_in_aoi.area / 1e6
    mask_area_km2 = n_valid_after * PIXEL_AREA_M2 / 1e6
    return {"backbone": backbone, "union_bbox_intersect_aoi_km2": union_area_km2,
            "masked_pixel_area_km2": mask_area_km2,
            "ratio": mask_area_km2 / union_area_km2 if union_area_km2 > 0 else float("nan")}


def spot_check_no_double_assignment(backbone: str, n_samples: int = 500, seed: int = 0,
                                     region: str = "london") -> dict:
    """Stichprobe: zieht gueltig-maskierte Pixel aus zufaelligen Kacheln, reprojiziert ihre
    Weltposition in JEDE Kachel mit Bounding-Box-Ueberlappung und prueft, ob dort GLEICHZEITIG
    auch mask==1 steht - waere ein doppelt vergebener Weltpunkt (Bug in der Eindeutigkeitsregel)."""
    pairs = list_tile_pairs(backbone, region)
    mask_dir = MASK_ROOT / backbone
    rng = np.random.default_rng(seed)

    tile_masks = {}
    tile_meta = {}
    for p in pairs:
        mpath = mask_dir / f"{Path(p.tile_name).stem}_mask.tif"
        with rasterio.open(mpath) as src:
            tile_masks[p.tile_name] = src.read(1).astype(bool)
            tile_meta[p.tile_name] = (src.transform, src.crs, src.shape)
        with rasterio.open(p.raw_path) as src:
            tile_meta[p.tile_name] += (src.bounds,)

    footprints_common = {name: _reprojected_bbox_polygon(meta[3], str(meta[1]), COMMON_CRS)
                          for name, meta in tile_meta.items()}

    names = list(tile_masks.keys())
    per_tile_n = max(1, n_samples // len(names))
    n_checked = 0
    n_double = 0
    for name in names:
        m = tile_masks[name]
        rows, cols = np.where(m)
        if len(rows) == 0:
            continue
        take = min(per_tile_n, len(rows))
        sel = rng.choice(len(rows), size=take, replace=False)
        transform, crs, shape, bounds = tile_meta[name]
        xs, ys = rasterio.transform.xy(transform, rows[sel], cols[sel])
        xs, ys = np.asarray(xs), np.asarray(ys)

        for other in names:
            if other == name:
                continue
            if not footprints_common[name].intersects(footprints_common[other]):
                continue
            o_transform, o_crs, o_shape, _ = tile_meta[other]
            to_other = Transformer.from_crs(str(crs), str(o_crs), always_xy=True)
            xo, yo = to_other.transform(xs, ys)
            rr, rc = rasterio.transform.rowcol(o_transform, xo, yo)
            rr, rc = np.asarray(rr), np.asarray(rc)
            in_bounds = (rr >= 0) & (rr < o_shape[0]) & (rc >= 0) & (rc < o_shape[1])
            if in_bounds.any():
                double = tile_masks[other][rr[in_bounds], rc[in_bounds]]
                n_double += int(double.sum())
        n_checked += take

    return {"backbone": backbone, "n_checked": n_checked, "n_double_assigned": n_double}


def main():
    print("=== Check: Arm-C-Zuschnitt Bounding Box vs. AOI-Polygon ===")
    bbox_check = check_aoi_is_bbox_crop("tessera")
    print(f"  {bbox_check}")

    summaries = {}
    for backbone in ("tessera", "ae"):
        print(f"\n=== {backbone}: Masken bauen ===")
        t0 = time.time()
        summary = build_backbone_masks(backbone)
        summary["seconds"] = time.time() - t0
        summaries[backbone] = summary
        print(f"  {summary}")

        union_check = verify_union_area(backbone, summary["n_valid_after"])
        print(f"  [Flaechenpruefung] {union_check}")

        dbl_check = spot_check_no_double_assignment(backbone)
        print(f"  [Stichprobe Doppelvergabe] {dbl_check}")

    print("\n=== Zusammenfassung ===")
    for backbone, s in summaries.items():
        print(f"  {backbone}: gueltig vorher={s['n_valid_before']:,} ({s['area_before_km2']:.2f} km^2), "
              f"gueltig nachher={s['n_valid_after']:,} ({s['area_after_km2']:.2f} km^2)")


if __name__ == "__main__":
    main()
