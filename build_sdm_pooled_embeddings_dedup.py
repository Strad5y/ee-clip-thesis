"""
build_sdm_pooled_embeddings_dedup.py

Schritt 1 (2026-09-17, Jakob): SDM-LSOA-Poolung NEU gerechnet mit der Eindeutigkeits- + AOI-
Polygon-Maske aus der Text-Query-Aufgabe (src/eval/text_query/masks.py) - behebt die zuvor
gefundene Kachel-Ueberlappungs-Doppelzaehlung in
build_sdm_pooled_embeddings.py::pool_backbone_arm() (TESSERA: 25,8% der LSOAs mit >=2
tatsaechlich ueberlappenden Kandidaten-Kacheln betroffen; AlphaEarth: bis zu 100%
Kandidaten-Ueberlappung, siehe Chat-Bericht "check_tile_overlap_side_effects.py").

Grundsatz: jeder Weltort zaehlt in JEDEM Feature-/Auswertungsschritt genau einmal. Massgeblich
ist die Eindeutigkeitsmaske (naechster Kachelmittelpunkt gewinnt, Tie-Break Dateiname) plus die
AOI-Polygon-Maske aus Phase 2 der Text-Query-Aufgabe - NICHT neu hergeleitet, sondern
wiederverwendet (data/interim/text_query_masks/{backbone}/*_mask.tif).

ERSETZT NICHTS: build_sdm_pooled_embeddings.py und seine Outputs
(data/processed/sdm_pooled_{backbone}_arm{a,b,c}.parquet) bleiben unveraendert. Diese neuen
Ergebnisse landen unter sdm_pooled_{backbone}_arm{a,b,c}_dedup.parquet.

Arm-Grid-Unterschied: Arm B/C lesen bereits die AOI-BBox-zugeschnittenen Arm-C-Felder (Phase 1
der Text-Query-Aufgabe) - deren Grid ist PIXEL-IDENTISCH mit dem Grid, auf dem die Phase-2-Maske
gebaut wurde (beide aus demselben project_tile()-Fenster). Die Maske kann dort DIREKT angewendet
werden. Arm A liest dagegen die ROHEN, UNBESCHNITTENEN Kacheln - deren Grid ist groesser (das
AOI-Fenster ist nur ein Ausschnitt davon). Die Phase-2-Maske wird deshalb an GENAU der
Fensterposition, die project_tile() beim Erzeugen des Arm-C-Feldes fuer dieselbe Rohkachel
verwendet hat, in ein volles Rohkachel-Grid eingebettet (ausserhalb des Fensters: False/
ausgeschlossen - das ist korrekt, denn ausserhalb der AOI-Bounding-Box liegt zwangslaeufig auch
ausserhalb des AOI-Polygons). Kein erneutes Berechnen der Eindeutigkeitsregel, kein teures
Voll-Array-Lesen nur fuer die Maske.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src" / "eval" / "text_query"))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from pyproj import Transformer
from rasterio.windows import Window, from_bounds

import config as cfg
import spatial_join as sj
from build_sdm_lsoa_splits import CELL_SIZE_M, get_aoi_utm_reference, load_lsoa_geometries_for_london
from build_sdm_targets import load_ts062, CODE_COL
from build_sdm_pooled_embeddings import load_cell_lookup, raster_paths_for_arm
from dequantize import dequantize_ae
from normalize import ZSCORE, apply_zscore, load_stats
from run_landuse_probe import EE_DIMS, LATENT_DIM, load_backbone_yaml_config, resolve_normalization

SPLITS_LONDON = cfg.DATA / "splits" / "splits_london.csv"
LSOA_SPLIT_A = cfg.DATA / "splits" / "sdm_lsoa_splits_london.csv"
MASK_DIR = cfg.DATA / "interim" / "text_query_masks"
AOI_PATH = cfg.AOI_DIR / "london.gpkg"
OUT_DIR = cfg.DATA / "processed"


def _raw_tile_window(raw_path: Path, aoi: gpd.GeoDataFrame) -> Window:
    """Exakt dieselbe Fensterberechnung wie build_sdm_projected_field.py::project_tile() -
    identischer AOI-Objekt-Ursprung (data/aoi/london.gpkg), identische Rundungslogik."""
    with rasterio.open(raw_path) as src:
        aoi_geom = aoi.to_crs(src.crs).geometry.iloc[0]
        minx, miny, maxx, maxy = aoi_geom.bounds
        tb = src.bounds
        ix0, iy0 = max(minx, tb.left), max(miny, tb.bottom)
        ix1, iy1 = min(maxx, tb.right), min(maxy, tb.top)
        window = from_bounds(ix0, iy0, ix1, iy1, transform=src.transform)
        window = window.round_lengths().round_offsets()
        window = window.intersection(Window(0, 0, src.width, src.height))
    return window


def load_mask_for_tile(backbone: str, raw_path: Path, tile_path_for_arm: Path, arm: str,
                        aoi: gpd.GeoDataFrame, raw_shape: tuple) -> np.ndarray:
    """Liefert die Eindeutigkeits+AOI-Maske im Grid, das exact_extract fuer DIESEN Arm tatsaechlich
    liest (Arm A: volles Rohkachel-Grid; Arm B/C: das bereits AOI-zugeschnittene Arm-C/B-Grid,
    pixel-identisch zur gespeicherten Maske)."""
    mask_path = MASK_DIR / backbone / f"{raw_path.stem}_mask.tif"
    with rasterio.open(mask_path) as src:
        crop_mask = src.read(1).astype(bool)

    if arm != "a":
        return crop_mask  # Arm B/C: bereits dasselbe Grid wie die gespeicherte Maske

    window = _raw_tile_window(raw_path, aoi)
    full = np.zeros(raw_shape, dtype=bool)
    r0, c0 = int(round(window.row_off)), int(round(window.col_off))
    h, w = crop_mask.shape
    assert (h, w) == (int(round(window.height)), int(round(window.width))), (
        f"{raw_path.name}: Arm-C-Maskenshape {crop_mask.shape} passt nicht zum rekonstruierten "
        f"AOI-Fenster ({window.height}x{window.width}) - Fensterberechnung weicht von "
        f"build_sdm_projected_field.py ab."
    )
    full[r0:r0 + h, c0:c0 + w] = crop_mask
    return full


def pool_backbone_arm_dedup(backbone: str, arm: str, lsoa_utm: gpd.GeoDataFrame, lsoa_split: dict,
                             cell_lookup: dict, utm_crs, minx: float, miny: float,
                             aoi: gpd.GeoDataFrame) -> pd.DataFrame:
    """Wie build_sdm_pooled_embeddings.py::pool_backbone_arm(), aber jeder exact_extract-Treffer
    wird zusaetzlich gegen die Eindeutigkeits+AOI-Maske gefiltert, bevor er in die laufenden
    Summen eingeht."""
    tile_paths = raster_paths_for_arm(backbone, arm)
    raw_tiles = {p.name: p for p in sj.get_backbones("london")[backbone]}
    latent_dim = LATENT_DIM if arm in ("b", "c") else EE_DIMS[backbone]
    needs_dequant = backbone == "ae" and arm == "a"

    if arm == "a":
        yaml_cfg = load_backbone_yaml_config(backbone)
        norm_mode, stats_path = resolve_normalization(yaml_cfg, f"configs/{backbone}_v1.yaml")
        norm_stats = load_stats(stats_path) if norm_mode == ZSCORE else None
    else:
        norm_mode, norm_stats = None, None

    lsoa_by_crs = {}
    acc = {}

    def get_acc(code):
        if code not in acc:
            acc[code] = {"sum_all": np.zeros(latent_dim, dtype=np.float64), "cov_all": 0.0,
                         "sum_pure": np.zeros(latent_dim, dtype=np.float64), "cov_pure": 0.0,
                         "n_all": 0, "n_pure": 0}
        return acc[code]

    t0 = time.time()
    n_masked_out_total = 0
    n_hits_total = 0
    for ti, tp in enumerate(tile_paths):
        # Arm A: tp IST die Rohkachel. Arm B/C: tp ist das Arm-Feld (<raw_stem>_proj256.tif) -
        # raw_path ist die zugehoerige Rohkachel (fuer den Masken-Dateinamen, gleicher Stem).
        raw_stem = tp.stem if arm == "a" else tp.stem.replace("_proj256", "")
        raw_path = tp if arm == "a" else next(p for p in raw_tiles.values() if p.stem == raw_stem)

        with rasterio.open(tp) as src:
            tile_crs = str(src.crs)
            width = src.width
            transform = src.transform
            bounds = src.bounds
            tile_shape = src.shape

        if tile_crs not in lsoa_by_crs:
            lsoa_by_crs[tile_crs] = lsoa_utm.to_crs(tile_crs) if tile_crs != str(utm_crs) else lsoa_utm
        lsoa_native = lsoa_by_crs[tile_crs]
        cand = lsoa_native.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
        if len(cand) == 0:
            continue

        res = exact_extract(str(tp), cand[["lsoa_code", "geometry"]], ["cell_id", "coverage"],
                             include_cols=["lsoa_code"], output="pandas")
        if len(res) == 0:
            continue

        mask = load_mask_for_tile(backbone, raw_path, tp, arm, aoi, tile_shape)

        with rasterio.open(tp) as src:
            arr = src.read()
        bands = arr.shape[0]
        if needs_dequant:
            arr = np.moveaxis(dequantize_ae(np.moveaxis(arr, 0, -1)), -1, 0)

        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        reproject = tile_crs != str(utm_crs)
        transformer = Transformer.from_crs(tile_crs, str(utm_crs), always_xy=True) if reproject else None

        for _, row in res.iterrows():
            cell_ids = np.asarray(row["band_1_cell_id"])
            coverages = np.asarray(row["band_1_coverage"], dtype=np.float64)
            if len(cell_ids) == 0:
                continue
            rows_ = cell_ids // width
            cols_ = cell_ids % width
            n_hits_total += len(rows_)

            keep_mask = mask[rows_, cols_]
            n_masked_out_total += int((~keep_mask).sum())
            if not keep_mask.any():
                continue
            rows_, cols_, coverages = rows_[keep_mask], cols_[keep_mask], coverages[keep_mask]

            emb = arr[:, rows_, cols_].T.astype(np.float64)
            valid = np.isfinite(emb).all(axis=1)
            if not valid.any():
                continue
            emb, coverages, rows_, cols_ = emb[valid], coverages[valid], rows_[valid], cols_[valid]

            if norm_mode == ZSCORE:
                emb = apply_zscore(emb.astype(np.float32), norm_stats).astype(np.float64)

            xs = a * (cols_ + 0.5) + b * (rows_ + 0.5) + c
            ys = d * (cols_ + 0.5) + e * (rows_ + 0.5) + f
            if reproject:
                xs, ys = transformer.transform(xs, ys)
            cell_col = np.floor((xs - minx) / CELL_SIZE_M).astype(int)
            cell_row = np.floor((ys - miny) / CELL_SIZE_M).astype(int)
            cell_splits = np.array([cell_lookup.get((r, c)) for r, c in zip(cell_row, cell_col)])

            a_ = get_acc(row["lsoa_code"])
            a_["sum_all"] += (coverages[:, None] * emb).sum(axis=0)
            a_["cov_all"] += coverages.sum()
            a_["n_all"] += len(coverages)

            pure_mask = cell_splits == lsoa_split.get(row["lsoa_code"])
            if pure_mask.any():
                a_["sum_pure"] += (coverages[pure_mask, None] * emb[pure_mask]).sum(axis=0)
                a_["cov_pure"] += coverages[pure_mask].sum()
                a_["n_pure"] += int(pure_mask.sum())
        del arr
        if (ti + 1) % 10 == 0 or ti == len(tile_paths) - 1:
            print(f"    [{backbone}/{arm}] Kachel {ti + 1}/{len(tile_paths)} verarbeitet "
                  f"({time.time() - t0:.1f}s)")

    print(f"  [{backbone}/{arm}] Maskierung: {n_masked_out_total}/{n_hits_total} exact_extract-Treffer "
          f"ausgeschlossen ({100*n_masked_out_total/max(n_hits_total,1):.2f}%)")

    rows = []
    for code, a_ in acc.items():
        rec = {"lsoa_code": code, "n_pixels_all": a_["n_all"], "n_pixels_pure": a_["n_pure"]}
        if a_["cov_all"] > 0:
            vec = a_["sum_all"] / a_["cov_all"]
            for i in range(latent_dim):
                rec[f"all_{i}"] = vec[i]
        if a_["cov_pure"] > 0:
            vec = a_["sum_pure"] / a_["cov_pure"]
            for i in range(latent_dim):
                rec[f"pure_{i}"] = vec[i]
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"  [{backbone}/{arm}] fertig: {len(df)} LSOAs erreicht in {time.time() - t0:.1f}s")
    return df


def main():
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    aoi_gpd = gpd.read_file(AOI_PATH)
    ts062 = load_ts062()
    ts062_codes = set(ts062[CODE_COL])
    lsoa_national = load_lsoa_geometries_for_london(aoi)
    lsoa_utm = lsoa_national[lsoa_national["LSOA21CD"].isin(ts062_codes)].copy()
    lsoa_utm = lsoa_utm.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]].to_crs(utm_crs)
    print(f"LSOAs (gematcht): {len(lsoa_utm)}")

    split_a = pd.read_csv(LSOA_SPLIT_A)
    lsoa_split = dict(zip(split_a["lsoa_code"], split_a["split"]))
    cell_lookup = load_cell_lookup()

    results = {}
    for backbone in ["ae", "tessera"]:
        for arm in ["a", "b", "c"]:
            print(f"\n=== {backbone} / Arm {arm.upper()} (dedupliziert) ===")
            df = pool_backbone_arm_dedup(backbone, arm, lsoa_utm, lsoa_split, cell_lookup,
                                          utm_crs, minx, miny, aoi_gpd)
            n_zero_all = int((df["n_pixels_all"] == 0).sum()) if len(df) else len(lsoa_utm)
            n_zero_pure = int((df["n_pixels_pure"] == 0).sum()) if len(df) else len(lsoa_utm)
            n_missing = len(lsoa_utm) - len(df)
            print(f"  LSOAs ganz ohne Treffer: {n_missing}")
            print(f"  LSOAs mit 0 Pixeln (alle Pixel im Polygon): {n_zero_all}")
            print(f"  LSOAs mit 0 Pixeln (partitionsrein): {n_zero_pure}")

            out_path = OUT_DIR / f"sdm_pooled_{backbone}_arm{arm}_dedup.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  Gespeichert: {out_path}")
            results[(backbone, arm)] = df.set_index("lsoa_code")[["n_pixels_all", "n_pixels_pure"]]

        print(f"\n=== {backbone}: Pixelzahl-Konsistenz ueber Arme A/B/C ===")
        idx_common = results[(backbone, "a")].index
        for other in ("b", "c"):
            idx_common = idx_common.intersection(results[(backbone, other)].index)
        print(f"  LSOAs mit Treffern in ALLEN drei Armen: {len(idx_common)}")
        a_all = results[(backbone, "a")].loc[idx_common, "n_pixels_all"]
        b_all = results[(backbone, "b")].loc[idx_common, "n_pixels_all"]
        c_all = results[(backbone, "c")].loc[idx_common, "n_pixels_all"]
        mismatch_ab = int((a_all != b_all).sum())
        mismatch_ac = int((a_all != c_all).sum())
        mismatch_bc = int((b_all != c_all).sum())
        print(f"  n_pixels_all - Mismatches A vs B: {mismatch_ab}, A vs C: {mismatch_ac}, B vs C: {mismatch_bc}")
        if mismatch_ab or mismatch_ac or mismatch_bc:
            diff = pd.DataFrame({"a": a_all, "b": b_all, "c": c_all})
            diff = diff[(diff["a"] != diff["b"]) | (diff["a"] != diff["c"])]
            print(diff.head(20).to_string())


if __name__ == "__main__":
    main()
