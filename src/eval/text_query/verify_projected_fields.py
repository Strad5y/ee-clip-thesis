"""
verify_projected_fields.py

Phase 1 (INSTRUCTIONS_text_query.md, Entscheidung 2026-09-17): die Arm-C-256d-Felder aus
data/interim/sdm_projected/{backbone}_armc/ werden UNVERAENDERT als projizierte Embeddings
uebernommen - dieses Skript rechnet nichts neu, sondern fuehrt nur die drei geforderten
Pruefungen aus:

  1. L2-Norm der gespeicherten Vektoren liegt bei 1 +/- 1e-3 (Stichprobe).
  2. 1.000 zufaellige gueltige Pixel werden aus den ROH-Mosaiken (inkl. dequantize_ae bzw.
     Z-Score mit den echten norm_stats) ueber run_landuse_probe.py::load_trained_head/project
     neu projiziert - Kosinus zum gespeicherten Arm-C-Wert an derselben Pixelposition muss
     >= 0.9999 sein.
  3. Raeumliche Ueberlappung der Rohkachel-Footprints (v.a. an der UTM-Zonengrenze 32630/32631)
     - nur Befund + Bezug auf die bekannten Handhabungen in spatial_join.py/
     build_splits_pixels.py, KEINE eigene Entscheidung ueber Korrektur.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Repo-Root

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

import config as cfg
from dequantize import dequantize_ae
from normalize import ZSCORE, apply_zscore, load_stats
from run_landuse_probe import load_backbone_yaml_config, load_trained_head, project, resolve_normalization
from tiles import list_tile_pairs

RNG_SEED = 0
N_L2_SAMPLE_TILES = 8
N_L2_PIXELS_PER_TILE = 500
N_REPROJECT_TOTAL = 1000
COS_THRESHOLD = 0.9999
L2_TOLERANCE = 1e-3


# ---------------------------------------------------------------------------
# Check 1: L2-Norm der gespeicherten Arm-C-Vektoren
# ---------------------------------------------------------------------------

def check_l2_norm(backbone: str, pairs, rng: np.random.Generator) -> dict:
    idx_pool = rng.permutation(len(pairs))[:min(N_L2_SAMPLE_TILES, len(pairs))]
    max_dev = 0.0
    n_checked = 0
    for i in idx_pool:
        p = pairs[i]
        with rasterio.open(p.proj_path) as src:
            h, w = src.height, src.width
            ws = min(64, h, w)
            found = False
            for _attempt in range(10):
                r0 = int(rng.integers(0, h - ws + 1))
                c0 = int(rng.integers(0, w - ws + 1))
                arr = src.read(window=Window(c0, r0, ws, ws))  # (256, ws, ws)
                valid = np.isfinite(arr).all(axis=0)
                vecs = arr[:, valid].T
                if len(vecs) > 0:
                    found = True
                    break
            if not found:
                continue
        take = min(N_L2_PIXELS_PER_TILE, len(vecs))
        sub = vecs[rng.choice(len(vecs), size=take, replace=False)]
        norms = np.linalg.norm(sub, axis=1)
        max_dev = max(max_dev, float(np.abs(norms - 1.0).max()))
        n_checked += take
    ok = max_dev <= L2_TOLERANCE
    return {"backbone": backbone, "n_tiles_sampled": len(idx_pool), "n_pixels_checked": n_checked,
            "max_abs_dev_from_1": max_dev, "ok": ok}


# ---------------------------------------------------------------------------
# Check 2: Reprojektion von 1.000 zufaelligen gueltigen Pixeln gegen den Trainings-Codepfad
# ---------------------------------------------------------------------------

def _sample_valid_positions(proj_path: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    """Liest NUR Band 1 (nicht alle 256), um gueltige (row,col)-Positionen zu finden - billig
    ggue. dem Lesen aller Kanaele, und Gueltigkeit ist bandunabhaengig (ein NaN-Pixel ist in
    ALLEN 256 Kanaelen NaN, siehe project_tile())."""
    with rasterio.open(proj_path) as src:
        band1 = src.read(1)
    valid_rc = np.argwhere(np.isfinite(band1))
    if len(valid_rc) == 0:
        return np.empty((0, 2), dtype=np.int64)
    take = min(n, len(valid_rc))
    sel = rng.choice(len(valid_rc), size=take, replace=False)
    return valid_rc[sel]


def check_reprojection(backbone: str, pairs, rng: np.random.Generator, device: str = "cpu") -> dict:
    head, ckpt_cfg = load_trained_head(backbone)
    yaml_cfg = load_backbone_yaml_config(backbone)
    mode, stats_path = resolve_normalization(yaml_cfg, f"configs/{backbone}_v1.yaml")
    ckpt_mode, ckpt_stats_path = resolve_normalization(ckpt_cfg, f"Checkpoint-Config ({backbone})")
    if mode != ckpt_mode or stats_path != ckpt_stats_path:
        raise ValueError(f"{backbone}: Normalisierung config vs. checkpoint weicht ab - "
                          f"({mode!r},{stats_path!r}) vs. ({ckpt_mode!r},{ckpt_stats_path!r})")
    norm_stats = load_stats(stats_path) if mode == ZSCORE else None

    # Pixel-Budget gleichmaessig, aber proportional zur Kachelzahl auf die Kacheln verteilen -
    # danach pro Kachel exakt so viele ziehen, wie budgetiert (Rest der letzten Kachel).
    n_tiles = len(pairs)
    base = N_REPROJECT_TOTAL // n_tiles
    remainder = N_REPROJECT_TOTAL % n_tiles
    per_tile_budget = [base + (1 if i < remainder else 0) for i in range(n_tiles)]

    cosines = []
    n_sampled = 0
    for p, budget in zip(pairs, per_tile_budget):
        if budget == 0:
            continue
        positions = _sample_valid_positions(p.proj_path, budget, rng)
        if len(positions) == 0:
            continue

        # BUGFIX (nach erstem Lauf): proj_path ist NICHT pixel-index-identisch zu raw_path -
        # project_tile() schneidet vor dem Schreiben auf die AOI-Bounding-Box zu (siehe
        # build_sdm_projected_field.py::project_tile(), window = from_bounds(...)). Ein
        # (row,col) in proj_path zeigt deshalb i.A. auf ein ANDERES (row,col) in raw_path,
        # verschoben um den Fensteroffset. Fix: ueber Weltkoordinaten (proj-Transform ->
        # x,y -> raw-Transform) statt ueber identische Pixelindizes mappen. Beide Raster
        # teilen dieselbe CRS/Aufloesung (keine Resampling, nur Crop) - die Rueckrechnung ist
        # exakt, kein Interpolationsfehler.
        with rasterio.open(p.proj_path) as src:
            proj_transform = src.transform
        rows_proj = positions[:, 0]
        cols_proj = positions[:, 1]
        xs, ys = rasterio.transform.xy(proj_transform, rows_proj, cols_proj)

        with rasterio.open(p.raw_path) as src:
            raw_transform = src.transform
            raw_rows, raw_cols = rasterio.transform.rowcol(raw_transform, xs, ys)
            raw_rows = np.asarray(raw_rows)
            raw_cols = np.asarray(raw_cols)
            out_of_bounds = ((raw_rows < 0) | (raw_rows >= src.height) |
                              (raw_cols < 0) | (raw_cols >= src.width))
            if out_of_bounds.any():
                raise ValueError(
                    f"{p.tile_name}: {int(out_of_bounds.sum())} rueckgerechnete Roh-Positionen "
                    f"liegen ausserhalb der Rohkachel ({src.height}x{src.width}) - "
                    f"Transform-Rueckrechnung fehlerhaft oder Kachel-Randfall."
                )
            raw_vals = np.stack([src.read(window=Window(int(c), int(r), 1, 1))[:, 0, 0]
                                  for r, c in zip(raw_rows, raw_cols)], axis=0)  # (n, bands)
        with rasterio.open(p.proj_path) as src:
            stored_vals = np.stack([src.read(window=Window(int(c), int(r), 1, 1))[:, 0, 0]
                                     for r, c in positions], axis=0)  # (n, 256)

        x = raw_vals.astype(np.float32)
        if backbone == "ae":
            x = dequantize_ae(x)
        finite = np.isfinite(x).all(axis=-1)
        if not finite.all():
            raise ValueError(f"{p.tile_name}: {int((~finite).sum())} der als 'gueltig' aus dem "
                              f"Arm-C-Feld gezogenen Positionen sind im Roh-Mosaik NICHT endlich "
                              f"- Rohkachel und Arm-C-Feld sind an dieser Position inkonsistent.")
        if mode == ZSCORE:
            x = apply_zscore(x, norm_stats)

        fresh = project(head, x, device=device)  # (n, 256)
        num = (fresh * stored_vals).sum(axis=1)
        denom = np.linalg.norm(fresh, axis=1) * np.linalg.norm(stored_vals, axis=1)
        cos = num / denom
        cosines.append(cos)
        n_sampled += len(positions)

    all_cos = np.concatenate(cosines) if cosines else np.array([])
    min_cos = float(all_cos.min()) if len(all_cos) else float("nan")
    n_below = int((all_cos < COS_THRESHOLD).sum()) if len(all_cos) else 0
    return {"backbone": backbone, "n_pixels_checked": n_sampled, "min_cosine": min_cos,
            "mean_cosine": float(all_cos.mean()) if len(all_cos) else float("nan"),
            "n_below_threshold": n_below, "ok": n_below == 0 and n_sampled >= N_REPROJECT_TOTAL * 0.95}


# ---------------------------------------------------------------------------
# Check 3: raeumliche Ueberlappung der Rohkachel-Footprints
# ---------------------------------------------------------------------------

def check_tile_overlap(backbone: str, pairs, common_crs: str = "EPSG:32630") -> dict:
    """Footprint jeder Rohkachel (Bounding Box in ihrer eigenen CRS) nach `common_crs`
    reprojiziert, dann paarweise auf Ueberlappung geprueft - relevant v.a. fuer TESSERA, das
    Kacheln aus zwei UTM-Zonen (32630/32631) am 0-Meridian mischt (siehe spatial_join.py-
    Docstring zur bekannten POI-Doppelzuordnung)."""
    boxes = []
    for p in pairs:
        with rasterio.open(p.raw_path) as src:
            b = src.bounds
        boxes.append({"tile": p.tile_name, "crs": str(p.crs), "geom": box(b.left, b.bottom, b.right, b.top)})

    # Je nach Zonenmischung: pro CRS-Gruppe eine eigene GeoDataFrame (Footprints sind in ihrer
    # eigenen nativen CRS gebaut), dann ALLE gemeinsam nach common_crs reprojiziert.
    reproj = []
    for crs_val in sorted(set(bx["crs"] for bx in boxes)):
        group = [bx for bx in boxes if bx["crs"] == crs_val]
        g = gpd.GeoDataFrame(group, geometry="geom", crs=crs_val).to_crs(common_crs)
        reproj.append(g)
    all_tiles = gpd.GeoDataFrame(pd_concat(reproj), geometry="geom", crs=common_crs)

    overlaps = []
    n = len(all_tiles)
    total_overlap_area = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = all_tiles.iloc[i], all_tiles.iloc[j]
            if gi["geom"].intersects(gj["geom"]):
                inter = gi["geom"].intersection(gj["geom"])
                area = inter.area
                if area > 0:
                    overlaps.append({"tile_a": gi["tile"], "crs_a": gi["crs"],
                                      "tile_b": gj["tile"], "crs_b": gj["crs"],
                                      "overlap_area_m2": area})
                    total_overlap_area += area

    cross_zone = [o for o in overlaps if o["crs_a"] != o["crs_b"]]
    return {"backbone": backbone, "n_tiles": n, "n_overlapping_pairs": len(overlaps),
            "n_cross_zone_overlapping_pairs": len(cross_zone),
            "total_overlap_area_m2": total_overlap_area,
            "total_overlap_area_km2": total_overlap_area / 1e6,
            "overlaps": overlaps}


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)
    for backbone in ("tessera", "ae"):
        print(f"\n=== {backbone} ===")
        pairs = list_tile_pairs(backbone)

        r1 = check_l2_norm(backbone, pairs, rng)
        print(f"[Check 1: L2-Norm] {r1}")

        r2 = check_reprojection(backbone, pairs, rng)
        print(f"[Check 2: Reprojektions-Kosinus] {r2}")

        r3 = check_tile_overlap(backbone, pairs)
        print(f"[Check 3: Kachel-Ueberlappung] n_tiles={r3['n_tiles']} "
              f"n_overlapping_pairs={r3['n_overlapping_pairs']} "
              f"n_cross_zone_overlapping_pairs={r3['n_cross_zone_overlapping_pairs']} "
              f"total_overlap_area_km2={r3['total_overlap_area_km2']:.4f}")
        for o in r3["overlaps"][:10]:
            print(f"    {o['tile_a']} ({o['crs_a']}) <-> {o['tile_b']} ({o['crs_b']}): "
                  f"{o['overlap_area_m2']:.1f} m^2")


if __name__ == "__main__":
    main()
