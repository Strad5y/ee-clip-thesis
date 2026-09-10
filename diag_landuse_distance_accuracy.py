"""
diag_landuse_distance_accuracy.py

Diagnose (Nutzeranfrage): verteilt sich die Land-Use-Probe-Testgenauigkeit systematisch ueber
die Distanz zum naechsten Trainpixel (Arm A roh vs. Arm C trainiert projiziert)? Reproduziert
dafuer EXAKT die Fitting-Prozedur aus run_landuse_probe.py (load_clean-Filter, dieselbe
Normalisierung, derselbe C-Grid, derselbe Checkpoint) - keine neue Methodik, nur die dort
NICHT persistierten pred_test-Vorhersagen werden reproduziert, um sie mit echten
Pixel-Weltkoordinaten (aus den Kachel-Affintransformationen, alle nach EPSG:32630
vereinheitlicht) zu verknuepfen.

compute_world_xy() ist bewusst als eigenstaendige, wiederverwendbare Funktion gehalten - genau
diese Pixel<->Koordinate-Verknuepfung wird fuer den kachelbasierten (nicht LSOA- oder
pixelbasierten) Bootstrap in Phase 7 der SDM-Auswertung wieder gebraucht.

Befund (8.9.2026, TESSERA + AlphaEarth, siehe MANIFEST.md fuer die volle Tabelle): Genauigkeit
ist zwischen 0-500m flach (kein Nachbarschafts-/Leakage-Gradient), faellt aber jenseits von
500m deutlich ab (~12-13 Punkte, beide Backbones, beide Arme, nur ~2% der Testpixel betroffen)
- eher eine Eigenschaft dieser kleinen, vermutlich randlagen-bedingten Teilmenge als ein
generischer Distanzeffekt.
"""
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

import config as cfg
import spatial_join as sj
from eval_landuse import linear_probe
from normalize import ZSCORE, apply_zscore, load_stats
from run_landuse_probe import (
    EXCLUDED_CLASS, load_backbone_yaml_config, load_trained_head, project,
    resolve_normalization,
)

C_GRID = (0.1, 1.0, 10.0)
DIST_BINS = [0, 50, 100, 250, 500, np.inf]
DIST_LABELS = ["0-50m", "50-100m", "100-250m", "250-500m", ">500m"]


def load_clean_with_coords(name: str) -> pd.DataFrame:
    path = cfg.DATA / "processed" / f"landuse_pairs_{name}.parquet"
    df = pd.read_parquet(path, columns=["tile", "row", "col", "code_2021", "split", "embedding", "utm_zone"])
    n_before = len(df)
    df = df[(df["code_2021"] != EXCLUDED_CLASS) & (df["split"] != "missing")].reset_index(drop=True)
    print(f"  {name}: {n_before} -> {len(df)} Zeilen nach Filter (Klasse {EXCLUDED_CLASS} + split=missing)")
    return df


def compute_world_xy(df: pd.DataFrame, backbone: str) -> np.ndarray:
    """Weltkoordinaten je Pixel (Zeilenmitte, wie aggregate_pixel_pairs.py::compute_pixel_cells),
    alle nach EPSG:32630 reprojiziert (30N-Tiles sind schon dort, 31N-Tiles werden transformiert).
    Wiederverwendbar fuer den kachelbasierten Bootstrap (Phase 7, INSTRUCTIONS_sdm.md) - dort
    ist die 1-km-Zelle die Cluster-Einheit, deren Zuordnung genau diese Koordinaten braucht."""
    tile_paths = {p.name: p for p in sj.get_backbones("london")[backbone]}
    xs = np.empty(len(df), dtype=np.float64)
    ys = np.empty(len(df), dtype=np.float64)
    crs_by_tile = {}
    for tile_name, idx in df.groupby("tile").groups.items():
        idx = np.asarray(idx)
        tp = tile_paths[tile_name]
        with rasterio.open(tp) as src:
            transform = src.transform
            tile_crs = str(src.crs)
        crs_by_tile[tile_name] = tile_crs
        rows = df["row"].to_numpy()[idx]
        cols = df["col"].to_numpy()[idx]
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        xs[idx] = a * (cols + 0.5) + b * (rows + 0.5) + c
        ys[idx] = d * (cols + 0.5) + e * (rows + 0.5) + f

    # Zonenweise nach EPSG:32630 reprojizieren
    target = "EPSG:32630"
    out_x, out_y = xs.copy(), ys.copy()
    for tile_name, tile_crs in crs_by_tile.items():
        if tile_crs == target:
            continue
        mask = (df["tile"] == tile_name).to_numpy()
        transformer = Transformer.from_crs(tile_crs, target, always_xy=True)
        tx, ty = transformer.transform(xs[mask], ys[mask])
        out_x[mask] = tx
        out_y[mask] = ty
    return np.column_stack([out_x, out_y])


def fit_and_predict(emb: np.ndarray, y: np.ndarray, split: np.ndarray, c_grid=C_GRID):
    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    X_train, y_train = emb[train_mask], y[train_mask]
    X_val, y_val = emb[val_mask], y[val_mask]
    X_test, y_test = emb[test_mask], y[test_mask]

    best = None
    for C in c_grid:
        res = linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, C=C, max_iter=1000, return_fit=True)
        if best is None or res["val_macro_f1"] > best["val_macro_f1"]:
            best = res
    print(f"    gewaehltes C={best['C']}, val_macro_f1={best['val_macro_f1']:.4f}, "
          f"test_macro_f1={best['test_macro_f1']:.4f}, test_oa={best['test_oa']:.4f}")
    return best["pred_test"], y_test, test_mask


def run_for_backbone(name: str):
    print(f"\n{'='*20} {name} {'='*20}")
    df = load_clean_with_coords(name)
    emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    y = df["code_2021"].to_numpy()
    split = df["split"].to_numpy()

    print("  Berechne Weltkoordinaten...")
    xy = compute_world_xy(df, name)

    yaml_cfg = load_backbone_yaml_config(name)
    yaml_mode, yaml_stats_path = resolve_normalization(yaml_cfg, f"configs/{name}_v1.yaml")
    if yaml_mode == ZSCORE:
        stats = load_stats(yaml_stats_path)
        emb_raw = apply_zscore(emb, stats)
    else:
        emb_raw = emb

    train_xy = xy[split == "train"]
    print(f"  KDTree ueber {len(train_xy)} Trainpixel...")
    tree = cKDTree(train_xy)

    test_mask_global = split == "test"
    test_xy = xy[test_mask_global]
    dist, _ = tree.query(test_xy, k=1)
    print(f"  Distanz-Stichprobe: min={dist.min():.1f} median={np.median(dist):.1f} max={dist.max():.1f}")

    results = {}
    print("  --- Arm A (roh) ---")
    pred_a, y_test, _ = fit_and_predict(emb_raw, y, split)
    results["A (roh)"] = pred_a

    print("  --- Arm C (trainiert projiziert) ---")
    trained_head, ckpt_cfg = load_trained_head(name)
    emb_proj = project(trained_head, emb_raw)
    pred_c, y_test_c, _ = fit_and_predict(emb_proj, y, split)
    assert np.array_equal(y_test, y_test_c)
    results["C (trainiert)"] = pred_c

    bin_idx = np.digitize(dist, DIST_BINS[1:-1])  # 0..len(DIST_LABELS)-1

    print(f"\n  === {name}: Accuracy je Distanzklasse ===")
    header = f"  {'Klasse':<12}{'n':>10}" + "".join(f"{arm:>18}" for arm in results)
    print(header)
    for b, label in enumerate(DIST_LABELS):
        mask = bin_idx == b
        n = int(mask.sum())
        row = f"  {label:<12}{n:>10}"
        for arm, pred in results.items():
            if n == 0:
                row += f"{'n/a':>18}"
            else:
                acc = float((pred[mask] == y_test[mask]).mean())
                row += f"{acc:>18.4f}"
        print(row)

    overall_row = f"  {'GESAMT':<12}{len(y_test):>10}"
    for arm, pred in results.items():
        overall_row += f"{float((pred == y_test).mean()):>18.4f}"
    print(overall_row)


if __name__ == "__main__":
    for backbone in ["tessera", "ae"]:
        run_for_backbone(backbone)
