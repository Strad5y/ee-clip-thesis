"""
diag_geo_alignment.py

Diagnose-Aufgabe (2026-09-18, Jakob), Teil 1: prueft, ob TESSERAs schwaechere trainierte-Kopf-
Performance (SDM: Arm A roh schlaegt Arm C, umgekehrt zu AlphaEarth) an einem Geo-Alignment-
Problem liegt.

1a. Versatztest: Land-Use-Probe (Arm A, roh, eingefroren) mit Labels, die um dy,dx in
    {-3..+3} Pixel VERSETZT sind (Embedding bei (row,col), Label bei (row+dy,col+dx),
    SELBE Kachel). Erwartung: Genauigkeit maximal bei (0,0). C fest aus dem bestehenden
    "roh"-Ergebnis (keine neue C-Suche pro Versatz - das wuerde 49x mehr Fits kosten, ohne
    die Richtungsfrage zu beantworten). Aus Rechenzeitgruenden Train auf eine feste Stichprobe
    subsampled (gleiche Positionen ueber alle Versaetze), Test bleibt die volle verfuegbare
    Menge je Versatz (guenstig, nur .predict()).
1b. UTM-Zonen-Konsistenz: fuer jede Rohkachel - stimmt ihr tatsaechliches CRS mit dem ueberein,
    was assign_utm_zone() fuer eine POI an ihrem eigenen Kachelzentrum annehmen wuerde
    (Laenge < 0 Grad -> 30N/32630, sonst 31N/32631)?

Nur lesen/berichten, kein Neutraining der CLIP-/SigLIP-Gewichte, keine neuen Checkpoints.
"""
import itertools
import time

import numpy as np
import pandas as pd
import pyproj
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
import spatial_join as sj
from normalize import ZSCORE, apply_zscore, load_stats
from run_landuse_probe import EXCLUDED_CLASS, load_backbone_yaml_config, resolve_normalization

OFFSETS = list(range(-3, 4))
TRAIN_SUBSAMPLE = 100_000
TEST_SUBSAMPLE = 50_000
SEED = 0

# Fixe C je Backbone - aus dem bestehenden "roh"-Ergebnis (landuse_probe_results_tessera_dedup.csv
# / landuse_probe_results_ae*.csv), NICHT neu gesucht (siehe Docstring).
FIXED_C = {"tessera": 10.0, "ae": 10.0}


def load_labeled_pixels(backbone: str) -> pd.DataFrame:
    path = cfg.DATA / "processed" / f"landuse_pairs_{backbone}.parquet"
    df = pd.read_parquet(path, columns=["tile", "row", "col", "embedding", "code_2021", "split"])
    n_before = len(df)
    df = df[(df["code_2021"] != EXCLUDED_CLASS) & (df["split"] != "missing")].reset_index(drop=True)
    print(f"  {backbone}: {n_before} -> {len(df)} Zeilen nach Filter")
    return df


def build_label_lookup(df: pd.DataFrame) -> dict:
    """tile -> pd.Series, indiziert ueber (row,col), Wert = code_2021 - fuer Versatz-Lookups."""
    lookup = {}
    for tile, g in df.groupby("tile"):
        lookup[tile] = g.set_index(["row", "col"])["code_2021"]
    return lookup


def precompute_tile_groups(df: pd.DataFrame) -> list:
    """EINMAL berechnet (nicht pro Versatz) - (tile, positional_idx, row_array, col_array) je
    Kachel, damit shifted_labels() nicht 49x neu gruppieren muss."""
    groups = []
    row_arr = df["row"].to_numpy()
    col_arr = df["col"].to_numpy()
    for tile, g in df.groupby("tile"):
        pos_idx = df.index.get_indexer(g.index)  # positionale Indizes in df (0..len(df)-1)
        groups.append((tile, pos_idx, row_arr[pos_idx], col_arr[pos_idx]))
    return groups


def shifted_labels(tile_groups: list, lookup: dict, dy: int, dx: int, n: int) -> np.ndarray:
    """Fuer jede Zeile (ueber die vorberechneten Kachel-Gruppen): Label an (row+dy, col+dx)
    DERSELBEN Kachel, falls vorhanden - sonst NaN (Zeile faellt fuer diesen Versatz raus)."""
    out = np.full(n, np.nan, dtype=object)
    for tile, pos_idx, rows_, cols_ in tile_groups:
        s = lookup.get(tile)
        if s is None:
            continue
        idx = pd.MultiIndex.from_arrays([rows_ + dy, cols_ + dx])
        out[pos_idx] = s.reindex(idx).to_numpy()
    return out


def offset_test(backbone: str, rng: np.random.Generator) -> pd.DataFrame:
    print(f"\n=== Versatztest: {backbone} ===")
    df = load_labeled_pixels(backbone)
    lookup = build_label_lookup(df)
    tile_groups = precompute_tile_groups(df)

    emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    yaml_cfg = load_backbone_yaml_config(backbone)
    mode, stats_path = resolve_normalization(yaml_cfg, f"configs/{backbone}_v1.yaml")
    if mode == ZSCORE:
        stats = load_stats(stats_path)
        emb = apply_zscore(emb, stats)

    train_idx_all = np.where(df["split"].to_numpy() == "train")[0]
    test_idx_all = np.where(df["split"].to_numpy() == "test")[0]
    train_sub = rng.choice(train_idx_all, size=min(TRAIN_SUBSAMPLE, len(train_idx_all)), replace=False)
    test_sub = rng.choice(test_idx_all, size=min(TEST_SUBSAMPLE, len(test_idx_all)), replace=False)
    print(f"  Train-Stichprobe: {len(train_sub)} / Test-Stichprobe: {len(test_sub)} (fix ueber alle Versaetze)")

    rows = []
    t0 = time.time()
    for dy, dx in itertools.product(OFFSETS, OFFSETS):
        y_shift = shifted_labels(tile_groups, lookup, dy, dx, len(df))

        tr_valid = pd.notna(y_shift[train_sub])
        te_valid = pd.notna(y_shift[test_sub])
        tr_idx = train_sub[tr_valid]
        te_idx = test_sub[te_valid]
        if len(tr_idx) < 100 or len(te_idx) < 100:
            rows.append({"dy": dy, "dx": dx, "oa": np.nan, "macro_f1": np.nan,
                         "n_train": len(tr_idx), "n_test": len(te_idx)})
            continue

        X_tr, y_tr = emb[tr_idx], y_shift[tr_idx]
        X_te, y_te = emb[te_idx], y_shift[te_idx]

        clf = LogisticRegression(C=FIXED_C[backbone], max_iter=1000)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        oa = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro", zero_division=0)
        rows.append({"dy": dy, "dx": dx, "oa": oa, "macro_f1": f1,
                     "n_train": len(tr_idx), "n_test": len(te_idx)})
        print(f"    dy={dy:+d} dx={dx:+d}: oa={oa:.4f} macro_f1={f1:.4f} "
              f"(n_train={len(tr_idx)}, n_test={len(te_idx)}) [{time.time()-t0:.0f}s]")

    return pd.DataFrame(rows)


def utm_zone_consistency_check() -> pd.DataFrame:
    print("\n=== UTM-Zonen-Konsistenz je Rohkachel ===")
    rows = []
    zone_expect = {"EPSG:32630": ("30N", lambda lon: lon < 0), "EPSG:32631": ("31N", lambda lon: lon >= 0)}
    for backbone in ("tessera", "ae"):
        for tile_path in sj.get_backbones("london")[backbone]:
            with rasterio.open(tile_path) as src:
                crs = str(src.crs)
                b = src.bounds
                cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
            to_wgs84 = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = to_wgs84.transform(cx, cy)
            expect = zone_expect.get(crs)
            consistent = expect[1](lon) if expect else None
            rows.append({"backbone": backbone, "tile": tile_path.name, "crs": crs,
                         "centroid_lon": lon, "centroid_lat": lat,
                         "expected_zone": expect[0] if expect else "?",
                         "consistent": consistent})
    df = pd.DataFrame(rows)
    n_bad = int((df["consistent"] == False).sum())  # noqa: E712
    print(f"  {len(df)} Kacheln geprueft, {n_bad} Abweichungen (CRS passt nicht zur eigenen "
          f"Zentroid-Laenge)")
    if n_bad > 0:
        print(df[df["consistent"] == False].to_string(index=False))  # noqa: E712
    else:
        print("  Keine Abweichung - jede Kachel liegt geografisch in der Zone, die ihr CRS behauptet.")
    return df


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    for backbone in ("tessera", "ae"):
        results[backbone] = offset_test(backbone, rng)
        out_path = cfg.DATA / "processed" / f"diag_offset_test_{backbone}.csv"
        results[backbone].to_csv(out_path, index=False)
        print(f"  Gespeichert: {out_path}")

        pivot_f1 = results[backbone].pivot(index="dy", columns="dx", values="macro_f1")
        print(f"\n  {backbone} - Macro-F1-Matrix (Zeile=dy, Spalte=dx):")
        print(pivot_f1.round(4).to_string())
        best = results[backbone].loc[results[backbone]["macro_f1"].idxmax()]
        print(f"  Maximum bei dy={best['dy']:.0f}, dx={best['dx']:.0f} (macro_f1={best['macro_f1']:.4f})")

    zone_df = utm_zone_consistency_check()
    zone_df.to_csv(cfg.DATA / "processed" / "diag_utm_zone_consistency.csv", index=False)


if __name__ == "__main__":
    main()
