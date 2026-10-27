"""
build_sdm_random_embedding_seedmean.py

Random-Embedding-Baseline (AETHERs "Random"-Zeile, Tab. 1: zufaellig initialisierte Embeddings
+ trainierter MLP-Kopf) auf die kanonische `_seedmean`-Basis heben (2026-09-30, Jakob).
Die bisherigen Werte in `sdm_phase7_metrics_dedup.csv` (`bsp7.evaluate_random_embedding_baseline()`)
haben die alte Inkonsistenz: Punktwert = 5-Seed-Mittel, CI nur aus Seed 0. Hier: Punktwert UND
CI als 5-Seed-Mittel, exakt wie `build_sdm_phase7_metrics_seedmean.py` (gleiche Seeds, gleicher
Zellebenen-Bootstrap 1000x/Startseed 2000, gleiche Pooled-Dateien fuer LSOA-Menge/Reihenfolge).

Pro Seed s: X_rand ~ N(0,1) mit rng=default_rng(s), dimensionsangepasst an den jeweiligen Arm
(AE/A 64, TESSERA/A 128, B/C 256), dann bsp6.fit_mlp(seed=s) - identisch zu Phase 6/7.
Ersetzt nichts - neue Datei `sdm_random_embedding_seedmean.csv`.
"""
import numpy as np
import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6
import build_sdm_phase7_metrics as bsp7
import build_sdm_phase7_metrics_seedmean as bsps

OUT_PATH = cfg.DATA / "processed" / "sdm_random_embedding_seedmean.csv"


def evaluate(backbone: str, arm: str) -> dict:
    bsp6.load_pooled = bsps.load_pooled_variant
    X, y, split_arr, n_classes, lsoa_codes = bsp6.build_xy(backbone, arm, bsps.SCENARIO)
    d_in = X.shape[1]
    per_seed = {m: [] for m in bsps.METRICS}
    ref_lsoa = None
    for s in bsp7.SEEDS:
        rng = np.random.default_rng(s)
        X_rand = rng.standard_normal((len(y), d_in)).astype(np.float32)
        r = bsp6.fit_mlp(X_rand, y, split_arr, s, d_in, n_classes, lsoa_codes=lsoa_codes)
        if ref_lsoa is None:
            ref_lsoa = r["test_lsoa"]
        else:
            assert list(r["test_lsoa"]) == list(ref_lsoa)
        m = bsp7.per_lsoa_metrics(r["y_test"], r["test_pred"])
        for metric in bsps.METRICS:
            per_seed[metric].append(m[metric])
    cell_ids = np.array([bsp7.LSOA_TO_CELL[c] for c in ref_lsoa])
    row = {"model": f"Random-Embedding ({backbone}/{arm}-dim)", "backbone": backbone, "arm": arm,
           "d_in": d_in, "n_test_lsoa": len(ref_lsoa)}
    for metric in bsps.METRICS:
        row[metric] = bsps.point_seedmean(per_seed[metric])
        row[f"{metric}_ci"] = bsp7.ci(bsps.cluster_bootstrap_seedmean(per_seed[metric], cell_ids))
    return row


def main():
    rows = []
    for backbone in bsps.BACKBONES:
        for arm in bsps.ARMS:
            row = evaluate(backbone, arm)
            rows.append(row)
            print(f"{row['model']} (d={row['d_in']}, n_test={row['n_test_lsoa']}): " + ", ".join(
                f"{m}={row[m]:.5f} CI={tuple(round(v, 5) for v in row[f'{m}_ci'])}" for m in bsps.METRICS))
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
