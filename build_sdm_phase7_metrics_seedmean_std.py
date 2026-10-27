"""
build_sdm_phase7_metrics_seedmean_std.py

SDM-Seedmean-Auswertung mit z-standardisierten Kopf-Eingaben (2026-10-01, Jakob).
Diagnose 2026-10-01: `bsp6.fit_mlp` laeuft in 29/30 Laeufen bis MAX_EPOCHS=500, weil die
Eingabeskala zwischen den Armen ~20x differiert (TESSERA/A z-score: Std/Dim ~0,37; B/C und AE
L2-normiert: ~0,008-0,03) - der Kopf ist bei kleinskaligen Armen unterangepasst.

Fix: vor `fit_mlp` jede Eingabedimension mit Mittelwert/Std der TRAIN-LSOAs standardisieren
(fit nur auf Train, auf Val/Test angewendet), identisch fuer alle Backbones/Arme. Sonst exakt
`build_sdm_phase7_metrics_seedmean.py` (gleiche Pooled-Dateien, Splits, Seeds, Bootstrap 1000x/
Startseed 2000). Zusaetzlich Random-Embedding-Baseline mit demselben Kopf-Protokoll.
Ersetzt nichts - neue Dateien mit Suffix `_seedmean_std`.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6
import build_sdm_phase7_metrics_seedmean as bsps
import build_sdm_random_embedding_seedmean as bsre

OUT_DIR = cfg.DATA / "processed"
_fit_mlp_raw = bsp6.fit_mlp
EPOCH_LOG = []


def fit_mlp_std(X, y, split, seed, d_in, n_classes, lsoa_codes=None):
    train_m = split == "train"
    mu = X[train_m].mean(axis=0)
    sd = X[train_m].std(axis=0) + 1e-8
    Xs = ((X - mu) / sd).astype(np.float32)
    r = _fit_mlp_raw(Xs, y, split, seed, d_in, n_classes, lsoa_codes=lsoa_codes)
    EPOCH_LOG.append(r["epochs"])
    return r


bsp6.fit_mlp = fit_mlp_std


def main():
    with tempfile.TemporaryDirectory() as tmp:
        bsps.OUT_DIR = Path(tmp)
        bsps.main()
        for f in Path(tmp).glob("*.csv"):
            dst = OUT_DIR / f.name.replace("_seedmean", "_seedmean_std")
            shutil.move(f, dst)
            print(f"Verschoben: {dst}")

    print(f"\nEpochen Hauptlaeufe (30 Fits): {EPOCH_LOG}  -> am Limit {bsp6.MAX_EPOCHS}: "
          f"{sum(e >= bsp6.MAX_EPOCHS for e in EPOCH_LOG)}")

    EPOCH_LOG.clear()
    rows = []
    for backbone in bsps.BACKBONES:
        for arm in bsps.ARMS:
            row = bsre.evaluate(backbone, arm)
            rows.append(row)
            print(f"{row['model']}: kl={row['kl']:.5f} CI={row['kl_ci']}")
    out = OUT_DIR / "sdm_random_embedding_seedmean_std.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Gespeichert: {out}")
    print(f"Epochen Random-Baseline: {EPOCH_LOG}")


if __name__ == "__main__":
    main()
