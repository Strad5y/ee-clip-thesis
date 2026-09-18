"""
build_sdm_pooled_embeddings_reversed_dedup.py

Folgetest (2026-09-18, Jakob) zur SDM-Pooling-Hypothese: baut Arm-C-LSOA-Embeddings in
UMGEKEHRTER Reihenfolge - erst coverage-gewichtet ueber die rohen (Z-Score-normierten, wo
zutreffend) Pixel mitteln, DANN den trainierten Kopf auf den gepoolten Vektor anwenden
(head(mean(x)) statt der bisherigen Reihenfolge mean(head(x))). Der Kopf endet selbst mit
L2-Normalisierung (model.py::ProjectionHead.forward) - kein separater Normalisierungsschritt
noetig.

Nutzt dafuer DIREKT die bereits berechneten Arm-A-Poolungen aus
build_sdm_pooled_embeddings_dedup.py (sdm_pooled_{backbone}_arma_dedup.parquet) - das SIND
bereits genau die coverage-gewichteten Mittel ueber die rohen (normierten) Pixel, mit
derselben Eindeutigkeits+AOI-Maske und denselben LSOAs wie der Dedup-Lauf. Keine Rasterdaten
werden neu gelesen - nur eine MLP-Vorwaertsrechnung auf 4.994 bereits gepoolten Vektoren.

Ersetzt nichts - neuer Output sdm_pooled_{backbone}_armc_reversed_dedup.parquet.
"""
import numpy as np
import pandas as pd

import config as cfg
from run_landuse_probe import load_trained_head, project

OUT_DIR = cfg.DATA / "processed"


def build_reversed(backbone: str) -> pd.DataFrame:
    arma = pd.read_parquet(OUT_DIR / f"sdm_pooled_{backbone}_arma_dedup.parquet")
    head, ckpt_cfg = load_trained_head(backbone)
    print(f"  {backbone}: {len(arma)} LSOAs aus Arm-A-Dedup-Poolung geladen")

    out = arma[["lsoa_code", "n_pixels_all", "n_pixels_pure"]].copy()

    for prefix, count_col in (("all", "n_pixels_all"), ("pure", "n_pixels_pure")):
        cols = [c for c in arma.columns if c.startswith(f"{prefix}_")]
        valid = arma[count_col] > 0
        n_valid = int(valid.sum())
        print(f"  {backbone}/{prefix}: {n_valid}/{len(arma)} LSOAs mit >0 Pixeln")
        if n_valid == 0:
            continue
        X_mean_raw = arma.loc[valid, cols].to_numpy(dtype=np.float32)
        X_proj = project(head, X_mean_raw)  # head(mean(x)) - Kopf endet bereits mit L2-normalize
        for i in range(X_proj.shape[1]):
            out.loc[valid, f"{prefix}_{i}"] = X_proj[:, i]

    return out


def main():
    for backbone in ("ae", "tessera"):
        print(f"\n=== {backbone}: Arm C, umgekehrte Reihenfolge (head(mean(x))) ===")
        df = build_reversed(backbone)
        out_path = OUT_DIR / f"sdm_pooled_{backbone}_armc_reversed_dedup.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {len(df)} LSOAs, gespeichert: {out_path}")


if __name__ == "__main__":
    main()
