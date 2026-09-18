"""
check_sdm_reversed_cosine.py

Folgetest (2026-09-18, Jakob), Schritt 3: wie stark unterscheiden sich die beiden Arm-C-
Varianten (bisherige Reihenfolge mean(head(x)) vs. umgekehrte Reihenfolge head(mean(x)))
pro LSOA? Kosinus-Aehnlichkeit, Minimum/Median/5.-Perzentil, getrennt nach Backbone und
Poolungsvariante (alle Pixel / partitionsrein). Nur lesen/berichten.
"""
import numpy as np
import pandas as pd

import config as cfg

OUT_DIR = cfg.DATA / "processed"


def cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / denom


def main():
    for backbone in ("ae", "tessera"):
        print(f"\n=== {backbone} ===")
        orig = pd.read_parquet(OUT_DIR / f"sdm_pooled_{backbone}_armc_dedup.parquet")
        rev = pd.read_parquet(OUT_DIR / f"sdm_pooled_{backbone}_armc_reversed_dedup.parquet")
        merged = orig.merge(rev, on="lsoa_code", suffixes=("_orig", "_rev"))
        print(f"  {len(merged)} gemeinsame LSOAs")

        for prefix in ("all", "pure"):
            cols_orig = [c for c in orig.columns if c.startswith(f"{prefix}_")]
            # merge() haengt _orig/_rev nur an KOLLIDIERENDE Spaltennamen - beide Frames
            # haben identische all_i/pure_i-Namen, also kollidieren ALLE dieser Spalten.
            cols_o = [f"{c}_orig" for c in cols_orig]
            cols_r = [f"{c}_rev" for c in cols_orig]
            valid = merged[[f"n_pixels_{prefix}_orig"]].to_numpy().ravel() > 0
            if not valid.any():
                print(f"  {prefix}: keine LSOAs mit >0 Pixeln")
                continue
            A = merged.loc[valid, cols_o].to_numpy(dtype=np.float64)
            B = merged.loc[valid, cols_r].to_numpy(dtype=np.float64)
            cos = cosine_per_row(A, B)
            print(f"  {prefix} ({valid.sum()} LSOAs): "
                  f"min={cos.min():.4f} p5={np.percentile(cos, 5):.4f} "
                  f"median={np.median(cos):.4f} mean={cos.mean():.4f} max={cos.max():.4f}")

            out = merged.loc[valid, ["lsoa_code"]].copy()
            out["cosine"] = cos
            out_path = OUT_DIR / f"diag_sdm_reversed_cosine_{backbone}_{prefix}.csv"
            out.to_csv(out_path, index=False)
            print(f"    Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
