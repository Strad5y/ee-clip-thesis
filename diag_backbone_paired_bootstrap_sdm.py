"""
diag_backbone_paired_bootstrap_sdm.py  (2026-09-22, Jakob) - NUR DIAGNOSE

Gepaarter Zellebenen-Bootstrap fuer die BACKBONE-DIFFERENZ (TESSERA-dedup minus AE-dedup), SDM,
Arm A (roh) und Arm C (trainiert, jeweils v1dedup-Checkpoint). Beide Backbones haben exakt
dieselben 417 Test-LSOAs (verifiziert unten) - in jedem Resample wird DIESELBE gezogene 1-km-
Zellmenge (LSOA -> Zelle ueber bsp7.LSOA_TO_CELL, Mehrheitsregel wie ueberall sonst in Phase 7)
auf beide Backbones angewendet.

Wiederverwendet bsp6.fit_mlp()/bsp7.per_lsoa_metrics() UNVERAENDERT, Seed 0 (derselbe
Referenz-Seed wie im bestehenden Arm-A-vs-C-Vergleich pro Backbone). Schreibt NUR eine
Diagnosedatei, ueberschreibt keine bestehende Ergebnisdatei.
"""
import numpy as np
import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6
import build_sdm_phase7_metrics as bsp7

POOLED = {
    ("tessera", "a"): cfg.DATA / "processed" / "sdm_pooled_tessera_arma_dedup.parquet",
    ("tessera", "c"): cfg.DATA / "processed" / "sdm_pooled_tessera_armc_v1dedup_ckpt.parquet",
    ("ae", "a"): cfg.DATA / "processed" / "sdm_pooled_ae_arma_v1dedup_ckpt.parquet",
    ("ae", "c"): cfg.DATA / "processed" / "sdm_pooled_ae_armc_v1dedup_ckpt.parquet",
}
ARMS = ["a", "c"]
METRICS = ["kl", "l1_sum", "l1_mean", "chebyshev"]
N_BOOTSTRAP = 1000
SEED = 4000
OUT_PATH = cfg.DATA / "processed" / "diag_backbone_paired_bootstrap_sdm.csv"
SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, Backbone-Vergleich)")


def fit_ref(backbone: str, arm: str) -> dict:
    bsp6.load_pooled = lambda b, a, _p=POOLED[(backbone, arm)]: pd.read_parquet(_p)
    X, y, split_arr, n_classes, lsoa_codes = bsp6.build_xy(backbone, arm, SCENARIO)
    r = bsp6.fit_mlp(X, y, split_arr, bsp7.SEEDS[0], X.shape[1], n_classes, lsoa_codes=lsoa_codes)
    m = bsp7.per_lsoa_metrics(r["y_test"], r["test_pred"])
    return {"lsoa": r["test_lsoa"], "metrics": m}


def main():
    refs = {(bb, arm): fit_ref(bb, arm) for bb in ("tessera", "ae") for arm in ARMS}
    rows = []
    for arm in ARMS:
        rt, ra = refs[("tessera", arm)], refs[("ae", arm)]
        codes_t, codes_a = set(rt["lsoa"]), set(ra["lsoa"])
        common = codes_t & codes_a
        print(f"Arm {arm.upper()}: TESSERA {len(codes_t)} / AE {len(codes_a)} Test-LSOAs, gemeinsam {len(common)}")
        assert common == codes_t == codes_a, "Test-LSOA-Mengen weichen zwischen den Backbones ab - Bug/Annahme verletzt."

        dt = pd.DataFrame({"lsoa_code": rt["lsoa"], **rt["metrics"]}).set_index("lsoa_code")
        da = pd.DataFrame({"lsoa_code": ra["lsoa"], **ra["metrics"]}).set_index("lsoa_code")
        codes = sorted(common)
        cell_ids = np.array([bsp7.LSOA_TO_CELL[c] for c in codes])
        unique_cells = np.unique(cell_ids)
        cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}

        rng = np.random.default_rng(SEED)
        resamples = [rng.choice(unique_cells, size=len(unique_cells), replace=True) for _ in range(N_BOOTSTRAP)]
        for metric in METRICS:
            vt = dt.loc[codes, metric].to_numpy()
            va = da.loc[codes, metric].to_numpy()
            point_diff = float(vt.mean() - va.mean())
            diffs = np.empty(N_BOOTSTRAP)
            for i, cells in enumerate(resamples):
                idx = np.concatenate([cell_to_idx[c] for c in cells])
                diffs[i] = vt[idx].mean() - va[idx].mean()
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            sig = bool(not (lo <= 0 <= hi))
            rows.append({"arm": arm, "metric": metric, "tessera_dedup": float(vt.mean()),
                        "ae_dedup": float(va.mean()), "point_diff_tessera_minus_ae": point_diff,
                        "ci_low": lo, "ci_high": hi, "significant": sig,
                        "n_test_lsoa": len(codes), "n_cells": len(unique_cells)})
            print(f"  [{arm}/{metric}] TESSERA-AE = {point_diff:+.5f}  95%-CI=[{lo:+.5f}, {hi:+.5f}]  signifikant={sig}")

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
