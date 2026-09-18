"""
diag_sdm_paired_bootstrap_arm_diff.py

Kurztest (2026-09-18, Jakob), bevor weiter nach Ursachen fuer den TESSERA-Arm-A-vs-C-
Unterschied gesucht wird: ist der Unterschied ueberhaupt statistisch gesichert? Die
95%-Zellebenen-CIs von TESSERA/A und TESSERA/C ueberlappen fast vollstaendig (unabhaengige
CIs je Arm) - das beantwortet aber nicht, ob die PAARWEISE Differenz (C-A) gesichert von 0
verschieden ist. Gepaarter Cluster-Bootstrap auf Zellebene (DIESELBEN gezogenen Zellen fuer
beide Arme in jedem Resample) beantwortet das direkt.

Wiederverwendet build_sdm_phase6_predict.py::fit_mlp() und build_sdm_phase7_metrics.py::
per_lsoa_metrics()/LSOA_TO_CELL - identische Hyperparameter/Splits/Seeds wie der bestehende
Hauptlauf (dedupliziert, Phase-7-Skript). Nur Messung - kein Neutraining der CLIP-/SigLIP-
Gewichte oder der Projektionskoepfe, nur die (ohnehin pro Lauf neu gefitteten) kleinen
SDM-Vorhersage-MLPs.
"""
import numpy as np
import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED0 = 2000


def load_pooled_dedup(backbone: str, arm: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_arm{arm}_dedup.parquet")


bsp6.load_pooled = load_pooled_dedup  # MUSS vor dem Import von build_sdm_phase7_metrics passieren

import build_sdm_phase7_metrics as bsp7  # noqa: E402

SEEDS = bsp7.SEEDS
METRICS = ["kl", "l1_sum", "l1_mean", "chebyshev"]
MAIN_SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, dedup)")


def per_seed_metrics_and_ref(backbone: str, arm: str, scenario) -> tuple:
    """5 Seeds (identisch zu build_sdm_phase7_metrics.py::evaluate_config): pro Seed die
    Mittelwerte der per-LSOA-Metriken (Verfahrensvarianz), plus fuer Seed 0 die vollen
    per-LSOA-Arrays + lsoa-Codes (fuer den gepaarten Zell-Bootstrap - derselbe Referenz-Seed
    wie im bestehenden Hauptlauf, bootstrap_seed_idx=0)."""
    X, y, split_arr, n_classes, lsoa_codes = bsp6.build_xy(backbone, arm, scenario)
    d_in = X.shape[1]
    per_seed = []
    ref = None
    for s in SEEDS:
        r = bsp6.fit_mlp(X, y, split_arr, s, d_in, n_classes, lsoa_codes=lsoa_codes)
        m = bsp7.per_lsoa_metrics(r["y_test"], r["test_pred"])
        per_seed.append({k: v.mean() for k, v in m.items()})
        if s == SEEDS[0]:
            ref = {"metrics": m, "lsoa": r["test_lsoa"]}
    return pd.DataFrame(per_seed), ref


def paired_cell_bootstrap_diff(metric_a: np.ndarray, metric_c: np.ndarray, cell_ids: np.ndarray,
                                n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED0) -> np.ndarray:
    """Standard-Cluster-Bootstrap, aber DIESELBEN gezogenen Zellen fuer metric_a UND metric_c
    in jedem Resample (gepaart) - Differenz C-A pro Resample."""
    unique_cells = np.unique(cell_ids)
    cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}
    rng = np.random.default_rng(seed)
    n_cells = len(unique_cells)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled = rng.choice(unique_cells, size=n_cells, replace=True)
        idx = np.concatenate([cell_to_idx[c] for c in sampled])
        diffs[i] = metric_c[idx].mean() - metric_a[idx].mean()
    return diffs


def analyze_backbone(backbone: str) -> dict:
    print(f"\n=== {backbone}: Arm A vs Arm C (gepaart) ===")
    scenario = MAIN_SCENARIO

    seeds_a, ref_a = per_seed_metrics_and_ref(backbone, "a", scenario)
    seeds_c, ref_c = per_seed_metrics_and_ref(backbone, "c", scenario)

    print(f"  Seed-Streuung Arm A (5 Seeds):\n{seeds_a.agg(['mean', 'std']).to_string()}")
    print(f"  Seed-Streuung Arm C (5 Seeds):\n{seeds_c.agg(['mean', 'std']).to_string()}")

    # Ausrichtung ueber lsoa_code, NICHT Zeilenposition - robust gegen etwaige abweichende
    # Reihenfolge zwischen den gepoolten Arm-Dateien.
    df_a = pd.DataFrame({"lsoa_code": ref_a["lsoa"], **ref_a["metrics"]})
    df_c = pd.DataFrame({"lsoa_code": ref_c["lsoa"], **ref_c["metrics"]})
    merged = df_a.merge(df_c, on="lsoa_code", suffixes=("_a", "_c"))
    print(f"  Gemeinsame Test-LSOAs (Seed {SEEDS[0]}): {len(merged)} "
          f"(Arm A: {len(df_a)}, Arm C: {len(df_c)})")
    assert len(merged) == len(df_a) == len(df_c), "Test-LSOA-Mengen weichen zwischen Arm A/C ab - Bug."

    cell_ids = np.array([bsp7.LSOA_TO_CELL[c] for c in merged["lsoa_code"]])

    diff_rows = []
    for metric in METRICS:
        a_vals = merged[f"{metric}_a"].to_numpy()
        c_vals = merged[f"{metric}_c"].to_numpy()
        point_diff = c_vals.mean() - a_vals.mean()
        diffs = paired_cell_bootstrap_diff(a_vals, c_vals, cell_ids)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        frac_pos = float((diffs > 0).mean())
        significant = bool(not (lo <= 0 <= hi))
        print(f"  {metric}: Punktdiff (C-A)={point_diff:+.5f}  95%-CI=[{lo:+.5f}, {hi:+.5f}]  "
              f"Anteil Resamples>0={frac_pos:.3f}  signifikant={significant}")
        diff_rows.append({"backbone": backbone, "metric": metric, "point_diff_c_minus_a": point_diff,
                           "ci_low": lo, "ci_high": hi, "frac_resamples_positive": frac_pos,
                           "significant": significant, "n_test_lsoa": len(merged),
                           "n_cells": len(np.unique(cell_ids))})

    return {"backbone": backbone, "seeds_a": seeds_a, "seeds_c": seeds_c, "diffs": diff_rows}


def main():
    all_diff_rows = []
    seed_rows = []
    for backbone in ("ae", "tessera"):
        res = analyze_backbone(backbone)
        all_diff_rows.extend(res["diffs"])
        for arm, seeds_df in (("a", res["seeds_a"]), ("c", res["seeds_c"])):
            for metric in METRICS:
                seed_rows.append({"backbone": backbone, "arm": arm, "metric": metric,
                                   "seed_mean": seeds_df[metric].mean(),
                                   "seed_std": seeds_df[metric].std()})

    diff_df = pd.DataFrame(all_diff_rows)
    seed_df = pd.DataFrame(seed_rows)
    diff_out = cfg.DATA / "processed" / "diag_sdm_paired_bootstrap_arm_diff.csv"
    seed_out = cfg.DATA / "processed" / "diag_sdm_seed_variance_arm_a_c.csv"
    diff_df.to_csv(diff_out, index=False)
    seed_df.to_csv(seed_out, index=False)
    print(f"\nGespeichert: {diff_out}, {seed_out}")


if __name__ == "__main__":
    main()
