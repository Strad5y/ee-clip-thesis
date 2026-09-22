"""
build_sdm_phase7_metrics_tessera_dedupnorm.py

TESSERA-Normalisierung nachziehen (2026-09-22, Jakob), Teil 2: SDM-Auswertung fuer TESSERA mit
ALLEN DREI Armen auf derselben z-score-Statistik (Arm A/B neu gepoolt mit den Dedup-Statistiken,
build_sdm_arm_ab_tessera_dedupnorm.py; Arm C unveraendert, nutzte diese Statistik bereits).
Identische Hyperparameter/Splits/Seeds/Zellebenen-Bootstrap wie build_sdm_phase7_metrics.py,
PLUS gepaarter Zell-Bootstrap C-A (wiederverwendet diag_sdm_paired_bootstrap_arm_diff.py::
paired_cell_bootstrap_diff()). Ersetzt nichts - neue Ausgabedateien mit Suffix `_dedupnorm`.
"""
import numpy as np
import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6
import build_sdm_phase7_metrics as bsp7
from diag_sdm_paired_bootstrap_arm_diff import paired_cell_bootstrap_diff  # noqa: E402

ARMS = ["a", "b", "c"]
BACKBONE = "tessera"
POOLED = {
    "a": cfg.DATA / "processed" / "sdm_pooled_tessera_arma_dedupnorm.parquet",
    "b": cfg.DATA / "processed" / "sdm_pooled_tessera_armb_dedupnorm.parquet",
    "c": cfg.DATA / "processed" / "sdm_pooled_tessera_armc_v1dedup_ckpt.parquet",
}


def load_pooled_variant(backbone: str, arm: str) -> pd.DataFrame:
    return pd.read_parquet(POOLED[arm])


bsp6.load_pooled = load_pooled_variant

MAIN_SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, TESSERA Arm A/B/C alle mit Dedup-Statistik)")
METRICS = ["kl", "l1_sum", "l1_mean", "chebyshev"]


def per_seed_metrics_and_ref(arm: str, scenario) -> dict:
    X, y, split_arr, n_classes, lsoa_codes = bsp6.build_xy(BACKBONE, arm, scenario)
    d_in = X.shape[1]
    ref = None
    for s in bsp7.SEEDS:
        r = bsp6.fit_mlp(X, y, split_arr, s, d_in, n_classes, lsoa_codes=lsoa_codes)
        m = bsp7.per_lsoa_metrics(r["y_test"], r["test_pred"])
        if s == bsp7.SEEDS[0]:
            ref = {"metrics": m, "lsoa": r["test_lsoa"]}
    return ref


def main():
    bsp6.load_pooled = load_pooled_variant  # defensiv nochmal gesetzt (Bugfix-Muster aus der AE-Aufgabe)
    assert bsp6.load_pooled is load_pooled_variant
    all_rows = []
    scenario = MAIN_SCENARIO

    mb = bsp7.evaluate_mean_baseline(scenario, do_lsoa_bootstrap=False)
    row = bsp7.fmt_row("Mittelwertsvorhersage", mb)
    row.update({"scenario": scenario.name, "backbone": "-", "arm": "-"})
    all_rows.append(row)
    print(f"Mittelwertsvorhersage: KL={mb['point_mean']['kl']:.4f} (CI Zelle {mb['kl_ci_tile']})")

    for arm in ARMS:
        r = bsp7.evaluate_config(BACKBONE, arm, scenario, do_lsoa_bootstrap=False)
        row = bsp7.fmt_row(f"{BACKBONE}/{arm}", r)
        row.update({"scenario": scenario.name, "backbone": BACKBONE, "arm": arm})
        all_rows.append(row)
        print(f"{BACKBONE}/{arm}: KL={r['point_mean']['kl']:.4f}+-{r['point_std']['kl']:.4f} "
              f"(CI Zelle {r['kl_ci_tile']})")

    results = pd.DataFrame(all_rows)
    out_path = cfg.DATA / "processed" / "sdm_phase7_metrics_tessera_dedupnorm.csv"
    results.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")

    print("\n=== Gepaarter Zell-Bootstrap C (Arm C) minus A (Arm A, Dedup-Statistik) ===")
    ref_a = per_seed_metrics_and_ref("a", scenario)
    ref_c = per_seed_metrics_and_ref("c", scenario)
    df_a = pd.DataFrame({"lsoa_code": ref_a["lsoa"], **ref_a["metrics"]})
    df_c = pd.DataFrame({"lsoa_code": ref_c["lsoa"], **ref_c["metrics"]})
    merged = df_a.merge(df_c, on="lsoa_code", suffixes=("_a", "_c"))
    print(f"  Gemeinsame Test-LSOAs: {len(merged)} (A: {len(df_a)}, C: {len(df_c)})")
    assert len(merged) == len(df_a) == len(df_c), "Test-LSOA-Mengen weichen zwischen Arm A/C ab."
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
        diff_rows.append({"metric": metric, "point_diff_c_minus_a": point_diff,
                           "ci_low": lo, "ci_high": hi, "frac_resamples_positive": frac_pos,
                           "significant": significant, "n_test_lsoa": len(merged)})

    diff_out = cfg.DATA / "processed" / "sdm_paired_bootstrap_tessera_dedupnorm_c_minus_a.csv"
    pd.DataFrame(diff_rows).to_csv(diff_out, index=False)
    print(f"\nGespeichert: {diff_out}")


if __name__ == "__main__":
    main()
