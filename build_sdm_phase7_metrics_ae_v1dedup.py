"""
build_sdm_phase7_metrics_ae_v1dedup.py

Schritt 3 AlphaEarth (2026-09-21, Jakob): SDM-Auswertung fuer AlphaEarth Arm A, B UND C (alle drei
neu gepoolt, Arm C mit dem NEUEN checkpoints/ae_v1_dedup/best.pt; build_sdm_ae_v1dedup.py) - identische Hyperparameter/Splits/Seeds/Zellebenen-
Bootstrap wie build_sdm_phase7_metrics.py, PLUS gepaarter Zell-Bootstrap C-A (wiederverwendet
diag_sdm_paired_bootstrap_arm_diff.py::paired_cell_bootstrap_diff()). Nur AlphaEarth - das
TESSERA-Pendant bleibt unveraendert. Ersetzt nichts.
"""
import numpy as np
import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6
import build_sdm_phase7_metrics as bsp7
# WICHTIG (Bugfix, 2026-09-19): dieses Modul importiert selbst build_sdm_phase6_predict und
# setzt bei seinem EIGENEN ersten Import bsp6.load_pooled auf SEINE eigene (auf die alte,
# checkpoint-unabhaengige *_dedup.parquet zeigende) Variante. Wird dieser Import NACH dem
# eigenen Monkeypatch unten platziert, ueberschreibt er ihn wieder - beobachtet: ein erster
# Lauf dieses Skripts lieferte dadurch fuer Arm C stillschweigend die ALTEN (checkpoints/
# tessera_v1-basierten) Pooling-Werte statt der neuen (tessera_v1_dedup-basierten), obwohl
# sdm_pooled_tessera_armc_v1dedup_ckpt.parquet nachweislich andere Werte enthaelt (siehe
# MANIFEST.md fuer den vollen Befund). Deshalb hier VOR dem eigenen Monkeypatch importiert,
# UND der Monkeypatch direkt vor jeder Nutzung nochmal defensiv gesetzt.
from diag_sdm_paired_bootstrap_arm_diff import paired_cell_bootstrap_diff  # noqa: E402

ARMS = ["a", "b", "c"]
BACKBONE = "ae"


def load_pooled_variant(backbone: str, arm: str) -> pd.DataFrame:
    if backbone != BACKBONE:
        # bsp7.evaluate_mean_baseline() ruft FEST build_xy("tessera", "a", ...) auf (fest verdrahteter
        # TESSERA-Wert in build_sdm_phase7_metrics.py, dort nur fuer Ziele/Split/LSOA-Codes noetig,
        # die vom Backbone unabhaengig sind). Diese Anfrage bedient die bestehende, unveraenderte
        # TESSERA-Arm-A-Dedup-Datei - exakt die Datei, die auch der TESSERA-Wrapper dafuer nutzt.
        return pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_arm{arm}_dedup.parquet")
    return pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_arm{arm}_v1dedup_ckpt.parquet")


bsp6.load_pooled = load_pooled_variant

MAIN_SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, AE Arm A/B/C neu, Arm C = ae_v1_dedup)")
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
    bsp6.load_pooled = load_pooled_variant  # defensiv nochmal gesetzt, siehe Bugfix-Kommentar oben
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
    out_path = cfg.DATA / "processed" / "sdm_phase7_metrics_ae_v1dedup.csv"
    results.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")

    print("\n=== Gepaarter Zell-Bootstrap C (ae_v1_dedup) minus A ===")
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

    diff_out = cfg.DATA / "processed" / "sdm_paired_bootstrap_ae_v1dedup_c_minus_a.csv"
    pd.DataFrame(diff_rows).to_csv(diff_out, index=False)
    print(f"\nGespeichert: {diff_out}")


if __name__ == "__main__":
    main()
