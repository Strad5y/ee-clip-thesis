"""
build_sdm_phase7_metrics_reversed.py

Folgetest (2026-09-18, Jakob): KL/L1/Chebyshev fuer DREI Varianten - Arm A, Arm C (bisherige
Reihenfolge, dedupliziert), Arm C (umgekehrte Reihenfolge, head(mean(x))). Nur Hauptlauf
(9cat/Split A/partitionsrein), identische Hyperparameter/Splits/Seeds/Bootstrap-Methodik
(Zellebene, ueber sdm_lsoa_cell_overlay_london.csv) wie build_sdm_phase7_metrics.py, per
Monkeypatch von load_pooled() wiederverwendet. Ersetzt nichts.
"""
import time

import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6

ARMS = ["a", "c", "c_reversed"]


def load_pooled_variant(backbone: str, arm: str) -> pd.DataFrame:
    if arm == "c_reversed":
        return pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_armc_reversed_dedup.parquet")
    return pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_arm{arm}_dedup.parquet")


bsp6.load_pooled = load_pooled_variant  # MUSS vor dem Import von build_sdm_phase7_metrics passieren

import build_sdm_phase7_metrics as bsp7  # noqa: E402

MAIN_SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, dedup, Reihenfolge-Vergleich)")


def main():
    t0 = time.time()
    all_rows = []
    scenario = MAIN_SCENARIO
    print(f"\n=== Szenario: {scenario.name} ===")

    mb = bsp7.evaluate_mean_baseline(scenario, do_lsoa_bootstrap=False)
    row = bsp7.fmt_row("Mittelwertsvorhersage", mb)
    row.update({"scenario": scenario.name, "backbone": "-", "arm": "-"})
    all_rows.append(row)
    print(f"  Mittelwertsvorhersage: KL={mb['point_mean']['kl']:.4f} (CI Zelle {mb['kl_ci_tile']})")

    for backbone in ["ae", "tessera"]:
        for arm in ARMS:
            r = bsp7.evaluate_config(backbone, arm, scenario, do_lsoa_bootstrap=False)
            row = bsp7.fmt_row(f"{backbone}/{arm}", r)
            row.update({"scenario": scenario.name, "backbone": backbone, "arm": arm})
            all_rows.append(row)
            print(f"  {backbone}/{arm}: KL={r['point_mean']['kl']:.4f}+-{r['point_std']['kl']:.4f} "
                  f"(CI Zelle {r['kl_ci_tile']})")

    results = pd.DataFrame(all_rows)
    out_path = cfg.DATA / "processed" / "sdm_phase7_metrics_reversed.csv"
    results.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")
    print(f"Gesamtlaufzeit: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
