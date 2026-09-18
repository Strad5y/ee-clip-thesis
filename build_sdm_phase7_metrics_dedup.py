"""
build_sdm_phase7_metrics_dedup.py

Schritt 3 (2026-09-17, Jakob): KL/L1/Chebyshev-Metriken NEU berechnet auf den deduplizierten
LSOA-Poolungen - NUR Hauptlauf (9cat/Split A/partitionsrein). Identische Hyperparameter/Splits/
Seeds/Bootstrap-Methodik wie build_sdm_phase7_metrics.py (Bootstrap-Cluster-Einheit: die 1-km-
Zelle, ueber sdm_lsoa_cell_overlay_london.csv - NICHT LSOA- oder Pixelebene), per Monkeypatch
von load_pooled() auf die *_dedup.parquet-Dateien wiederverwendet, nicht neu geschrieben.
Ersetzt nichts - neuer Output sdm_phase7_metrics_dedup.csv.
"""
import time

import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6


def load_pooled_dedup(backbone: str, arm: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_arm{arm}_dedup.parquet")


bsp6.load_pooled = load_pooled_dedup  # MUSS vor dem Import von build_sdm_phase7_metrics passieren

import build_sdm_phase7_metrics as bsp7  # noqa: E402 (Monkeypatch-Reihenfolge ist hier absichtlich)

MAIN_SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, dedupliziert)")


def main():
    t0 = time.time()
    all_rows = []
    scenario = MAIN_SCENARIO
    print(f"\n=== Szenario: {scenario.name} ===")

    mb = bsp7.evaluate_mean_baseline(scenario, do_lsoa_bootstrap=True)
    row = bsp7.fmt_row("Mittelwertsvorhersage", mb)
    row.update({"scenario": scenario.name, "backbone": "-", "arm": "-"})
    all_rows.append(row)
    print(f"  Mittelwertsvorhersage: KL={mb['point_mean']['kl']:.4f} (CI Zelle {mb['kl_ci_tile']})")

    for backbone in ["ae", "tessera"]:
        for arm in ["a", "b", "c"]:
            r = bsp7.evaluate_config(backbone, arm, scenario, do_lsoa_bootstrap=True)
            row = bsp7.fmt_row(f"{backbone}/{arm}", r)
            row.update({"scenario": scenario.name, "backbone": backbone, "arm": arm})
            all_rows.append(row)
            print(f"  {backbone}/{arm}: KL={r['point_mean']['kl']:.4f}+-{r['point_std']['kl']:.4f} "
                  f"(CI Zelle {r['kl_ci_tile']}) (CI LSOA {r.get('kl_ci_lsoa')})")

            rb = bsp7.evaluate_random_embedding_baseline(backbone, arm, scenario, do_lsoa_bootstrap=True)
            row = bsp7.fmt_row(f"Random-Embedding ({backbone}/{arm}-dim)", rb)
            row.update({"scenario": scenario.name, "backbone": backbone, "arm": arm})
            all_rows.append(row)
            print(f"    Random-Embedding: KL={rb['point_mean']['kl']:.4f} (CI Zelle {rb['kl_ci_tile']})")

    results = pd.DataFrame(all_rows)
    out_path = cfg.DATA / "processed" / "sdm_phase7_metrics_dedup.csv"
    results.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")
    print(f"Gesamtlaufzeit: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
