"""
build_sdm_phase6_predict_reversed.py

Folgetest (2026-09-18, Jakob): SDM-Vorhersagekopf-Neuauswertung mit DREI Varianten - Arm A
(roh), Arm C bisherige Reihenfolge (mean(head(x)), dedupliziert), Arm C umgekehrte Reihenfolge
(head(mean(x)), aus build_sdm_pooled_embeddings_reversed_dedup.py). Nur Hauptlauf (9cat/Split
A/partitionsrein), identische Hyperparameter/Splits/Seeds wie build_sdm_phase6_predict.py
(per Monkeypatch von load_pooled() wiederverwendet). Ersetzt nichts.
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


bsp6.load_pooled = load_pooled_variant

MAIN_SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, dedup, Reihenfolge-Vergleich)")


def main():
    t0 = time.time()
    all_rows = []
    scenario = MAIN_SCENARIO
    print(f"\n=== Szenario: {scenario.name} ===")

    for backbone in ["ae", "tessera"]:
        for arm in ARMS:
            X, y, split_arr, n_classes, _lsoa = bsp6.build_xy(backbone, arm, scenario)
            d_in = X.shape[1]
            n_train = (split_arr == "train").sum()
            n_val = (split_arr == "val").sum()
            n_test = (split_arr == "test").sum()
            res = bsp6.run_seeds(X, y, split_arr, d_in, n_classes)
            print(f"  {backbone}/{arm} (d={d_in}, n={len(X)}={n_train}/{n_val}/{n_test}): "
                  f"test_loss={bsp6.summarize(res)}  (val={bsp6.summarize(res, 'val_loss')})")
            for _, row in res.iterrows():
                all_rows.append({"scenario": scenario.name, "backbone": backbone, "arm": arm,
                                  **row.to_dict()})

    results = pd.DataFrame(all_rows)
    out_path = cfg.DATA / "processed" / "sdm_phase6_head_results_reversed.csv"
    results.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")
    print(f"Gesamtlaufzeit: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
