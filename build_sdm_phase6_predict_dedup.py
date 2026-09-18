"""
build_sdm_phase6_predict_dedup.py

Schritt 3 (2026-09-17, Jakob): SDM-Vorhersagekopf NEU trainiert auf den deduplizierten LSOA-
Poolungen (build_sdm_pooled_embeddings_dedup.py, Schritt 1) - NUR der Hauptlauf (9 Kategorien,
Split-Variante A, partitionsreine Poolung). Identische Hyperparameter/Splits/Seeds wie
build_sdm_phase6_predict.py - werden per Monkeypatch von load_pooled() auf die *_dedup.parquet-
Dateien wiederverwendet (importiert, nicht neu geschrieben/dupliziert). Ersetzt nichts - neue
Outputs unter sdm_phase6_head_results_dedup.csv / sdm_phase6_baselines_dedup.csv.
"""
import time

import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6


def load_pooled_dedup(backbone: str, arm: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_arm{arm}_dedup.parquet")


bsp6.load_pooled = load_pooled_dedup  # build_xy() liest ab hier die deduplizierten Dateien

MAIN_SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, dedupliziert)")


def main():
    t0 = time.time()
    all_rows, baseline_rows = [], []
    scenario = MAIN_SCENARIO
    print(f"\n=== Szenario: {scenario.name} ===")

    _, y_any, split_any, _, _ = bsp6.build_xy("tessera", "a", scenario)
    mb = bsp6.mean_baseline(y_any, split_any)
    print(f"  Mittelwertsvorhersage-Baseline: test_loss={mb['test_loss']:.4f} (deterministisch)")
    baseline_rows.append({"scenario": scenario.name, "baseline": "mean", "backbone": "-", "arm": "-",
                           "test_loss_mean": mb["test_loss"], "test_loss_std": 0.0})

    for backbone in ["ae", "tessera"]:
        for arm in ["a", "b", "c"]:
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

            rb = bsp6.random_embedding_baseline(y, split_arr, d_in, n_classes)
            print(f"    Random-Embedding-Baseline (d={d_in}): test_loss={bsp6.summarize(rb)}")
            baseline_rows.append({"scenario": scenario.name, "baseline": "random_embedding",
                                   "backbone": backbone, "arm": arm,
                                   "test_loss_mean": rb["test_loss"].mean(),
                                   "test_loss_std": rb["test_loss"].std()})

    results = pd.DataFrame(all_rows)
    baselines = pd.DataFrame(baseline_rows)
    out_results = cfg.DATA / "processed" / "sdm_phase6_head_results_dedup.csv"
    out_baselines = cfg.DATA / "processed" / "sdm_phase6_baselines_dedup.csv"
    results.to_csv(out_results, index=False)
    baselines.to_csv(out_baselines, index=False)
    print(f"\nGespeichert: {out_results}, {out_baselines}")
    print(f"Gesamtlaufzeit: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
