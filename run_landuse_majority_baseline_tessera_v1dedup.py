"""
run_landuse_majority_baseline_tessera_v1dedup.py

TESSERA-Pendant (2026-09-30, Jakob) zu run_landuse_majority_baseline_ae_v1dedup.py: Haeufigkeitsniveau-Baseline fuer
landuse_pairs_tessera_dedup.parquet - dieselbe Funktion und derselbe Filter (load_clean) wie im
Land-Use-Wrapper; run_landuse_probe_tessera_v1dedup.py schreibt diese Baseline nicht selbst. Die aeltere
landuse_probe_majority_baseline_tessera_dedup.csv (5.9.) ist veraltet (Parquet am 18.9. neu gebaut).
Ausgabe: landuse_probe_majority_baseline_tessera_dedup_ckpt.csv.
"""
import pandas as pd

import config as cfg
import run_landuse_probe as rlp

DATA_NAME = "tessera_dedup"


def main():
    df = rlp.load_clean(DATA_NAME)
    y, split = df["code_2021"].to_numpy(), df["split"].to_numpy()
    res = rlp.majority_baseline(y[split == "train"], y[split == "val"], y[split == "test"])
    out = pd.DataFrame([{"config": f"{DATA_NAME}_haeufigkeitsniveau", **{k: v for k, v in res.items() if k != "pred_test"}}])
    path = cfg.DATA / "processed" / f"landuse_probe_majority_baseline_{DATA_NAME}_ckpt.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"Gespeichert: {path}")


if __name__ == "__main__":
    main()
