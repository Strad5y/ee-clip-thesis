"""
run_retrieval_baselines_tessera_v1dedup.py

Schritt 3 (2026-09-19, Jakob): Retrieval-Baseline fuer checkpoints/tessera_v1_dedup/best.pt auf
dem deduplizierten pixels_tessera_dedup.parquet (Test-Split) - identische Methodik wie
run_retrieval_baselines.py (dessen Funktionen unveraendert importiert werden), nur Checkpoint-
und Datenpfad sind neu. Ersetzt nichts - neuer Output retrieval_baselines_tessera_v1_dedup.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from chunked_join_runner import LOG_DIR, _Tee
from run_retrieval_baselines import (
    DEVICE, apply_checkpoint_normalization, evaluate_full, load_dataset_with_values, load_model,
    print_table, report_distribution, summarize, to_dataframe, top_k_set,
)

CKPT_PATH = Path("checkpoints/tessera_v1_dedup/best.pt")
PIXELS_PATH = "data/processed/pixels_tessera_dedup.parquet"
OUT_PATH = Path("data/processed") / "retrieval_baselines_tessera_v1_dedup.csv"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"retrieval_baselines_tessera_v1_dedup_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print("=== run_retrieval_baselines_tessera_v1dedup.py gestartet ===")
    print(f"Log-Datei: {log_path}")
    print(f"Device: {DEVICE}")

    model, cfg = load_model(CKPT_PATH)
    ds = load_dataset_with_values(PIXELS_PATH)
    apply_checkpoint_normalization(ds, cfg)

    idx_val = np.where(ds.split == "val")[0]
    idx_test = np.where(ds.split == "test")[0]
    print(f"\nVal: {len(idx_val)} Zeilen, Test: {len(idx_test)} Zeilen")

    report_distribution("primary_key (Test)", ds.primary_key[idx_test])
    report_distribution("primary_value (Test)", ds.primary_value[idx_test])

    top_key_test = top_k_set(ds.primary_key[idx_test])
    top_val_test = top_k_set(ds.primary_value[idx_test])
    top_key_val = top_k_set(ds.primary_key[idx_val])
    top_val_val = top_k_set(ds.primary_value[idx_val])

    t0 = time.time()
    acc_test = evaluate_full(model, ds, idx_test, top_key_test, top_val_test)
    print(f"\nTest-Auswertung fertig ({time.time() - t0:.1f}s)")
    summ_test = summarize(acc_test)
    print_table(summ_test, "TEST")

    t0 = time.time()
    acc_val = evaluate_full(model, ds, idx_val, top_key_val, top_val_val)
    print(f"\nVal-Auswertung fertig ({time.time() - t0:.1f}s)")
    summ_val = summarize(acc_val)
    print_table(summ_val, "VAL")

    df = pd.concat([to_dataframe(summ_test, "test"), to_dataframe(summ_val, "val")], ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nErgebnistabelle gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
