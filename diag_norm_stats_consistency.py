"""
diag_norm_stats_consistency.py

Diagnose-Aufgabe (2026-09-18, Jakob), Teil 2: prueft die TESSERA-Z-Score-Statistik
(norm_stats_tessera_pixels.json) gegen die AKTUELLEN Trainingspixel - nicht nur per
Zeitstempel (siehe [[feedback-data-freshness-check]]-Vorgabe: mtime-Reihenfolge allein reicht
nicht), sondern durch direktes Neu-Fitten auf dem heutigen Train-Split und Kanal-fuer-Kanal-
Vergleich. Nur lesen/berichten, KEINE neue Statistikdatei wird gespeichert/verwendet.
"""
import json

import numpy as np
import pandas as pd

import config as cfg
from normalize import fit_zscore, load_stats

PIXELS_PATH = cfg.DATA / "processed" / "pixels_tessera.parquet"
STATS_PATH = cfg.DATA / "processed" / "norm_stats_tessera_pixels.json"


def main():
    print(f"Lade {PIXELS_PATH} (train-Split, embedding+split)...")
    df = pd.read_parquet(PIXELS_PATH, columns=["embedding", "split"])
    print(f"  {len(df)} Zeilen gesamt, Split-Verteilung:\n{df['split'].value_counts()}")

    train = df[df["split"] == "train"]
    emb_train = np.stack(train["embedding"].to_numpy()).astype(np.float32)
    print(f"  {len(emb_train)} Train-Pixel, Dim={emb_train.shape[1]}")

    stored = load_stats(STATS_PATH)
    stored_mean = np.asarray(stored["mean"])
    stored_std = np.asarray(stored["std"])

    fresh = fit_zscore(emb_train)
    fresh_mean = np.asarray(fresh["mean"])
    fresh_std = np.asarray(fresh["std"])

    print(f"\n=== Vergleich gespeicherte vs. frisch gefittete Statistik ({STATS_PATH}) ===")
    mean_diff = np.abs(stored_mean - fresh_mean)
    std_diff = np.abs(stored_std - fresh_std)
    print(f"  Mittelwert-Differenz: max={mean_diff.max():.6g} mean={mean_diff.mean():.6g}")
    print(f"  Std-Differenz:        max={std_diff.max():.6g} mean={std_diff.mean():.6g}")
    if mean_diff.max() < 1e-4 and std_diff.max() < 1e-4:
        print("  -> IDENTISCH (bis auf Rundung) - Statistik stammt von genau dieser Trainingspixelmenge.")
    else:
        print("  -> ABWEICHUNG - Statistik stammt NICHT von der aktuellen Trainingspixelmenge, "
              "oder es gibt einen anderen Fitting-Unterschied.")
        worst = np.argsort(-mean_diff)[:10]
        print(f"  Kanaele mit groesster Mittelwert-Abweichung: {worst.tolist()}")
        print(pd.DataFrame({"channel": worst, "stored_mean": stored_mean[worst],
                             "fresh_mean": fresh_mean[worst],
                             "stored_std": stored_std[worst], "fresh_std": fresh_std[worst]}
                            ).to_string(index=False))

    print(f"\n=== Kanaele mit sehr kleiner Std / konstante Kanaele (frisch gefittet) ===")
    small_std_thresh = 1e-3
    small_std = np.where(fresh_std < small_std_thresh)[0]
    print(f"  Schwelle: std < {small_std_thresh}")
    print(f"  Betroffene Kanaele: {len(small_std)} von {len(fresh_std)}")
    if len(small_std) > 0:
        print(pd.DataFrame({"channel": small_std, "mean": fresh_mean[small_std],
                             "std": fresh_std[small_std]}).to_string(index=False))

    print(f"\n=== Vollstaendige Kanal-Tabelle (gespeicherte Statistik) ===")
    tab = pd.DataFrame({"channel": np.arange(len(stored_mean)), "stored_mean": stored_mean,
                         "stored_std": stored_std, "fresh_mean": fresh_mean, "fresh_std": fresh_std})
    print(tab.describe().to_string())


if __name__ == "__main__":
    main()
