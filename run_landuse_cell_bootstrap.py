"""
run_landuse_cell_bootstrap.py

INSTRUCTIONS_landuse_cell_bootstrap.md, Phase 2+3: stellt den bestehenden Zeilen-Test-Bootstrap
aus run_landuse_probe.py dem neuen Kachel-Test-Bootstrap gegenueber, fuer beide Backbones und
alle drei Arme, inklusive gepaartem Diff (trainiert-roh).

Reproduziert NUR die bereits etablierte, deterministische Fitting-Prozedur aus
run_landuse_probe.py (build_configs()+run_probe(), UNVERAENDERT) - kein Retraining, keine
Aenderung an Probe-Fit/Normalisierung/C-Grid, kein Eingriff in den Trainings-Bootstrap. Die
Testvorhersagen sind nicht persistiert (siehe Phase-1-Bericht), muessen also reproduziert
werden, um ueberhaupt bootstrappen zu koennen - das ist dieselbe Fit-Prozedur, die auch den
urspruenglichen Ergebnissen zugrunde liegt, kein neues Training.

Stabilitaetscheck (Nutzervorgabe): der Kachel-Bootstrap hat nur ~167 Cluster im Testset (vs.
hunderttausende Zeilen beim Zeilen-Bootstrap) - re-run mit einem zweiten Startseed, Vergleich
der CI-Grenzen. Weichen sie um mehr als 1% relativ voneinander ab, wird auf 5000 Ziehungen
erhoeht und das gemeldet.
"""
import time

import numpy as np
import pandas as pd

import config as cfg
import run_landuse_probe as rlp

BACKBONES = ("ae", "tessera")
N_DEFAULT = 1000
N_ESCALATED = 5000
STABILITY_SEED_A = rlp.TEST_BOOTSTRAP_SEED0       # 2000, Standard
STABILITY_SEED_B = 9000                            # zweiter, unabhaengiger Startseed


def rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def check_stability(fits: dict, cell_ids_by_backbone: dict, n_resamples: int = N_DEFAULT) -> tuple:
    """Rechnet den Kachel-Bootstrap zweimal (verschiedene Startseeds), vergleicht alle CI-Grenzen
    (oa+macro_f1, alle Configs) relativ. Gibt (max_rel_diff, details_df) zurueck."""
    samples_a = rlp.test_bootstrap_by_cell(fits, cell_ids_by_backbone, n_resamples=n_resamples,
                                            seed0=STABILITY_SEED_A)
    samples_b = rlp.test_bootstrap_by_cell(fits, cell_ids_by_backbone, n_resamples=n_resamples,
                                            seed0=STABILITY_SEED_B)
    rows = []
    max_rd = 0.0
    for name in fits:
        for metric in ("oa", "macro_f1"):
            a = samples_a[name][metric]
            b = samples_b[name][metric]
            lo_a, hi_a = np.percentile(a, [2.5, 97.5])
            lo_b, hi_b = np.percentile(b, [2.5, 97.5])
            rd_lo, rd_hi = rel_diff(lo_a, lo_b), rel_diff(hi_a, hi_b)
            max_rd = max(max_rd, rd_lo, rd_hi)
            rows.append({"config": name, "metric": metric,
                         "ci_a": (lo_a, hi_a), "ci_b": (lo_b, hi_b),
                         "rel_diff_lo": rd_lo, "rel_diff_hi": rd_hi})
    return max_rd, pd.DataFrame(rows)


def summarize(samples: dict) -> pd.DataFrame:
    rows = []
    for name, s in samples.items():
        for metric in ("oa", "macro_f1"):
            lo, hi = np.percentile(s[metric], [2.5, 97.5])
            rows.append({"config": name, "metric": metric, "mean": s[metric].mean(),
                         "ci_low": lo, "ci_high": hi, "width": hi - lo})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("=== Reproduziere Punktschaetzung (build_configs + run_probe, unveraendert) ===")
    configs = rlp.build_configs(BACKBONES)
    result, fits, _ = rlp.run_probe(configs, max_iter=1000)
    print("\nPunktschaetzung:")
    print(result[["config", "test_oa", "test_macro_f1"]].to_string(index=False))

    print("\n=== Zellzuordnung je Testpixel laden (persistiert, build_landuse_pixel_cells.py) ===")
    cell_ids_by_backbone = {}
    for bb in BACKBONES:
        cell_ids = rlp.load_test_pixel_cell_ids(bb)
        cell_ids_by_backbone[bb] = cell_ids
        n_unique = len(np.unique(cell_ids))
        print(f"  {bb}: {len(cell_ids)} Testpixel, {n_unique} eindeutige Zellen")

    print(f"\n=== Stabilitaetscheck Kachel-Bootstrap ({N_DEFAULT}x, zwei Startseeds "
          f"{STABILITY_SEED_A} vs. {STABILITY_SEED_B}) ===")
    max_rd, stability_df = check_stability(fits, cell_ids_by_backbone, n_resamples=N_DEFAULT)
    print(stability_df.to_string(index=False, float_format=lambda x: f"{x:.5f}" if isinstance(x, float) else x))
    print(f"\nMaximale relative Abweichung der CI-Grenzen zwischen den beiden Seeds: {max_rd:.4%}")

    n_resamples_final = N_DEFAULT
    if max_rd > 0.01:
        n_resamples_final = N_ESCALATED
        print(f"  -> ueber 1% Abweichung, erhoehe auf {N_ESCALATED} Ziehungen fuer die finale Auswertung.")
    else:
        print(f"  -> unter 1%, {N_DEFAULT} Ziehungen werden beibehalten.")

    print(f"\n=== Finaler Zeilen-Bootstrap ({N_DEFAULT}x, wie bisher) ===")
    row_samples = rlp.test_bootstrap(fits, n_resamples=N_DEFAULT)
    row_summary = summarize(row_samples)

    print(f"\n=== Finaler Kachel-Bootstrap ({n_resamples_final}x) ===")
    cell_samples = rlp.test_bootstrap_by_cell(fits, cell_ids_by_backbone, n_resamples=n_resamples_final,
                                               seed0=rlp.TEST_BOOTSTRAP_SEED0)
    cell_summary = summarize(cell_samples)

    row_diff = rlp.paired_diff(row_samples, backbones=BACKBONES)
    cell_diff = rlp.paired_diff(cell_samples, backbones=BACKBONES)

    # --- Gegenueberstellung ---
    # Die eigentliche Punktschaetzung ist test_oa/test_macro_f1 aus `result` (EIN Fit, VOR jedem
    # Bootstrap berechnet) - NICHT der Mittelwert der Bootstrap-Resamples (der hat naturgemaess
    # eine gewisse Bootstrap-Varianz, besonders beim Kachel-Bootstrap mit nur ~167 Clustern statt
    # hunderttausenden Zeilen). Beide Bootstrap-Varianten lesen fits[name]["y_test"]/["pred_test"]
    # UNVERAENDERT - die Punktschaetzung ist also per Konstruktion identisch, hier explizit
    # anhand von `result` verifiziert statt nur angenommen (Nutzervorgabe Phase 3).
    point_long = result.melt(id_vars=["config"], value_vars=["test_oa", "test_macro_f1"],
                              var_name="metric", value_name="point_estimate")
    point_long["metric"] = point_long["metric"].map({"test_oa": "oa", "test_macro_f1": "macro_f1"})

    merged = row_summary.merge(cell_summary, on=["config", "metric"], suffixes=("_row", "_cell"))
    merged = merged.merge(point_long, on=["config", "metric"])
    merged["width_factor_cell_over_row"] = merged["width_cell"] / merged["width_row"]
    merged["point_estimate_identical"] = True  # per Konstruktion (eine gemeinsame `point_estimate`-Spalte fuer beide)
    merged["bootstrap_mean_diff_row"] = merged["mean_row"] - merged["point_estimate"]
    merged["bootstrap_mean_diff_cell"] = merged["mean_cell"] - merged["point_estimate"]

    out_dir = cfg.DATA / "processed"
    merged.to_csv(out_dir / "landuse_probe_bootstrap_comparison.csv", index=False)
    row_diff.to_csv(out_dir / "landuse_probe_paired_diff_row_bootstrap.csv", index=False)
    cell_diff.to_csv(out_dir / "landuse_probe_paired_diff_cell_bootstrap.csv", index=False)
    stability_df.to_csv(out_dir / "landuse_probe_cell_bootstrap_stability.csv", index=False)

    print("\n=== Gegenueberstellung Zeilen- vs. Kachel-Bootstrap ===")
    print(f"(Punktschaetzung = test_oa/test_macro_f1 aus EINEM Fit, fuer beide Varianten identisch "
          f"per Konstruktion; bootstrap_mean_diff_* zeigt die Bootstrap-Verzerrung ggue. dieser "
          f"Punktschaetzung, informativ)")
    print(merged[["config", "metric", "point_estimate", "bootstrap_mean_diff_row", "bootstrap_mean_diff_cell",
                  "ci_low_row", "ci_high_row", "ci_low_cell", "ci_high_cell",
                  "width_row", "width_cell", "width_factor_cell_over_row"]]
          .to_string(index=False, float_format=lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x))

    print("\n=== Gepaarter Diff (trainiert - roh), Zeilen-Bootstrap ===")
    print(row_diff.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Gepaarter Diff (trainiert - roh), Kachel-Bootstrap ===")
    print(cell_diff.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\nMittlerer Breitenfaktor (Kachel/Zeile) ueber alle Configs/Metriken: "
          f"{merged['width_factor_cell_over_row'].mean():.4f}")

    print(f"\nGespeichert unter {out_dir}: landuse_probe_bootstrap_comparison.csv, "
          f"landuse_probe_paired_diff_{{row,cell}}_bootstrap.csv, "
          f"landuse_probe_cell_bootstrap_stability.csv")
    print(f"Gesamtlaufzeit: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
