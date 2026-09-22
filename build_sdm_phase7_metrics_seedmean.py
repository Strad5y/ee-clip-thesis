"""
build_sdm_phase7_metrics_seedmean.py

Seed-Inkonsistenz im SDM-Zellebenen-Bootstrap beheben (2026-09-22, Jakob): die bisherige
Haupttabelle (`build_sdm_phase7_metrics.py::evaluate_config()`) berichtet als Punktwert das Mittel
ueber 5 Probe-Seeds, das zugehoerige 95%-CI aber ausschliesslich aus Seed 0
(`bootstrap_seed_idx=0`) - Punktwert und CI beschreiben nicht dieselbe Groesse (siehe
`results/backbone_comparison_dedup.md`, Abschnitt 7 "Zahlenbasis"). Dieses Skript berechnet
stattdessen fuer jede Ziehung des Zellebenen-Bootstraps die Metrik je der 5 Probe-Seeds und
mittelt danach ueber die Seeds - Punktwert (5-Seed-Mittel ueber alle Testzellen) und CI (5-Seed-
Mittel je Resample) beschreiben jetzt dieselbe Groesse. Gleiche Ziehungszahl (1000) und
Startseed (2000) wie bisher (`bsp7.N_BOOTSTRAP`/`bsp7.BOOTSTRAP_SEED0`) - dieselbe Zell-Ziehungs-
folge wie im bisherigen Einzelseed-Bootstrap (ein `rng.choice(...)`-Aufruf je Resample, gleiche
Groesse -> identische Zufallszahlenfolge fuer denselben Seed).

KEIN Refit im Sinne einer neuen/geaenderten Trainingsprozedur - `bsp6.fit_mlp()` ist bei
gegebenem Seed deterministisch (`torch.manual_seed(seed)` innerhalb der Funktion), ein erneuter
Aufruf reproduziert exakt dieselben Vorhersagen wie in der Haupttabelle/den bisherigen Diagnosen.
Es werden lediglich alle 5 (statt nur 1) der ohnehin bei jedem Aufruf gefitteten Seeds behalten,
statt 4 davon zu verwerfen (wie es `evaluate_config()`/`per_seed_metrics_and_ref()` bisher tun) -
reine Kosten weniger Sekunden je Arm/Backbone (MLP-Training ist winzig, siehe Phase-6-Laufzeiten).

Rechnet fuer BEIDE Backbones, ALLE DREI Arme, ALLE VIER Metriken neu. TESSERA Arm A/B nutzen die
bereits nachgezogene Dedup-Statistik (`sdm_pooled_tessera_arm{a,b}_dedupnorm.parquet`), Arm C wie
gehabt (`sdm_pooled_tessera_armc_v1dedup_ckpt.parquet`); AE alle drei Arme wie in Schritt 3
(`sdm_pooled_ae_arm{a,b,c}_v1dedup_ckpt.parquet`). Ersetzt nichts - neue Dateien mit Suffix
`_seedmean`.
"""
import numpy as np
import pandas as pd

import config as cfg
import build_sdm_phase6_predict as bsp6
import build_sdm_phase7_metrics as bsp7

POOLED = {
    ("tessera", "a"): cfg.DATA / "processed" / "sdm_pooled_tessera_arma_dedupnorm.parquet",
    ("tessera", "b"): cfg.DATA / "processed" / "sdm_pooled_tessera_armb_dedupnorm.parquet",
    ("tessera", "c"): cfg.DATA / "processed" / "sdm_pooled_tessera_armc_v1dedup_ckpt.parquet",
    ("ae", "a"): cfg.DATA / "processed" / "sdm_pooled_ae_arma_v1dedup_ckpt.parquet",
    ("ae", "b"): cfg.DATA / "processed" / "sdm_pooled_ae_armb_v1dedup_ckpt.parquet",
    ("ae", "c"): cfg.DATA / "processed" / "sdm_pooled_ae_armc_v1dedup_ckpt.parquet",
}
ARMS = ["a", "b", "c"]
BACKBONES = ["tessera", "ae"]
METRICS = ["kl", "l1_sum", "l1_mean", "chebyshev"]
SCENARIO = bsp6.Scenario("Hauptlauf (9cat/A/pure, Seed-Mittel-Bootstrap)")
OUT_DIR = cfg.DATA / "processed"


def load_pooled_variant(backbone: str, arm: str) -> pd.DataFrame:
    return pd.read_parquet(POOLED[(backbone, arm)])


bsp6.load_pooled = load_pooled_variant


def fit_all_seeds(backbone: str, arm: str, scenario) -> tuple:
    """Fittet alle 5 Probe-Seeds (deterministisch, kein neuer Zufall), gibt je Metrik eine
    Liste von 5 per-LSOA-Arrays zurueck (Reihenfolge = SEEDS), plus die zugehoerigen LSOA-Codes
    (identisch ueber alle Seeds - Test-Split haengt nicht vom Seed ab, wird unten geprueft)."""
    bsp6.load_pooled = load_pooled_variant  # defensiv, siehe Bugfix-Muster aus der AE-Aufgabe
    X, y, split_arr, n_classes, lsoa_codes = bsp6.build_xy(backbone, arm, scenario)
    d_in = X.shape[1]
    per_seed = {m: [] for m in METRICS}
    ref_lsoa = None
    for s in bsp7.SEEDS:
        r = bsp6.fit_mlp(X, y, split_arr, s, d_in, n_classes, lsoa_codes=lsoa_codes)
        if ref_lsoa is None:
            ref_lsoa = r["test_lsoa"]
        else:
            assert list(r["test_lsoa"]) == list(ref_lsoa), (
                f"{backbone}/{arm}: Test-LSOA-Reihenfolge weicht zwischen Seed {s} und "
                f"Seed {bsp7.SEEDS[0]} ab - Annahme (seed-unabhaengiger Test-Split) verletzt."
            )
        m = bsp7.per_lsoa_metrics(r["y_test"], r["test_pred"])
        for metric in METRICS:
            per_seed[metric].append(m[metric])
    return per_seed, ref_lsoa


def cluster_bootstrap_seedmean(values_per_seed: list, cell_ids: np.ndarray,
                                n_resamples: int = bsp7.N_BOOTSTRAP,
                                seed: int = bsp7.BOOTSTRAP_SEED0) -> np.ndarray:
    """Wie bsp7.cluster_bootstrap(), aber je Resample wird die Metrik fuer JEDEN der 5 Seed-
    Arrays auf denselben gezogenen Zellen berechnet und ueber die 5 Seeds gemittelt - identische
    Zieh-Sequenz (ein rng.choice()-Aufruf je Resample, gleiche Zellmenge/-groesse) wie im
    bisherigen Einzelseed-Bootstrap fuer denselben Seed."""
    unique_cells = np.unique(cell_ids)
    cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}
    rng = np.random.default_rng(seed)
    n_cells = len(unique_cells)
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled = rng.choice(unique_cells, size=n_cells, replace=True)
        idx = np.concatenate([cell_to_idx[c] for c in sampled])
        seed_means = [v[idx].mean() for v in values_per_seed]
        out[i] = float(np.mean(seed_means))
    return out


def point_seedmean(values_per_seed: list) -> float:
    """5-Seed-Mittel ueber ALLE Testzellen (unresampled) - dieselbe Groesse, die jetzt auch der
    Bootstrap beschreibt (anders als bisher, wo das CI nur Seed 0 beschrieb)."""
    return float(np.mean([v.mean() for v in values_per_seed]))


def evaluate_backbone_arm(backbone: str, arm: str) -> dict:
    per_seed, lsoa = fit_all_seeds(backbone, arm, SCENARIO)
    cell_ids = np.array([bsp7.LSOA_TO_CELL[c] for c in lsoa])
    result = {"n_test_lsoa": len(lsoa)}
    for metric in METRICS:
        result[metric] = point_seedmean(per_seed[metric])
        samples = cluster_bootstrap_seedmean(per_seed[metric], cell_ids)
        result[f"{metric}_ci"] = bsp7.ci(samples)
        result[f"{metric}_samples"] = samples  # fuer gepaarte Differenzen weiterverwendet
    result["per_seed"] = per_seed
    result["lsoa"] = lsoa
    result["cell_ids"] = cell_ids
    return result


def main():
    all_main_rows = []
    all_paired_rows = []
    backbone_diff_rows = []
    per_backbone_results = {}

    mb = bsp7.evaluate_mean_baseline(SCENARIO, do_lsoa_bootstrap=False)
    print(f"Mittelwertsvorhersage (seed-unabhaengig, unveraendert): KL={mb['point_mean']['kl']:.4f} "
          f"(CI Zelle {mb['kl_ci_tile']})")
    mb_row = {"model": "Mittelwertsvorhersage", "backbone": "-", "arm": "-"}
    for metric in METRICS:
        mb_row[metric] = mb["point_mean"][metric]
        mb_row[f"{metric}_ci"] = mb[f"{metric}_ci_tile"]
    all_main_rows.append(mb_row)

    for backbone in BACKBONES:
        per_backbone_results[backbone] = {}
        for arm in ARMS:
            print(f"\n=== {backbone}/{arm}: 5 Seeds fitten + Seed-Mittel-Bootstrap ({bsp7.N_BOOTSTRAP}x) ===")
            res = evaluate_backbone_arm(backbone, arm)
            per_backbone_results[backbone][arm] = res
            row = {"model": f"{backbone}/{arm}", "backbone": backbone, "arm": arm,
                   "n_test_lsoa": res["n_test_lsoa"]}
            for metric in METRICS:
                row[metric] = res[metric]
                row[f"{metric}_ci"] = res[f"{metric}_ci"]
            all_main_rows.append(row)
            print(f"  {backbone}/{arm}: " + ", ".join(
                f"{m}={res[m]:.5f} CI={res[f'{m}_ci']}" for m in METRICS))

        # Gepaarter C-A (Differenz pro Ziehung aus den 5-Seed-Mitteln)
        res_a, res_c = per_backbone_results[backbone]["a"], per_backbone_results[backbone]["c"]
        assert list(res_a["lsoa"]) == list(res_c["lsoa"]), \
            f"{backbone}: Test-LSOA-Mengen weichen zwischen Arm A/C ab."
        print(f"\n=== {backbone}: gepaarter Zell-Bootstrap C-A (Seed-Mittel) ===")
        for metric in METRICS:
            point_diff = res_c[metric] - res_a[metric]
            diffs = res_c[f"{metric}_samples"] - res_a[f"{metric}_samples"]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            frac_pos = float((diffs > 0).mean())
            sig = bool(not (lo <= 0 <= hi))
            all_paired_rows.append({"backbone": backbone, "metric": metric,
                                    "point_diff_c_minus_a": point_diff, "ci_low": lo, "ci_high": hi,
                                    "frac_resamples_positive": frac_pos, "significant": sig,
                                    "n_test_lsoa": res_a["n_test_lsoa"]})
            print(f"  {metric}: Punktdiff (C-A)={point_diff:+.5f}  95%-CI=[{lo:+.5f}, {hi:+.5f}]  "
                  f"signifikant={sig}")

    # Backbone-Differenz (TESSERA - AE) je Arm, aus den 5-Seed-Mitteln
    print("\n=== Backbone-Differenz (TESSERA - AE), Seed-Mittel ===")
    for arm in ARMS:
        rt = per_backbone_results["tessera"][arm]
        ra = per_backbone_results["ae"][arm]
        codes_t, codes_a = set(rt["lsoa"]), set(ra["lsoa"])
        common = codes_t & codes_a
        assert common == codes_t == codes_a, f"Arm {arm}: Test-LSOA-Mengen weichen ab."
        for metric in METRICS:
            point_diff = rt[metric] - ra[metric]
            diffs = rt[f"{metric}_samples"] - ra[f"{metric}_samples"]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            sig = bool(not (lo <= 0 <= hi))
            backbone_diff_rows.append({"arm": arm, "metric": metric, "tessera_dedup": rt[metric],
                                       "ae_dedup": ra[metric], "point_diff_tessera_minus_ae": point_diff,
                                       "ci_low": lo, "ci_high": hi, "significant": sig,
                                       "n_test_lsoa": len(common)})
            print(f"  [{arm}/{metric}] TESSERA-AE = {point_diff:+.5f}  95%-CI=[{lo:+.5f}, {hi:+.5f}]  "
                  f"signifikant={sig}")

    main_df = pd.DataFrame(all_main_rows)
    for backbone in BACKBONES:
        sub = main_df[main_df["backbone"].isin([backbone, "-"])]
        out_path = OUT_DIR / f"sdm_phase7_metrics_{backbone}_seedmean.csv"
        sub.to_csv(out_path, index=False)
        print(f"\nGespeichert: {out_path}")

    paired_df = pd.DataFrame(all_paired_rows)
    for backbone in BACKBONES:
        out_path = OUT_DIR / f"sdm_paired_bootstrap_{backbone}_seedmean_c_minus_a.csv"
        paired_df[paired_df["backbone"] == backbone].to_csv(out_path, index=False)
        print(f"Gespeichert: {out_path}")

    diff_out = OUT_DIR / "diag_backbone_paired_bootstrap_sdm_seedmean.csv"
    pd.DataFrame(backbone_diff_rows).to_csv(diff_out, index=False)
    print(f"Gespeichert: {diff_out}")


if __name__ == "__main__":
    main()
