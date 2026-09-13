"""
build_sdm_phase7_metrics.py

Phase 7 (INSTRUCTIONS_sdm.md): KL-Divergenz, L1-Distanz, Chebyshev-Distanz ueber das Testset,
mit Bootstrap-Konfidenzintervallen, fuer alle 4 Szenarien x 2 Backbones x 3 Arme x 5 Seeds
plus beide Baselines (Mittelwertsvorhersage, Random-Embedding).

Offene Entscheidungen (explizit getroffen):
  - KL-Richtung: KL(wahr || vorhergesagt). Begruendung: das Trainingsziel (Soft-Target-Cross-
    Entropy, build_sdm_phase6_predict.py::soft_ce_loss) ist bereits H(wahr)+KL(wahr||vorhergesagt) -
    das Modell wurde exakt auf diese Richtung hin optimiert.
  - Nullen in der wahren Verteilung: empirisch 0 Vorkommen bei 9cat UND 8cat (4.994 LSOAs) -
    geprueft, nicht angenommen (siehe check_zeros_in_true()). Kein Glaetten/Ausschliessen noetig.
    Nur eine numerische Sicherheitskonstante (EPS=1e-12) gegen Float-Rundung.
  - L1: Summe ueber Kategorien ist die primaere Zahl (vergleichbar mit AETHERs Tabellenwerten -
    siehe Sanity-Check-Bericht: L1-Mittel x Kategorienzahl = L1-Summe, bestaetigt den Faktor).
    L1-Mittel wird als zweite Spalte mitgefuehrt, keine eigene Herleitung mehr noetig.

Bootstrap-Cluster-Einheit (Nutzervorgabe): die 1-km-Zelle. Zuordnung LSOA->Zelle aus
sdm_lsoa_cell_overlay_london.csv - fuer JEDES LSOA die Zelle mit der GROESSTEN Ueberlappungsflaeche
(piece_area_m2), also dieselbe Mehrheitsregel wie fuer die LSOA->Split-Zuordnung in Phase 3, nur
jetzt auf Zellebene statt auf Split-Label-Ebene angewendet (siehe majority_cell_per_lsoa()).
Fuer den Hauptlauf zusaetzlich ein einfacher LSOA-Bootstrap (keine Cluster) zum direkten
Intervallbreiten-Vergleich - Beleg, dass die Cluster-Einheit einen Unterschied macht.
"""
import time

import numpy as np
import pandas as pd

import config as cfg
from build_sdm_phase6_predict import SCENARIOS, Scenario, build_xy, fit_mlp

SEEDS = [0, 1, 2, 3, 4]
EPS = 1e-12
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED0 = 2000
OVERLAY_PATH = cfg.DATA / "splits" / "sdm_lsoa_cell_overlay_london.csv"


def majority_cell_per_lsoa() -> pd.Series:
    ov = pd.read_csv(OVERLAY_PATH)
    idx = ov.groupby("lsoa_code")["piece_area_m2"].idxmax()
    maj = ov.loc[idx, ["lsoa_code", "cell_row", "cell_col"]].copy()
    maj["cell_id"] = maj["cell_row"].astype(str) + "_" + maj["cell_col"].astype(str)
    return maj.set_index("lsoa_code")["cell_id"]


LSOA_TO_CELL = majority_cell_per_lsoa()


def check_zeros_in_true(y: np.ndarray, label: str) -> int:
    n_zero_rows = int((y == 0).any(axis=1).sum())
    print(f"  [{label}] Zeilen mit exakter 0 in der wahren Verteilung: {n_zero_rows} von {len(y)}")
    return n_zero_rows


def kl_true_pred(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true * np.log((y_true + EPS) / (y_pred + EPS))).sum(axis=1)


def l1_sum(y_true, y_pred) -> np.ndarray:
    return np.abs(y_true - y_pred).sum(axis=1)


def l1_mean(y_true, y_pred) -> np.ndarray:
    return np.abs(y_true - y_pred).mean(axis=1)


def chebyshev(y_true, y_pred) -> np.ndarray:
    return np.abs(y_true - y_pred).max(axis=1)


def per_lsoa_metrics(y_test: np.ndarray, pred: np.ndarray) -> dict:
    return {"kl": kl_true_pred(y_test, pred), "l1_sum": l1_sum(y_test, pred),
            "l1_mean": l1_mean(y_test, pred), "chebyshev": chebyshev(y_test, pred)}


def cluster_bootstrap(values: np.ndarray, cell_ids: np.ndarray, n_resamples=N_BOOTSTRAP,
                       seed=BOOTSTRAP_SEED0) -> np.ndarray:
    """Resampled die EINDEUTIGEN Zellen mit Zuruecklegen (Standard-Cluster-Bootstrap) - eine
    resampelte Zelle bringt ALLE ihre LSOAs mit (mehrfach, falls die Zelle mehrfach gezogen
    wird). Statistik je Resample: Mittelwert ueber alle so gesammelten LSOA-Werte."""
    unique_cells = np.unique(cell_ids)
    cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}
    rng = np.random.default_rng(seed)
    n_cells = len(unique_cells)
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled = rng.choice(unique_cells, size=n_cells, replace=True)
        idx = np.concatenate([cell_to_idx[c] for c in sampled])
        out[i] = values[idx].mean()
    return out


def simple_bootstrap(values: np.ndarray, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(values)
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        out[i] = values[idx].mean()
    return out


def ci(samples: np.ndarray, alpha=0.05) -> tuple:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate_config(backbone: str, arm: str, scenario: Scenario, seeds=SEEDS,
                     bootstrap_seed_idx: int = 0, do_lsoa_bootstrap: bool = False) -> dict:
    X, y, split_arr, n_classes, lsoa_codes = build_xy(backbone, arm, scenario)
    d_in = X.shape[1]

    per_seed_point = []
    ref_pred, ref_y_test, ref_lsoa = None, None, None
    for i, s in enumerate(seeds):
        r = fit_mlp(X, y, split_arr, s, d_in, n_classes, lsoa_codes=lsoa_codes)
        m = per_lsoa_metrics(r["y_test"], r["test_pred"])
        per_seed_point.append({k: v.mean() for k, v in m.items()})
        if i == bootstrap_seed_idx:
            ref_pred, ref_y_test, ref_lsoa = r["test_pred"], r["y_test"], r["test_lsoa"]

    point = pd.DataFrame(per_seed_point)
    m_ref = per_lsoa_metrics(ref_y_test, ref_pred)
    cell_ids = np.array([LSOA_TO_CELL[c] for c in ref_lsoa])

    result = {"point_mean": point.mean().to_dict(), "point_std": point.std().to_dict()}
    for metric_name, vec in m_ref.items():
        samples = cluster_bootstrap(vec, cell_ids)
        result[f"{metric_name}_ci_tile"] = ci(samples)
        if do_lsoa_bootstrap:
            samples_lsoa = simple_bootstrap(vec)
            result[f"{metric_name}_ci_lsoa"] = ci(samples_lsoa)
    return result


def evaluate_mean_baseline(scenario: Scenario, do_lsoa_bootstrap: bool = False) -> dict:
    _, y, split_arr, n_classes, lsoa_codes = build_xy("tessera", "a", scenario)
    train_mean = y[split_arr == "train"].mean(axis=0)
    train_mean = train_mean / train_mean.sum()
    test_m = split_arr == "test"
    y_test = y[test_m]
    pred = np.tile(train_mean, (len(y_test), 1))
    m = per_lsoa_metrics(y_test, pred)
    cell_ids = np.array([LSOA_TO_CELL[c] for c in lsoa_codes[test_m]])
    result = {"point_mean": {k: v.mean() for k, v in m.items()},
              "point_std": {k: 0.0 for k in m}}
    for metric_name, vec in m.items():
        result[f"{metric_name}_ci_tile"] = ci(cluster_bootstrap(vec, cell_ids))
        if do_lsoa_bootstrap:
            result[f"{metric_name}_ci_lsoa"] = ci(simple_bootstrap(vec))
    return result


def evaluate_random_embedding_baseline(backbone: str, arm: str, scenario: Scenario, seeds=SEEDS,
                                        bootstrap_seed_idx: int = 0,
                                        do_lsoa_bootstrap: bool = False) -> dict:
    _, y, split_arr, n_classes, lsoa_codes = build_xy(backbone, arm, scenario)
    # Dimensionalitaet dimensionsangepasst wie in Phase 6 - aus der jeweiligen Pooled-Datei
    from build_sdm_phase6_predict import load_pooled
    prefix = "pure_" if scenario.pooling == "pure" else "all_"
    d_in = len([c for c in load_pooled(backbone, arm).columns if c.startswith(prefix)])

    per_seed_point = []
    ref_pred, ref_y_test, ref_lsoa = None, None, None
    n = len(y)
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        X_rand = rng.standard_normal((n, d_in)).astype(np.float32)
        r = fit_mlp(X_rand, y, split_arr, s, d_in, n_classes, lsoa_codes=lsoa_codes)
        m = per_lsoa_metrics(r["y_test"], r["test_pred"])
        per_seed_point.append({k: v.mean() for k, v in m.items()})
        if i == bootstrap_seed_idx:
            ref_pred, ref_y_test, ref_lsoa = r["test_pred"], r["y_test"], r["test_lsoa"]

    point = pd.DataFrame(per_seed_point)
    m_ref = per_lsoa_metrics(ref_y_test, ref_pred)
    cell_ids = np.array([LSOA_TO_CELL[c] for c in ref_lsoa])
    result = {"point_mean": point.mean().to_dict(), "point_std": point.std().to_dict()}
    for metric_name, vec in m_ref.items():
        result[f"{metric_name}_ci_tile"] = ci(cluster_bootstrap(vec, cell_ids))
        if do_lsoa_bootstrap:
            result[f"{metric_name}_ci_lsoa"] = ci(simple_bootstrap(vec))
    return result


def fmt_row(label: str, res: dict) -> dict:
    pm, ps = res["point_mean"], res["point_std"]
    row = {
        "model": label,
        "kl": pm["kl"], "kl_std": ps["kl"], "kl_ci_tile": res.get("kl_ci_tile"),
        "l1_sum": pm["l1_sum"], "l1_sum_std": ps["l1_sum"], "l1_sum_ci_tile": res.get("l1_sum_ci_tile"),
        "l1_mean": pm["l1_mean"], "l1_mean_std": ps["l1_mean"],
        "chebyshev": pm["chebyshev"], "chebyshev_std": ps["chebyshev"],
        "chebyshev_ci_tile": res.get("chebyshev_ci_tile"),
    }
    if "kl_ci_lsoa" in res:
        row["kl_ci_lsoa"] = res["kl_ci_lsoa"]
        row["l1_sum_ci_lsoa"] = res["l1_sum_ci_lsoa"]
        row["chebyshev_ci_lsoa"] = res["chebyshev_ci_lsoa"]
    return row


def main():
    t0 = time.time()
    all_rows = []

    for si, scenario in enumerate(SCENARIOS):
        is_main = si == 0
        print(f"\n{'=' * 12} Szenario: {scenario.name} {'=' * 12}")

        mb = evaluate_mean_baseline(scenario, do_lsoa_bootstrap=is_main)
        row = fmt_row("Mittelwertsvorhersage", mb)
        row.update({"scenario": scenario.name, "backbone": "-", "arm": "-"})
        all_rows.append(row)
        print(f"  Mittelwertsvorhersage: KL={mb['point_mean']['kl']:.4f} "
              f"(CI Kachel {mb['kl_ci_tile']})")

        for backbone in ["ae", "tessera"]:
            for arm in ["a", "b", "c"]:
                r = evaluate_config(backbone, arm, scenario, do_lsoa_bootstrap=is_main)
                row = fmt_row(f"{backbone}/{arm}", r)
                row.update({"scenario": scenario.name, "backbone": backbone, "arm": arm})
                all_rows.append(row)
                print(f"  {backbone}/{arm}: KL={r['point_mean']['kl']:.4f}+-{r['point_std']['kl']:.4f} "
                      f"(CI Kachel {r['kl_ci_tile']})"
                      + (f" (CI LSOA {r.get('kl_ci_lsoa')})" if is_main else ""))

                rb = evaluate_random_embedding_baseline(backbone, arm, scenario, do_lsoa_bootstrap=is_main)
                row = fmt_row(f"Random-Embedding ({backbone}/{arm}-dim)", rb)
                row.update({"scenario": scenario.name, "backbone": backbone, "arm": arm})
                all_rows.append(row)
                print(f"    Random-Embedding: KL={rb['point_mean']['kl']:.4f} "
                      f"(CI Kachel {rb['kl_ci_tile']})")

    results = pd.DataFrame(all_rows)
    out_path = cfg.DATA / "processed" / "sdm_phase7_metrics.csv"
    results.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")
    print(f"Gesamtlaufzeit: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
