"""
run_landuse_probe_heads_std.py

Alternative LUC-Koepfe (LogReg-Subsample, CatBoost, MLP) mit z-standardisierten Eingaben
(2026-10-02, Jakob). Analog zu `run_landuse_probe_std.py` (LogReg full) und
`build_sdm_phase7_metrics_seedmean_std.py`: jede Eingabedimension mit Mittelwert/Std des vollen
TRAIN-Splits standardisieren (auf Subsample/Val/Test angewendet), identisch fuer alle Arme.

Sonst identisch zu `run_landuse_probe_heads.py` (gleiche Konfigurationen, Subsample-Indizes,
Iterationen/Epochen-Limits, Batchgroessen, Seeds); TabPFN entfaellt (Lizenz-Blocker). Abweichungen:
- LogReg-Subsample: C-Grid (0.01, 0.1, 1, 10, 100) wie `run_landuse_probe_std.py`.
- Metriken mit fester Labelmenge K (`run_landuse_probe_prec_rec.scores`) statt sklearn ohne
  `labels=` (Phantomklasse 40000), alle vier Macro-Metriken + OA, Zell-Bootstrap 1000x und
  gepaarte Differenz trainiert - roh je Kopf/Regime.
CatBoost ist als Baumverfahren gegenueber affinen Transformationen pro Merkmal praktisch invariant -
dort ist keine Aenderung zu erwarten (Kontrolle). Ersetzt nichts - neue Dateien mit Suffix `_std`.
"""
import argparse
import time

import numpy as np
import pandas as pd

import run_landuse_probe as rlp
import run_landuse_probe_prec_rec as rpr
from eval_landuse import linear_probe
from eval_landuse_heads import catboost_probe, mlp_probe, stratified_subsample
from run_landuse_probe_heads import OUT_DIR, build_configs
from run_landuse_probe_std import C_GRID, standardize

METRICS = rpr.METRICS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True, choices=("tessera", "ae"))
    ap.add_argument("--catboost-iterations-sub", type=int, default=1000)
    ap.add_argument("--catboost-iterations-full", type=int, default=500)
    ap.add_argument("--mlp-epochs-sub", type=int, default=100)
    ap.add_argument("--mlp-epochs-full", type=int, default=30)
    args = ap.parse_args()

    print(f"=== Konfigurationen aufbauen ({args.backbone}) ===")
    configs, bc = build_configs(args.backbone)
    data_name = bc["data_name"]

    rows, fits = [], {}

    def record(arm, head, regime, n_train, res, extra, y_val, y_test):
        row = {"arm": arm, "head": head, "regime": regime, "n_train": n_train, "extra": extra}
        for part, yy, pp in (("val", y_val, res["pred_val"]), ("test", y_test, res["pred_test"])):
            row.update({f"{part}_{m}": v for m, v in rpr.scores(yy, pp, np.unique(yy)).items()})
        rows.append(row)
        fits[f"{head}_{regime}_{arm}"] = {"y_test": y_test, "pred_test": res["pred_test"]}
        print(f"  {head}/{regime} ({extra}): " + " ".join(f"test_{m}={row[f'test_{m}']:.4f}" for m in METRICS),
              flush=True)

    for arm, (X, y, split) in configs.items():
        tr, va, te = split == "train", split == "val", split == "test"
        Xs = standardize(X, tr)
        X_train, y_train = Xs[tr], y[tr]
        X_val, y_val, X_test, y_test = Xs[va], y[va], Xs[te], y[te]
        sub_idx = stratified_subsample(y_train)
        X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
        print(f"\n### Arm '{arm}' (dim={X.shape[1]}): train={len(X_train)} sub={len(sub_idx)} "
              f"val={len(X_val)} test={len(X_test)}", flush=True)

        best = None
        for C in C_GRID:
            res = linear_probe(X_sub, y_sub, X_val, y_val, X_test, y_test, C=C, max_iter=1000, return_fit=True)
            f1 = rpr.scores(y_val, res["pred_val"], np.unique(y_val))["macro_f1"]
            if best is None or f1 > best[0]:
                best = (f1, C, res)
        edge = " GRID-RAND" if best[1] in (C_GRID[0], C_GRID[-1]) else ""
        record(arm, "logreg", "subsample", len(sub_idx), best[2], f"C={best[1]}{edge}", y_val, y_test)

        for regime, (Xtr, ytr, iters) in (("subsample", (X_sub, y_sub, args.catboost_iterations_sub)),
                                           ("full", (X_train, y_train, args.catboost_iterations_full))):
            t0 = time.time()
            res = catboost_probe(Xtr, ytr, X_val, y_val, X_test, y_test, iterations=iters, return_fit=True)
            cap = " LIMIT" if res["best_iteration"] >= iters - 1 else ""
            record(arm, "catboost", regime, len(Xtr), res,
                   f"best_iter={res['best_iteration']}/{iters}{cap}, {time.time() - t0:.0f}s", y_val, y_test)

        for regime, (Xtr, ytr, epochs, bs) in (("subsample", (X_sub, y_sub, args.mlp_epochs_sub, 512)),
                                                ("full", (X_train, y_train, args.mlp_epochs_full, 8192))):
            t0 = time.time()
            res = mlp_probe(Xtr, ytr, X_val, y_val, X_test, y_test, max_epochs=epochs, batch_size=bs,
                            return_fit=True)
            cap = " LIMIT" if res["epochs_run"] >= epochs else ""
            record(arm, "mlp", regime, len(Xtr), res,
                   f"epochs={res['epochs_run']}/{epochs}{cap}, {time.time() - t0:.0f}s", y_val, y_test)

    result = pd.DataFrame(rows)
    out = OUT_DIR / f"landuse_probe_heads_results_{data_name}_std.csv"
    result.to_csv(out, index=False)
    print(f"\nGespeichert: {out}")
    print(result.sort_values(["head", "regime", "arm"]).to_string(index=False))

    print("\n=== Zellebenen-Bootstrap (Test, 1000x), roh + trainiert ===")
    boot_fits = {n: f for n, f in fits.items() if n.endswith("_roh") or n.endswith("_trainiert_projiziert")}
    cell_ids = rlp.load_test_pixel_cell_ids(data_name)
    labels_test = np.unique(next(iter(boot_fits.values()))["y_test"])
    samples = rpr.cell_bootstrap(boot_fits, cell_ids, labels_test)
    boot = pd.DataFrame([{"config": n, "metric": m, "mean": s[m].mean(),
                          **dict(zip(("ci_low", "ci_high"), np.percentile(s[m], [2.5, 97.5])))}
                         for n, s in samples.items() for m in METRICS])
    boot.to_csv(OUT_DIR / f"landuse_probe_heads_cell_bootstrap_{data_name}_std.csv", index=False)

    diff_rows = []
    for hr in sorted({n.rsplit("_", 1)[0] for n in boot_fits if n.endswith("_roh")}):
        tr_s, roh_s = samples[f"{hr}_trainiert_projiziert"], samples[f"{hr}_roh"]
        for m in METRICS:
            d = tr_s[m] - roh_s[m]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff_rows.append({"backbone": data_name, "head_regime": hr, "metric": m,
                              "mean_diff_trainiert_minus_roh": d.mean(), "ci_low_95": lo, "ci_high_95": hi,
                              "significant": bool(not (lo <= 0 <= hi))})
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(OUT_DIR / f"landuse_probe_heads_paired_diff_cell_{data_name}_std.csv", index=False)
    print("\n=== Gepaarte Differenz trainiert - roh ===")
    print(diff.to_string(index=False))


if __name__ == "__main__":
    main()
