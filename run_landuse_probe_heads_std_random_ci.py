"""
run_landuse_probe_heads_std_random_ci.py

Nachtrag (2026-10-02, Jakob): Zell-Bootstrap-CIs fuer den Arm 'zufaellig_projiziert' der
alternativen Koepfe. `run_landuse_probe_heads_std.py` hat diesen Arm nur als Punktschaetzung
gerechnet und seine Vorhersagen nicht gespeichert -> Refit NUR dieses Arms mit identischem
Protokoll (gleiche Standardisierung, Subsample, C-Grid, Iterationen/Epochen, Seeds), Abgleich der
Testmetriken gegen `landuse_probe_heads_results_{data_name}_std.csv` (Abweichungen werden
gemeldet, nicht verschluckt - CatBoost/MLP laufen auf GPU), dann `rpr.cell_bootstrap()` mit
denselben Zellziehungen (gleicher Startseed). Vorhersagen werden diesmal persistiert.
Ersetzt nichts - neue Datei `landuse_probe_heads_cell_bootstrap_{data_name}_std_random.csv`.
"""
import argparse

import numpy as np
import pandas as pd

import run_landuse_probe as rlp
import run_landuse_probe_prec_rec as rpr
from eval_landuse import linear_probe
from eval_landuse_heads import catboost_probe, mlp_probe, stratified_subsample
from run_landuse_probe_heads import OUT_DIR, build_configs
from run_landuse_probe_std import C_GRID, standardize

ARM = "zufaellig_projiziert"
METRICS = rpr.METRICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=("tessera", "ae"))
    args = ap.parse_args()
    configs, bc = build_configs(args.backbone)
    data_name = bc["data_name"]
    stored = pd.read_csv(OUT_DIR / f"landuse_probe_heads_results_{data_name}_std.csv")
    stored = stored[stored["arm"] == ARM].set_index(["head", "regime"])

    X, y, split = configs[ARM]
    tr, va, te = split == "train", split == "val", split == "test"
    Xs = standardize(X, tr)
    X_train, y_train, X_val, y_val, X_test, y_test = Xs[tr], y[tr], Xs[va], y[va], Xs[te], y[te]
    sub_idx = stratified_subsample(y_train)
    X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]

    preds = {}
    best = None
    for C in C_GRID:
        res = linear_probe(X_sub, y_sub, X_val, y_val, X_test, y_test, C=C, max_iter=1000, return_fit=True)
        f1 = rpr.scores(y_val, res["pred_val"], np.unique(y_val))["macro_f1"]
        if best is None or f1 > best[0]:
            best = (f1, res)
    preds[("logreg", "subsample")] = best[1]["pred_test"]
    for regime, (Xtr, ytr, iters) in (("subsample", (X_sub, y_sub, 1000)), ("full", (X_train, y_train, 500))):
        preds[("catboost", regime)] = catboost_probe(Xtr, ytr, X_val, y_val, X_test, y_test, iterations=iters,
                                                     return_fit=True)["pred_test"]
    for regime, (Xtr, ytr, epochs, bs) in (("subsample", (X_sub, y_sub, 100, 512)),
                                            ("full", (X_train, y_train, 30, 8192))):
        preds[("mlp", regime)] = mlp_probe(Xtr, ytr, X_val, y_val, X_test, y_test, max_epochs=epochs,
                                           batch_size=bs, return_fit=True)["pred_test"]

    labels_test = np.unique(y_test)
    for key, p in preds.items():
        now = rpr.scores(y_test, p, labels_test)
        dev = max(abs(now[m] - stored.loc[key, f"test_{m}"]) for m in METRICS)
        print(f"  {key[0]}/{key[1]}: test_macro_f1={now['macro_f1']:.4f} "
              f"(gespeichert {stored.loc[key, 'test_macro_f1']:.4f}), max. Abweichung {dev:.2e}"
              + ("" if dev < 1e-9 else "  <-- NICHT exakt reproduziert"), flush=True)

    np.savez(OUT_DIR / f"landuse_probe_heads_preds_{data_name}_std_random.npz", y_test=y_test,
             **{f"{h}_{r}": p for (h, r), p in preds.items()})

    cell_ids = rlp.load_test_pixel_cell_ids(data_name)
    fits = {f"{h}_{r}_{ARM}": {"y_test": y_test, "pred_test": p} for (h, r), p in preds.items()}
    samples = rpr.cell_bootstrap(fits, cell_ids, labels_test)
    boot = pd.DataFrame([{"config": n, "metric": m, "mean": s[m].mean(),
                          **dict(zip(("ci_low", "ci_high"), np.percentile(s[m], [2.5, 97.5])))}
                         for n, s in samples.items() for m in METRICS])
    out = OUT_DIR / f"landuse_probe_heads_cell_bootstrap_{data_name}_std_random.csv"
    boot.to_csv(out, index=False)
    print(f"Gespeichert: {out}")
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
