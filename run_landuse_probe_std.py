"""
run_landuse_probe_std.py

LUC-LogReg-Probe mit z-standardisierten Eingaben (2026-10-01, Jakob). Analog zum SDM-Fix
(`build_sdm_phase7_metrics_seedmean_std.py`): die Arme haben stark unterschiedliche Eingabeskalen
(roh z-score vs. L2-normierte 256-dim Projektionen), TESSERA waehlte im alten Grid (0.1, 1, 10)
in allen Armen den Rand C=10. Regularisierungsstaerke ist damit skalenabhaengig zwischen Armen.

Aenderungen gegenueber run_landuse_probe_{tessera,ae}_v1dedup.py:
- Jede Eingabedimension mit Mittelwert/Std der TRAIN-Pixel standardisieren (auf Val/Test angewendet).
- C-Grid (0.01, 0.1, 1, 10, 100), Auswahl wie bisher per val_macro_f1; Randtreffer werden gemeldet.
Sonst identisch: dieselben Konfigurationen (build_configs_dedup), Split, MAX_ITER, Macro-Metriken
mit fester Labelmenge K, Zell-Bootstrap 1000x und gepaarte Differenz trainiert - roh
(Funktionen aus run_landuse_probe_prec_rec.py). Ersetzt nichts - neue Dateien mit Suffix `_std`.
"""
import argparse
import importlib
import time

import numpy as np
import pandas as pd

import run_landuse_probe as rlp
import run_landuse_probe_prec_rec as rpr
from eval_landuse import linear_probe

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
METRICS = rpr.METRICS


def standardize(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mu = X[train_mask].mean(axis=0)
    sd = X[train_mask].std(axis=0) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=("tessera", "ae"), required=True)
    args = ap.parse_args()
    drv = importlib.import_module(f"run_landuse_probe_{args.backbone}_v1dedup")
    data_name, out_dir = drv.DATA_NAME, drv.OUT_DIR

    print("=== Konfigurationen aufbauen ===")
    configs = drv.build_configs_dedup()

    rows, preds = [], {}
    for name, (X, y, split) in configs.items():
        tr, va, te = split == "train", split == "val", split == "test"
        Xs = standardize(X, tr)
        print(f"\n=== {name} === dim={X.shape[1]} Std/Dim (Median) roh={np.median(X[tr].std(0)):.4f} -> 1")
        best = None
        for C in C_GRID:
            t0 = time.time()
            res = linear_probe(Xs[tr], y[tr], Xs[va], y[va], Xs[te], y[te], C=C,
                               max_iter=drv.MAX_ITER, return_fit=True)
            labels_val = np.unique(y[va])
            val_f1 = rpr.scores(y[va], res["pred_val"], labels_val)["macro_f1"]
            print(f"  C={C}: val_macro_f1(K)={val_f1:.4f} n_iter={res['clf'].n_iter_.max()} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if best is None or val_f1 > best[0]:
                best = (val_f1, C, res)
        _, C, res = best
        if C in (C_GRID[0], C_GRID[-1]):
            print(f"  WARNUNG: {name} waehlt Grid-Rand C={C}")
        p = {"y_val": y[va], "pred_val": res["pred_val"], "y_test": y[te], "pred_test": res["pred_test"]}
        preds[name] = p
        labels_test = np.unique(p["y_test"])
        rpr.self_test(p["y_test"], p["pred_test"], labels_test)
        row = {"config": name, "dim": X.shape[1], "C": C, "n_iter": int(res["clf"].n_iter_.max()),
               "n_classes_test": len(labels_test)}
        for part, yy, pp in (("val", p["y_val"], p["pred_val"]), ("test", p["y_test"], p["pred_test"])):
            row.update({f"{part}_{m}": v for m, v in rpr.scores(yy, pp, np.unique(yy)).items()})
        rows.append(row)
        print(f"  -> C={C}: " + " ".join(f"test_{m}={row[f'test_{m}']:.4f}" for m in METRICS))

    np.savez(out_dir / f"landuse_probe_preds_{data_name}_ckpt_std.npz", names=np.array(list(preds)),
             **{f"{n}__{k}": v for n, d in preds.items() for k, v in d.items()})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / f"landuse_probe_prec_rec_{data_name}_ckpt_std.csv", index=False)
    print(result.to_string(index=False))

    print("\n=== Zellebenen-Bootstrap (Test, 1000x) ===")
    cell_ids = rlp.load_test_pixel_cell_ids(data_name)
    labels_test = np.unique(next(iter(preds.values()))["y_test"])
    fits = {n: {"y_test": p["y_test"], "pred_test": p["pred_test"]} for n, p in preds.items()}
    samples = rpr.cell_bootstrap(fits, cell_ids, labels_test)
    boot = pd.DataFrame([{"config": n, "metric": m, "mean": s[m].mean(),
                          **dict(zip(("ci_low", "ci_high"), np.percentile(s[m], [2.5, 97.5])))}
                         for n, s in samples.items() for m in METRICS])
    boot.to_csv(out_dir / f"landuse_probe_prec_rec_cell_bootstrap_{data_name}_ckpt_std.csv", index=False)
    print(boot.to_string(index=False))

    trained, roh = samples[f"{data_name}_trainiert_projiziert"], samples[f"{data_name}_roh"]
    diff_rows = []
    for m in METRICS:
        d = trained[m] - roh[m]
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff_rows.append({"backbone": data_name, "metric": m, "mean_diff_trainiert_minus_roh": d.mean(),
                          "ci_low_95": lo, "ci_high_95": hi, "significant": bool(not (lo <= 0 <= hi))})
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(out_dir / f"landuse_probe_prec_rec_paired_diff_cell_{data_name}_ckpt_std.csv", index=False)
    print("\n=== Gepaarte Differenz trainiert - roh ===")
    print(diff.to_string(index=False))


if __name__ == "__main__":
    main()
