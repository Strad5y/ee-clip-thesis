"""
run_landuse_probe_prec_rec.py

Nachtrag (2026-10-01, Jakob): AETHER berichtet fuer LUC Macro-F1, Macro-Precision und
Macro-Recall - wir bisher nur OA + Macro-F1. Dieses Skript ergaenzt Macro-Precision und
Macro-Recall fuer die bestehende LogReg-Probe (run_landuse_probe_{tessera,ae}_v1dedup.py),
OHNE deren Methodik zu aendern:

- Vorhersagen wurden damals nicht persistiert -> Refit beim bereits per val_macro_f1 gewaehlten
  C aus landuse_probe_results_{backbone}_dedup_ckpt.csv (keine neue C-Suche).
- LogisticRegression+lbfgs ist bei fixen Daten deterministisch -> test_oa/test_macro_f1 MUESSEN
  die gespeicherten Werte reproduzieren; sonst Abbruch (Assert), statt neue Zahlen zu mischen.
- Zellebenen-Bootstrap (1000x, Seed rlp.TEST_BOOTSTRAP_SEED0) mit denselben Resample-Zellsets
  wie test_bootstrap_by_cell() (gleicher RNG-Ablauf), fuer alle vier Metriken; gepaarte
  Differenz trainiert - roh wie paired_diff().

Macro-Klassenmenge (Nutzerentscheidung 2026-10-01, Jakob, "Option 2"): FESTE Labelmenge K =
alle Klassen im vollen Testsplit (bzw. Valsplit fuer val_*). Ein Macro-Mittel laeuft ueber die
Klassen aus K mit Support > 0 in der jeweiligen Auswertungsmenge:
  - Volles Testset: alle Klassen aus K -> identisch zu sklearn mit labels=K, zero_division=0.
  - Bootstrap-Resample: eine Klasse aus K, die im Resample in y_true fehlt (passiert, weil z. B.
    Klasse 32000 im Testsplit nur in 2 Zellen liegt -> ~16 % der Resamples), wird fuer dieses
    Resample NICHT gemittelt (statt mit 0 einzugehen oder implizit ueber die zufaellige
    Union y_true|y_pred zu laufen, wie es sklearn ohne labels= tut).
  - Vorhergesagte Klassen ausserhalb von K (im Train, nicht im Test) zaehlen nie als eigene
    Klasse - ihre Vorhersagen sind Fehler und senken Recall/F1 der wahren Klasse.
  - Precision einer Klasse, die nie vorhergesagt wird: 0 (wie zero_division=0, Anzahl solcher
    Klassen wird mitgeschrieben).
Konsequenz: test_macro_* kann von den gespeicherten sklearn-Werten (ohne labels=) abweichen,
falls die Probe Klassen ausserhalb von K vorhersagt - beide Werte werden ausgegeben.

Metriken werden aus der Konfusionsmatrix (np.bincount) berechnet statt mit vier sklearn-Aufrufen
je Resample (Laufzeit). Testvorhersagen werden persistiert (landuse_probe_preds_*_ckpt.npz);
--reuse-preds ueberspringt den Refit, wenn sie existieren.
"""
import argparse
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import run_landuse_probe as rlp
from eval_landuse import linear_probe

METRICS = ("oa", "macro_f1", "macro_precision", "macro_recall")


def encode(y: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> tuple:
    """Kodiert y/pred als Indizes; labels (= K) belegen 0..len(K)-1, Vorhersagen ausserhalb von K
    bekommen weitere Indizes (zaehlen nur als Fehler, nie als eigene Macro-Klasse)."""
    extra = np.setdiff1d(np.unique(pred), labels)
    all_labels = np.concatenate([labels, extra])
    lut = {c: i for i, c in enumerate(all_labels)}
    yi = np.fromiter((lut[c] for c in y), dtype=np.int64, count=len(y))
    pi = np.fromiter((lut[c] for c in pred), dtype=np.int64, count=len(pred))
    return yi, pi, len(labels), len(all_labels)


def scores_from_codes(yi: np.ndarray, pi: np.ndarray, n_k: int, n_all: int) -> dict:
    cm = np.bincount(yi * n_all + pi, minlength=n_all * n_all).reshape(n_all, n_all)
    tp = np.diag(cm)[:n_k].astype(np.float64)
    support = cm.sum(axis=1)[:n_k]
    n_pred = cm.sum(axis=0)[:n_k]
    active = support > 0
    tp, support, n_pred = tp[active], support[active], n_pred[active]
    precision = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    recall = tp / support
    f1 = 2 * tp / (support + n_pred)  # = 2TP/(2TP+FP+FN), support>0 -> nie 0/0
    return {"oa": np.trace(cm) / len(yi), "macro_f1": f1.mean(),
            "macro_precision": precision.mean(), "macro_recall": recall.mean()}


def scores(y, p, labels) -> dict:
    return scores_from_codes(*encode(y, p, labels))


def self_test(y: np.ndarray, p: np.ndarray, labels: np.ndarray) -> None:
    """Volles Set (alle Klassen aus K vorhanden): muss sklearn mit labels=K, zero_division=0 treffen."""
    ours = scores(y, p, labels)
    ref = {"oa": accuracy_score(y, p),
           "macro_f1": f1_score(y, p, labels=labels, average="macro", zero_division=0),
           "macro_precision": precision_score(y, p, labels=labels, average="macro", zero_division=0),
           "macro_recall": recall_score(y, p, labels=labels, average="macro", zero_division=0)}
    for m in METRICS:
        assert np.isclose(ours[m], ref[m], rtol=0, atol=1e-12), f"Self-Test {m}: {ours[m]} vs. sklearn {ref[m]}"


def cell_bootstrap(fits: dict, cell_ids: np.ndarray, labels: np.ndarray, n_resamples: int = rlp.N_TEST_BOOTSTRAP,
                   seed0: int = rlp.TEST_BOOTSTRAP_SEED0) -> dict:
    """Wie rlp.test_bootstrap_by_cell() (identische Zellsets: ein RNG, einmal pro Backbone gezogen,
    fuer alle Arme wiederverwendet), nur mit allen vier Metriken und fester Labelmenge K."""
    unique_cells = np.unique(cell_ids)
    cell_to_idx = {c: np.where(cell_ids == c)[0] for c in unique_cells}
    rng = np.random.default_rng(seed0)
    cellsets = [rng.choice(unique_cells, size=len(unique_cells), replace=True) for _ in range(n_resamples)]
    idx_sets = [np.concatenate([cell_to_idx[c] for c in cells]) for cells in cellsets]

    n_missing = np.array([len(labels) - len(np.unique(next(iter(fits.values()))["y_test"][idx])) for idx in idx_sets])
    print(f"  Resamples mit >=1 fehlenden Testklasse(n): {(n_missing > 0).sum()} von {n_resamples} "
          f"(max. {n_missing.max()} fehlend) - dort Macro ueber die vorhandenen Klassen")

    samples = {}
    for name, fit in fits.items():
        assert len(cell_ids) == len(fit["y_test"])
        yi, pi, n_k, n_all = encode(fit["y_test"], fit["pred_test"], labels)
        out = {m: np.empty(n_resamples) for m in METRICS}
        for i, idx in enumerate(idx_sets):
            for m, v in scores_from_codes(yi[idx], pi[idx], n_k, n_all).items():
                out[m][i] = v
        samples[name] = out
        print(f"  {name}: Zell-Bootstrap {n_resamples}x fertig")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=("tessera", "ae"), required=True)
    ap.add_argument("--reuse-preds", action="store_true",
                    help="Persistierte Vorhersagen laden statt Refit (falls vorhanden)")
    args = ap.parse_args()
    drv = importlib.import_module(f"run_landuse_probe_{args.backbone}_v1dedup")
    data_name = drv.DATA_NAME
    out_dir = drv.OUT_DIR
    preds_path = out_dir / f"landuse_probe_preds_{data_name}_ckpt.npz"

    stored = pd.read_csv(out_dir / f"landuse_probe_results_{data_name}_ckpt.csv").set_index("config")

    if args.reuse_preds and preds_path.exists():
        print(f"=== Vorhersagen laden: {preds_path} ===")
        z = np.load(preds_path, allow_pickle=True)
        names = list(z["names"])
        preds = {n: {k: z[f"{n}__{k}"] for k in ("y_val", "pred_val", "y_test", "pred_test")} for n in names}
        dims = {n: int(z[f"{n}__dim"]) for n in names}
    else:
        print("=== Konfigurationen aufbauen ===")
        configs = drv.build_configs_dedup()
        preds, dims = {}, {}
        for name, (X, y, split) in configs.items():
            C = float(stored.loc[name, "C"])
            tr, va, te = split == "train", split == "val", split == "test"
            res = linear_probe(X[tr], y[tr], X[va], y[va], X[te], y[te], C=C,
                               max_iter=drv.MAX_ITER, return_fit=True)
            for col in ("test_oa", "test_macro_f1", "val_oa", "val_macro_f1"):
                assert np.isclose(res[col], stored.loc[name, col], rtol=0, atol=1e-9), (
                    f"{name}: {col} reproduziert nicht ({res[col]} vs. gespeichert {stored.loc[name, col]})")
            print(f"  {name} (C={C}): Reproduktion OK")
            preds[name] = {"y_val": y[va], "pred_val": res["pred_val"], "y_test": y[te], "pred_test": res["pred_test"]}
            dims[name] = X.shape[1]
        np.savez(preds_path, names=np.array(list(preds)),
                 **{f"{n}__{k}": v for n, d in preds.items() for k, v in d.items()},
                 **{f"{n}__dim": d for n, d in dims.items()})
        print(f"Vorhersagen gespeichert: {preds_path}")

    rows = []
    for name, p in preds.items():
        labels_val, labels_test = np.unique(p["y_val"]), np.unique(p["y_test"])
        self_test(p["y_test"], p["pred_test"], labels_test)
        row = {"config": name, "dim": dims[name], "C": float(stored.loc[name, "C"]),
               "n_classes_test": len(labels_test)}
        for part, yy, pp, lab in (("val", p["y_val"], p["pred_val"], labels_val),
                                  ("test", p["y_test"], p["pred_test"], labels_test)):
            row.update({f"{part}_{m}": v for m, v in scores(yy, pp, lab).items()})
        row["test_macro_f1_sklearn_ohne_labels"] = float(stored.loc[name, "test_macro_f1"])
        row["n_classes_never_predicted_test"] = len(np.setdiff1d(labels_test, np.unique(p["pred_test"])))
        row["n_pred_classes_outside_test"] = len(np.setdiff1d(np.unique(p["pred_test"]), labels_test))
        rows.append(row)
        print(f"  {name}: " + " ".join(f"test_{m}={row[f'test_{m}']:.4f}" for m in METRICS)
              + f" | sklearn-ohne-labels macro_f1={row['test_macro_f1_sklearn_ohne_labels']:.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / f"landuse_probe_prec_rec_{data_name}_ckpt.csv", index=False)
    print(result.to_string(index=False))

    print("\n=== Zellebenen-Bootstrap (Test, 1000x) ===")
    cell_ids = rlp.load_test_pixel_cell_ids(data_name)
    labels_test = np.unique(next(iter(preds.values()))["y_test"])
    assert all(np.array_equal(np.unique(p["y_test"]), labels_test) for p in preds.values())
    fits = {n: {"y_test": p["y_test"], "pred_test": p["pred_test"]} for n, p in preds.items()}
    samples = cell_bootstrap(fits, cell_ids, labels_test)
    boot_rows = []
    for name, s in samples.items():
        for m in METRICS:
            lo, hi = np.percentile(s[m], [2.5, 97.5])
            boot_rows.append({"config": name, "metric": m, "mean": s[m].mean(), "ci_low": lo, "ci_high": hi})
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out_dir / f"landuse_probe_prec_rec_cell_bootstrap_{data_name}_ckpt.csv", index=False)
    print(boot.to_string(index=False))

    diff_rows = []
    trained, roh = samples[f"{data_name}_trainiert_projiziert"], samples[f"{data_name}_roh"]
    for m in METRICS:
        d = trained[m] - roh[m]
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff_rows.append({"backbone": data_name, "metric": m, "mean_diff_trainiert_minus_roh": d.mean(),
                          "ci_low_95": lo, "ci_high_95": hi, "significant": bool(not (lo <= 0 <= hi))})
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(out_dir / f"landuse_probe_prec_rec_paired_diff_cell_{data_name}_ckpt.csv", index=False)
    print("\n=== Gepaarte Differenz trainiert - roh ===")
    print(diff.to_string(index=False))


if __name__ == "__main__":
    main()
