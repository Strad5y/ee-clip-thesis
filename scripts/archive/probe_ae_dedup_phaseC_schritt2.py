"""
Phase C, Schritt 2: Voller AlphaEarth-Probe-Lauf auf der deduplizierten Datenbasis
(data/processed/landuse_pairs_ae.parquet, 4.363.718 Zeilen), max_iter=1000 statt 200.
Alles andere identisch zu run_landuse_probe.py: gleiches C-Grid {0.1,1.0,10.0}, gleiche
Arm-Definitionen (roh/zufaellig/trainiert projiziert), gleicher Ausschluss (12400, missing).
Dateien mit Suffix _ae_dedup - die bestehenden _ae-Dateien vom 04.09. werden NICHT beruehrt.
"""
import sys
import time
import warnings
sys.path.insert(0, "/home/minorai/PycharmProjects/ee-clip-thesis")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
import run_landuse_probe as rlp
from eval_landuse import linear_probe

MAX_ITER = 1000
C_GRID = (0.1, 1.0, 10.0)
OUT_SUFFIX = "_ae_dedup"
TRAIN_BOOTSTRAP_BUDGET_S = 2 * 3600

print(f"=== Schritt 2: voller Lauf, max_iter={MAX_ITER}, Basis={OUT_SUFFIX} ===")
configs = rlp.build_configs(("ae",))

# --- run_probe-Aequivalent mit max_iter=1000, zusaetzlich n_iter_/ConvergenceWarning je (arm,C) ---
rows = []
fits = {}
fit_durations = []
per_c_records = []
for name, (X, y, split) in configs.items():
    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\n=== {name} ===  train={len(X_train)} val={len(X_val)} test={len(X_test)} dim={X.shape[1]}")
    best = None
    for C in C_GRID:
        t0 = time.time()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            res = linear_probe(X_train, y_train, X_val, y_val, X_test, y_test,
                                C=C, max_iter=MAX_ITER, return_fit=True)
            conv = [w for w in wlist if "onverg" in str(w.message)]
        dt = time.time() - t0
        fit_durations.append(dt)
        n_iter = int(np.atleast_1d(res["clf"].n_iter_)[0])
        per_c_records.append({"config": name, "C": C, "n_iter_": n_iter,
                               "convergence_warning": len(conv) > 0,
                               "val_macro_f1": res["val_macro_f1"], "val_oa": res["val_oa"],
                               "test_macro_f1": res["test_macro_f1"], "test_oa": res["test_oa"]})
        print(f"  C={C}: val_macro_f1={res['val_macro_f1']:.4f} val_oa={res['val_oa']:.4f} "
              f"test_macro_f1={res['test_macro_f1']:.4f} test_oa={res['test_oa']:.4f} "
              f"n_iter_={n_iter} ConvergenceWarning={len(conv) > 0} ({dt:.1f}s)")
        if best is None or res["val_macro_f1"] > best["val_macro_f1"]:
            best = res
            best["n_iter_"] = n_iter
            best["convergence_warning"] = len(conv) > 0
    rows.append({"config": name, "dim": X.shape[1],
                  **{k: v for k, v in best.items() if k not in ("clf", "pred_val", "pred_test")}})
    fits[name] = {
        "C": best["C"], "pred_val": best["pred_val"], "pred_test": best["pred_test"],
        "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "n_iter_": best["n_iter_"], "convergence_warning": best["convergence_warning"],
    }

result = pd.DataFrame(rows)
out_path = cfg.DATA / "processed" / f"landuse_probe_results{OUT_SUFFIX}.csv"
result.to_csv(out_path, index=False)
per_c_df = pd.DataFrame(per_c_records)
per_c_df.to_csv(cfg.DATA / "processed" / f"landuse_probe_c_grid{OUT_SUFFIX}.csv", index=False)
print(f"\nErgebnisse gespeichert: {out_path}")
print(result.to_string(index=False))

print("\n=== Gewaehltes C / n_iter_ / ConvergenceWarning je Arm (finaler Fit) ===")
for name, fit in fits.items():
    print(f"  {name}: C={fit['C']} n_iter_={fit['n_iter_']} "
          f"ConvergenceWarning={fit['convergence_warning']}")

# --- Haeufigkeitsniveau ---
print("\n=== Haeufigkeitsniveau ===")
majority_rows = []
majority_fits = {}
for bb in ("ae",):
    _, y_bb, split_bb = configs[f"{bb}_roh"]
    y_train_arr = y_bb[split_bb == "train"]
    y_val_arr = y_bb[split_bb == "val"]
    y_test_arr = y_bb[split_bb == "test"]
    maj = rlp.majority_baseline(y_train_arr, y_val_arr, y_test_arr)
    majority_fits[bb] = maj
    print(f"  {bb}: Mehrheitsklasse={maj['majority_class']}  "
          f"val_oa={maj['val_oa']:.4f} val_macro_f1={maj['val_macro_f1']:.4f}  "
          f"test_oa={maj['test_oa']:.4f} test_macro_f1={maj['test_macro_f1']:.4f}")
    majority_rows.append({"config": f"{bb}_haeufigkeitsniveau", "dim": 0, "C": None,
                           "val_oa": maj["val_oa"], "val_macro_f1": maj["val_macro_f1"],
                           "test_oa": maj["test_oa"], "test_macro_f1": maj["test_macro_f1"]})
majority_df = pd.DataFrame(majority_rows)
majority_out = cfg.DATA / "processed" / f"landuse_probe_majority_baseline{OUT_SUFFIX}.csv"
majority_df.to_csv(majority_out, index=False)
print(f"Gespeichert: {majority_out}")

# --- Per-Klassen-F1 ---
print("\n=== Per-Klassen-F1 (Test) ===")
per_class_frames = {}
for name, fit in fits.items():
    s = rlp.per_class_f1(fit["y_test"], fit["pred_test"])
    per_class_frames[name] = s
for bb, maj in majority_fits.items():
    y_test = configs[f"{bb}_roh"][1][configs[f"{bb}_roh"][2] == "test"]
    s = rlp.per_class_f1(y_test, maj["pred_test"])
    per_class_frames[f"{bb}_haeufigkeitsniveau"] = s
per_class_df = pd.DataFrame(per_class_frames)
per_class_out = cfg.DATA / "processed" / f"landuse_probe_per_class_f1{OUT_SUFFIX}.csv"
per_class_df.to_csv(per_class_out)
print(f"Gespeichert: {per_class_out}")

# --- Konfusionsmatrizen ---
print("\n=== Konfusionsmatrizen (Test) ===")
for name, fit in fits.items():
    cm = rlp.confusion(fit["y_test"], fit["pred_test"])
    cm_out = cfg.DATA / "processed" / f"landuse_probe_confusion_ae_dedup_{name.split('_', 1)[1]}.csv"
    cm.to_csv(cm_out)
    print(f"  {name}: gespeichert unter {cm_out} ({cm.shape[0]}x{cm.shape[1]})")

# --- Train-Bootstrap-Budgetschaetzung + ggf. Ausfuehrung ---
avg_fit_time = float(np.mean(fit_durations))
n_train_bootstrap_fits = rlp.N_TRAIN_BOOTSTRAP * len(fits)
projected_train_bootstrap_s = avg_fit_time * n_train_bootstrap_fits
print(f"\nTrainings-Bootstrap-Schaetzung: {n_train_bootstrap_fits} Refits x ~{avg_fit_time:.1f}s "
      f"= {projected_train_bootstrap_s / 60:.1f} min ({projected_train_bootstrap_s / 3600:.2f} h)")

if projected_train_bootstrap_s > TRAIN_BOOTSTRAP_BUDGET_S:
    print(f"Geschaetzte Trainings-Bootstrap-Laufzeit > {TRAIN_BOOTSTRAP_BUDGET_S/3600:.0f}h - "
          "UEBERSPRUNGEN (Nutzervorgabe: vorher melden statt laufen lassen). "
          "Punktschaetzung oben bleibt gueltig; Test-Bootstrap laeuft trotzdem.")
else:
    print(f"\n=== Trainings-Bootstrap ({rlp.N_TRAIN_BOOTSTRAP} Seeds je Config) ===")
    train_boot = rlp.train_bootstrap(fits, max_iter=MAX_ITER)
    train_boot.to_csv(cfg.DATA / "processed" / f"landuse_probe_train_bootstrap{OUT_SUFFIX}.csv", index=False)
    train_boot_summary = rlp.summarize_bootstrap(train_boot)
    train_boot_summary.to_csv(cfg.DATA / "processed" / f"landuse_probe_train_bootstrap_summary{OUT_SUFFIX}.csv", index=False)
    print(train_boot_summary.to_string(index=False))

# --- Test-Bootstrap ---
print(f"\n=== Test-Bootstrap ({rlp.N_TEST_BOOTSTRAP}x je Config) ===")
test_samples = rlp.test_bootstrap(fits)
test_boot_summary = pd.DataFrame([
    {"config": name, "test_oa_mean": s["oa"].mean(), "test_oa_std": s["oa"].std(),
     "test_macro_f1_mean": s["macro_f1"].mean(), "test_macro_f1_std": s["macro_f1"].std()}
    for name, s in test_samples.items()
])
test_boot_summary.to_csv(cfg.DATA / "processed" / f"landuse_probe_test_bootstrap_summary{OUT_SUFFIX}.csv", index=False)
print(test_boot_summary.to_string(index=False))

print("\n=== Gepaarter Test-Bootstrap: trainiert - roh (95%-Intervall) ===")
diff_result = rlp.paired_diff(test_samples, backbones=("ae",))
diff_result.to_csv(cfg.DATA / "processed" / f"landuse_probe_paired_diff{OUT_SUFFIX}.csv", index=False)
print(diff_result.to_string(index=False))

print("\n=== FERTIG ===")

# fits/configs fuer Schritt 3 im Prozessspeicher nicht verfuegbar (neuer Prozess) - Schritt 3
# laedt daher die Konfiguration erneut ueber build_configs(), das ist deterministisch (keine
# neue Zufallsquelle im Fit selbst - nur C-Grid-Punktschaetzung, dieselbe wie hier).
