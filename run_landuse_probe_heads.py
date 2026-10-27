"""
run_landuse_probe_heads.py

Nutzervorgabe (2026-09-28, Jakob, via Betreuer): Land-Use-Probe zusaetzlich zu LogisticRegression
(run_landuse_probe_{tessera,ae}_v1dedup.py, siehe landuse_probe_results_*_dedup_ckpt.csv) mit
CatBoost, einem 2-Layer-MLP und TabPFN auswerten - Frage, ob das schwache Delta
roh<->trainiert projiziert ein Artefakt der Linearitaet des LogReg-Probes ist. Generischer
Nachfolger zu den zwei fast identischen *_v1dedup-Skripten (--backbone statt Kopie), operiert
NUR auf den deduplizierten Checkpoints/Daten (tessera_v1_dedup/ae_v1_dedup), identische Methodik
(C-Grid-Suche fuer LogReg, raeumlich geblockter train/val/test-Split, EXCLUDED_CLASS/RANDOM_HEAD_
SEED wie run_landuse_probe.py) - siehe dortigen Docstring fuer die Herkunft dieser Entscheidungen.

Sample-Regime pro Kopf:
  - LogReg: 'full' wird NICHT neu gerechnet, sondern aus landuse_probe_results_{data_name}_ckpt.csv
    uebernommen (bereits vorhandene, aktuelle Zahlen - kein Grund, das nochmal zu fitten).
    Zusaetzlich 'subsample' (identisches Subsample wie CatBoost/MLP/TabPFN) als fairer Vergleich.
  - CatBoost, MLP: 'subsample' UND 'full' (koennen beide Groessenordnungen).
  - TabPFN: NUR 'subsample' (harte Grenze der Library, siehe eval_landuse_heads.py-Docstring).

Bootstrap: Zellebenen-Test-Bootstrap (rlp.test_bootstrap_by_cell(), 1000x, KEIN Refit - billig,
siehe run_landuse_probe.py-Docstring) + gepaarter Diff (trainiert - roh) fuer JEDE Kopf/Regime-
Kombination mit return_fit=True. 'zufaellig_projiziert' laeuft als dritter Arm nur in der
Punktschaetzung mit (Kontrollarm, nicht Gegenstand des gepaarten Vergleichs - identisch zur
Methodik in run_landuse_probe.py::paired_diff()).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import yaml

import config as cfg
import run_landuse_probe as rlp
from eval_landuse import linear_probe
from eval_landuse_heads import (
    SUBSAMPLE_SIZE, TabPFNUnavailable, catboost_probe, mlp_probe, stratified_subsample, tabpfn_probe,
)
from model import ProjectionHead
from normalize import ZSCORE, apply_zscore, load_stats

BACKBONE_CFG = {
    "tessera": dict(data_name="tessera_dedup", cell_prefix="tessera", ee_dim=128,
                     ckpt_path=cfg.ROOT / "checkpoints" / "tessera_v1_dedup" / "best.pt",
                     yaml_cfg_path=cfg.ROOT / "configs" / "tessera_v1_dedup.yaml"),
    "ae": dict(data_name="ae_dedup", cell_prefix="ae", ee_dim=64,
               ckpt_path=cfg.ROOT / "checkpoints" / "ae_v1_dedup" / "best.pt",
               yaml_cfg_path=cfg.ROOT / "configs" / "ae_v1_dedup.yaml"),
}
LATENT_DIM = rlp.LATENT_DIM
HIDDEN_DIM = rlp.HIDDEN_DIM
DROPOUT = rlp.DROPOUT
RANDOM_HEAD_SEED = rlp.RANDOM_HEAD_SEED
LOGREG_C_GRID = (0.1, 1.0, 10.0)
OUT_DIR = cfg.DATA / "processed"


def load_trained_head(ckpt_path, ee_dim: int) -> tuple[ProjectionHead, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["cfg"]
    expected = {"ee_dim": ee_dim, "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT}
    for k, want in expected.items():
        got = ckpt_cfg.get(k)
        if got != want:
            raise ValueError(f"Checkpoint-Config {k}={got!r} weicht von erwarteter Konstante {want!r} ab.")
    head = ProjectionHead(ee_dim, LATENT_DIM, HIDDEN_DIM, DROPOUT)
    state = {k.replace("ee_head.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("ee_head.")}
    head.load_state_dict(state)
    head.eval()
    print(f"  trainierter Kopf geladen ({ckpt_path}, Epoche {ckpt['epoch']})")
    return head, ckpt_cfg


def random_head(ee_dim: int) -> ProjectionHead:
    torch.manual_seed(RANDOM_HEAD_SEED)
    head = ProjectionHead(ee_dim, LATENT_DIM, HIDDEN_DIM, DROPOUT)
    head.eval()
    return head


def build_configs(backbone: str) -> tuple[dict, dict]:
    bc = BACKBONE_CFG[backbone]
    df = rlp.load_clean(bc["data_name"])
    emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    y = df["code_2021"].to_numpy()
    split = df["split"].to_numpy()

    with open(bc["yaml_cfg_path"]) as f:
        yaml_cfg = yaml.safe_load(f)
    yaml_mode, yaml_stats_path = rlp.resolve_normalization(yaml_cfg, str(bc["yaml_cfg_path"]))
    print(f"  Normalisierung (roh/zufaellig, aus {bc['yaml_cfg_path']}): {yaml_mode}"
          + (f", Statistik={yaml_stats_path}" if yaml_stats_path else ""))

    trained_head, ckpt_cfg = load_trained_head(bc["ckpt_path"], bc["ee_dim"])
    ckpt_mode, ckpt_stats_path = rlp.resolve_normalization(ckpt_cfg, f"Checkpoint-Config ({bc['data_name']})")
    print(f"  Normalisierung (trainiert, aus Checkpoint-Config): {ckpt_mode}"
          + (f", Statistik={ckpt_stats_path}" if ckpt_stats_path else ""))
    if ckpt_mode != yaml_mode or ckpt_stats_path != yaml_stats_path:
        raise ValueError(f"Normalisierung weicht zwischen Config ({yaml_mode!r},{yaml_stats_path!r}) "
                          f"und Checkpoint ({ckpt_mode!r},{ckpt_stats_path!r}) ab.")

    emb_raw = apply_zscore(emb, load_stats(yaml_stats_path)) if yaml_mode == ZSCORE else emb
    rand_head = random_head(bc["ee_dim"])

    configs = {
        "roh": (emb_raw, y, split),
        "zufaellig_projiziert": (rlp.project(rand_head, emb_raw), y, split),
        "trainiert_projiziert": (rlp.project(trained_head, emb_raw), y, split),
    }
    return configs, bc


def load_cached_full_logreg(data_name: str) -> pd.DataFrame:
    """LogReg-'full'-Zahlen aus dem bereits vorhandenen, aktuellen Lauf uebernehmen
    (landuse_probe_results_{data_name}_ckpt.csv) statt neu zu fitten - siehe Docstring oben."""
    path = OUT_DIR / f"landuse_probe_results_{data_name}_ckpt.csv"
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        arm = r["config"].replace(f"{data_name}_", "")
        rows.append({"arm": arm, "head": "logreg", "regime": "full", "n_train": None,
                      "val_oa": r["val_oa"], "val_macro_f1": r["val_macro_f1"],
                      "test_oa": r["test_oa"], "test_macro_f1": r["test_macro_f1"],
                      "extra": f"C={r['C']}", "source": str(path)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True, choices=list(BACKBONE_CFG))
    ap.add_argument("--skip-full", action="store_true", help="CatBoost/MLP nur auf Subsample, nicht auf vollem Train")
    ap.add_argument("--skip-tabpfn", action="store_true", help="TabPFN ueberspringen (z.B. kein TABPFN_TOKEN gesetzt)")
    ap.add_argument("--catboost-iterations-sub", type=int, default=1000)
    ap.add_argument("--catboost-iterations-full", type=int, default=500)
    ap.add_argument("--mlp-epochs-sub", type=int, default=100)
    ap.add_argument("--mlp-epochs-full", type=int, default=30)
    ap.add_argument("--n-bootstrap", type=int, default=rlp.N_TEST_BOOTSTRAP,
                     help="Zellebenen-Test-Bootstrap-Resamples (Default wie run_landuse_probe.py, "
                          "1000x). Kostendominant ist die Anzahl Kopf/Regime-Fits, nicht diese Zahl - "
                          "fuer schnelle Exploration trotzdem reduzierbar.")
    args = ap.parse_args()

    bc_name = args.backbone
    print(f"=== Konfigurationen aufbauen ({bc_name}) ===")
    configs, bc = build_configs(bc_name)
    data_name = bc["data_name"]

    result_rows = []
    fits = {}  # key: f"{head}_{regime}_{arm}" -> {"pred_val","pred_test","y_val","y_test"}

    for arm, (X, y, split) in configs.items():
        train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        print(f"\n### Arm '{arm}' (dim={X.shape[1]}): train={len(X_train)} val={len(X_val)} test={len(X_test)}")

        # y_train ist ueber alle drei Arme identisch (nur X unterscheidet sich) -> mit fixem
        # Seed liefert stratified_subsample() hier fuer jeden Arm dieselben Zeilenindizes.
        sub_idx = stratified_subsample(y_train)
        n_sub = len(sub_idx)
        print(f"  Subsample: {n_sub} Zeilen (Ziel {SUBSAMPLE_SIZE}, {len(np.unique(y_train[sub_idx]))} Klassen)")
        X_train_sub, y_train_sub = X_train[sub_idx], y_train[sub_idx]

        # --- LogReg auf Subsample (fairer Referenzpunkt fuer CatBoost/MLP/TabPFN-Subsample) ---
        best = None
        for C in LOGREG_C_GRID:
            res = linear_probe(X_train_sub, y_train_sub, X_val, y_val, X_test, y_test, C=C,
                                max_iter=1000, return_fit=True)
            if best is None or res["val_macro_f1"] > best["val_macro_f1"]:
                best = res
        print(f"  logreg/subsample: test_macro_f1={best['test_macro_f1']:.4f} test_oa={best['test_oa']:.4f}")
        result_rows.append({"arm": arm, "head": "logreg", "regime": "subsample", "n_train": n_sub,
                             "val_oa": best["val_oa"], "val_macro_f1": best["val_macro_f1"],
                             "test_oa": best["test_oa"], "test_macro_f1": best["test_macro_f1"],
                             "extra": f"C={best['C']}", "source": None})
        fits[f"logreg_subsample_{arm}"] = {"pred_val": best["pred_val"], "pred_test": best["pred_test"],
                                            "y_val": y_val, "y_test": y_test}

        # --- CatBoost ---
        for regime, (Xtr, ytr) in (("subsample", (X_train_sub, y_train_sub)),
                                    ("full", (X_train, y_train))):
            if regime == "full" and args.skip_full:
                continue
            iters = args.catboost_iterations_sub if regime == "subsample" else args.catboost_iterations_full
            t0 = time.time()
            res = catboost_probe(Xtr, ytr, X_val, y_val, X_test, y_test, iterations=iters, return_fit=True)
            dt = time.time() - t0
            print(f"  catboost/{regime} (n={len(Xtr)}, iters={iters}, best_iter={res['best_iteration']}): "
                  f"test_macro_f1={res['test_macro_f1']:.4f} test_oa={res['test_oa']:.4f} ({dt:.1f}s)")
            result_rows.append({"arm": arm, "head": "catboost", "regime": regime, "n_train": len(Xtr),
                                 "val_oa": res["val_oa"], "val_macro_f1": res["val_macro_f1"],
                                 "test_oa": res["test_oa"], "test_macro_f1": res["test_macro_f1"],
                                 "extra": f"best_iteration={res['best_iteration']}", "source": None})
            fits[f"catboost_{regime}_{arm}"] = {"pred_val": res["pred_val"], "pred_test": res["pred_test"],
                                                 "y_val": y_val, "y_test": y_test}

        # --- MLP ---
        for regime, (Xtr, ytr, epochs, bs) in (
            ("subsample", (X_train_sub, y_train_sub, args.mlp_epochs_sub, 512)),
            ("full", (X_train, y_train, args.mlp_epochs_full, 8192)),
        ):
            if regime == "full" and args.skip_full:
                continue
            t0 = time.time()
            res = mlp_probe(Xtr, ytr, X_val, y_val, X_test, y_test, max_epochs=epochs, batch_size=bs,
                             return_fit=True)
            dt = time.time() - t0
            print(f"  mlp/{regime} (n={len(Xtr)}, epochs_run={res['epochs_run']}): "
                  f"test_macro_f1={res['test_macro_f1']:.4f} test_oa={res['test_oa']:.4f} ({dt:.1f}s)")
            result_rows.append({"arm": arm, "head": "mlp", "regime": regime, "n_train": len(Xtr),
                                 "val_oa": res["val_oa"], "val_macro_f1": res["val_macro_f1"],
                                 "test_oa": res["test_oa"], "test_macro_f1": res["test_macro_f1"],
                                 "extra": f"epochs_run={res['epochs_run']}", "source": None})
            fits[f"mlp_{regime}_{arm}"] = {"pred_val": res["pred_val"], "pred_test": res["pred_test"],
                                            "y_val": y_val, "y_test": y_test}

        # --- TabPFN (nur Subsample) ---
        if not args.skip_tabpfn:
            t0 = time.time()
            try:
                res = tabpfn_probe(X_train_sub, y_train_sub, X_val, y_val, X_test, y_test, return_fit=True)
                dt = time.time() - t0
                print(f"  tabpfn/subsample (n={n_sub}): test_macro_f1={res['test_macro_f1']:.4f} "
                      f"test_oa={res['test_oa']:.4f} ({dt:.1f}s)")
                result_rows.append({"arm": arm, "head": "tabpfn", "regime": "subsample", "n_train": n_sub,
                                     "val_oa": res["val_oa"], "val_macro_f1": res["val_macro_f1"],
                                     "test_oa": res["test_oa"], "test_macro_f1": res["test_macro_f1"],
                                     "extra": "ignore_pretraining_limits=True (>10 Klassen)", "source": None})
                fits[f"tabpfn_subsample_{arm}"] = {"pred_val": res["pred_val"], "pred_test": res["pred_test"],
                                                    "y_val": y_val, "y_test": y_test}
            except TabPFNUnavailable as e:
                print(f"  tabpfn/subsample UEBERSPRUNGEN: {e}")

    result = pd.DataFrame(result_rows)
    cached_logreg_full = load_cached_full_logreg(data_name)
    result = pd.concat([result, cached_logreg_full], ignore_index=True)
    out_path = OUT_DIR / f"landuse_probe_heads_results_{data_name}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")
    print(result.sort_values(["head", "regime", "arm"]).to_string(index=False))

    # --- Zellebenen-Bootstrap + gepaarter Diff (trainiert - roh) je Kopf/Regime ---
    # Nur 'roh'/'trainiert_projiziert' gehen in den (teuren) Bootstrap - 'zufaellig_projiziert'
    # ist der Kontrollarm aus der Punktschaetzung oben, wird von paired_diff() ohnehin nicht
    # gebraucht (die vergleicht nur trainiert-roh) und wuerde die Laufzeit um ein Drittel
    # erhoehen, ohne eine zusaetzliche Frage zu beantworten.
    bootstrap_fits = {name: fit for name, fit in fits.items()
                       if name.endswith("_roh") or name.endswith("_trainiert_projiziert")}
    print(f"\n=== Zellzuordnung Testpixel laden ===")
    cell_ids = rlp.load_test_pixel_cell_ids(data_name)

    print(f"\n=== Zellebenen-Bootstrap (Test, {args.n_bootstrap}x) je Kopf/Regime "
          f"({len(bootstrap_fits)} Fits, 'zufaellig_projiziert' ausgeschlossen) ===")
    cell_samples = rlp.test_bootstrap_by_cell(
        bootstrap_fits, {"logreg": cell_ids, "catboost": cell_ids, "mlp": cell_ids, "tabpfn": cell_ids},
        n_resamples=args.n_bootstrap,
    )
    bootstrap_rows = []
    for name, s in cell_samples.items():
        for metric in ("oa", "macro_f1"):
            lo, hi = np.percentile(s[metric], [2.5, 97.5])
            bootstrap_rows.append({"config": name, "metric": metric, "mean": s[metric].mean(),
                                    "ci_low": lo, "ci_high": hi})
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_path = OUT_DIR / f"landuse_probe_heads_cell_bootstrap_{data_name}.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)
    print(f"Gespeichert: {bootstrap_path}")

    head_regimes = sorted({name.rsplit("_", 1)[0] for name in fits if name.endswith("_roh")})
    diff = rlp.paired_diff(cell_samples, backbones=tuple(head_regimes))
    diff_path = OUT_DIR / f"landuse_probe_heads_paired_diff_cell_{data_name}.csv"
    diff.to_csv(diff_path, index=False)
    print(f"Gespeichert: {diff_path}")
    print(diff.to_string(index=False))


if __name__ == "__main__":
    main()
