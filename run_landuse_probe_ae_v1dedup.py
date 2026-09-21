"""
run_landuse_probe_ae_v1dedup.py

Schritt 3 AlphaEarth (2026-09-21, Jakob): Land-Use-Probe mit checkpoints/ae_v1_dedup/best.pt auf dem
deduplizierten landuse_pairs_ae_dedup.parquet - drei Arme (roh / zufaellig projiziert /
trainiert projiziert), identische Methodik (C-Grid-Suche via eval_landuse.linear_probe(),
identische Konstanten EE_DIM/LATENT_DIM/HIDDEN_DIM/DROPOUT/RANDOM_HEAD_SEED/EXCLUDED_CLASS) wie
run_landuse_probe.py - dessen namensbasierte Dict-Lookups (CHECKPOINTS/EE_DIMS) aber NICHT
generisch fuer den neuen Checkpoint-/Datei-Namen 'ae_dedup' sind, deshalb hier mit
expliziten Pfaden statt Monkeypatch nachgebaut (transparenter als mehrere globale Dicts zu
patchen).

Bootstrap: NUR Zellebene (test_bootstrap_by_cell(), bereits in run_landuse_probe.py vorhanden -
der Fix fuer den bekannten Pixel-/Zeilen-Bootstrap-Bug, siehe MANIFEST.md fuer den separaten
Fix-Bericht). Kein Zeilen-Bootstrap mehr fuer diesen neuen Lauf.
"""
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
import run_landuse_probe as rlp
from run_landuse_cell_bootstrap import check_stability
from eval_landuse import linear_probe
from model import ProjectionHead
from normalize import ZSCORE, apply_zscore, load_stats

DATA_NAME = "ae_dedup"
CKPT_PATH = cfg.ROOT / "checkpoints" / "ae_v1_dedup" / "best.pt"
YAML_CFG_PATH = cfg.ROOT / "configs" / "ae_v1_dedup.yaml"
EE_DIM = 64
LATENT_DIM = rlp.LATENT_DIM
HIDDEN_DIM = rlp.HIDDEN_DIM
DROPOUT = rlp.DROPOUT
RANDOM_HEAD_SEED = rlp.RANDOM_HEAD_SEED
C_GRID = (0.1, 1.0, 10.0)
MAX_ITER = 1000
OUT_DIR = cfg.DATA / "processed"


def load_trained_head_dedup():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["cfg"]
    expected = {"ee_dim": EE_DIM, "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT}
    for k, want in expected.items():
        got = ckpt_cfg.get(k)
        if got != want:
            raise ValueError(f"Checkpoint-Config {k}={got!r} weicht von erwarteter Konstante {want!r} ab.")
    head = ProjectionHead(EE_DIM, LATENT_DIM, HIDDEN_DIM, DROPOUT)
    state = {k.replace("ee_head.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("ee_head.")}
    head.load_state_dict(state)
    head.eval()
    print(f"  trainierter Kopf geladen ({CKPT_PATH}, Epoche {ckpt['epoch']})")
    return head, ckpt_cfg


def random_head_dedup() -> ProjectionHead:
    torch.manual_seed(RANDOM_HEAD_SEED)
    head = ProjectionHead(EE_DIM, LATENT_DIM, HIDDEN_DIM, DROPOUT)
    head.eval()
    return head


def build_configs_dedup() -> dict:
    df = rlp.load_clean(DATA_NAME)
    emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    y = df["code_2021"].to_numpy()
    split = df["split"].to_numpy()

    with open(YAML_CFG_PATH) as f:
        yaml_cfg = yaml.safe_load(f)
    yaml_mode, yaml_stats_path = rlp.resolve_normalization(yaml_cfg, str(YAML_CFG_PATH))
    print(f"  Normalisierung (roh/zufaellig, aus {YAML_CFG_PATH}): {yaml_mode}"
          + (f", Statistik={yaml_stats_path}" if yaml_stats_path else ""))

    trained_head, ckpt_cfg = load_trained_head_dedup()
    ckpt_mode, ckpt_stats_path = rlp.resolve_normalization(ckpt_cfg, "Checkpoint-Config (ae_v1_dedup)")
    print(f"  Normalisierung (trainiert, aus Checkpoint-Config): {ckpt_mode}"
          + (f", Statistik={ckpt_stats_path}" if ckpt_stats_path else ""))
    if ckpt_mode != yaml_mode or ckpt_stats_path != yaml_stats_path:
        raise ValueError(f"Normalisierung weicht zwischen Config ({yaml_mode!r},{yaml_stats_path!r}) "
                          f"und Checkpoint ({ckpt_mode!r},{ckpt_stats_path!r}) ab.")

    if yaml_mode == ZSCORE:
        stats = load_stats(yaml_stats_path)
        emb_raw = apply_zscore(emb, stats)
    else:
        emb_raw = emb

    rand_head = random_head_dedup()

    configs = {
        f"{DATA_NAME}_roh": (emb_raw, y, split),
        f"{DATA_NAME}_zufaellig_projiziert": (rlp.project(rand_head, emb_raw), y, split),
        f"{DATA_NAME}_trainiert_projiziert": (rlp.project(trained_head, emb_raw), y, split),
    }
    return configs


def run_probe_dedup(configs: dict) -> tuple:
    rows = []
    fits = {}
    for name, (X, y, split) in configs.items():
        train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        print(f"\n=== {name} === train={len(X_train)} val={len(X_val)} test={len(X_test)} dim={X.shape[1]}")

        best = None
        for C in C_GRID:
            t0 = time.time()
            res = linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, C=C,
                                max_iter=MAX_ITER, return_fit=True)
            dt = time.time() - t0
            print(f"  C={C}: val_macro_f1={res['val_macro_f1']:.4f} val_oa={res['val_oa']:.4f} "
                  f"test_macro_f1={res['test_macro_f1']:.4f} test_oa={res['test_oa']:.4f} ({dt:.1f}s)")
            if best is None or res["val_macro_f1"] > best["val_macro_f1"]:
                best = res
        rows.append({"config": name, "dim": X.shape[1],
                      **{k: v for k, v in best.items() if k not in ("clf", "pred_val", "pred_test")}})
        fits[name] = {"C": best["C"], "pred_val": best["pred_val"], "pred_test": best["pred_test"],
                      "y_test": y_test, "y_val": y_val}
    return pd.DataFrame(rows), fits


def main():
    print("=== Konfigurationen aufbauen (roh/zufaellig/trainiert) ===")
    configs = build_configs_dedup()

    print("\n=== Punktschaetzung (C-Grid-Suche) ===")
    result, fits = run_probe_dedup(configs)
    out_path = OUT_DIR / f"landuse_probe_results_{DATA_NAME}_ckpt.csv"
    result.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")
    print(result.to_string(index=False))

    print("\n=== Zellzuordnung Testpixel laden ===")
    cell_ids = rlp.load_test_pixel_cell_ids(DATA_NAME)
    n_unique = len(np.unique(cell_ids))
    print(f"  {len(cell_ids)} Testpixel, {n_unique} eindeutige Zellen")

    print("\n=== Zellebenen-Bootstrap (Test, 1000x) ===")
    # test_bootstrap_by_cell() extrahiert den Backbone-Namen intern per name.split("_",1)[0]
    # aus den Fit-Namen (z.B. "ae_dedup_roh" -> "ae") - der Dict-Key hier muss dazu
    # passen ("ae"), NICHT DATA_NAME ("ae_dedup", enthaelt selbst einen Unterstrich).
    cell_samples = rlp.test_bootstrap_by_cell(fits, {"ae": cell_ids})
    rows = []
    for name, s in cell_samples.items():
        for metric in ("oa", "macro_f1"):
            lo, hi = np.percentile(s[metric], [2.5, 97.5])
            rows.append({"config": name, "metric": metric, "mean": s[metric].mean(),
                         "ci_low": lo, "ci_high": hi})
    cell_summary = pd.DataFrame(rows)
    cell_summary.to_csv(OUT_DIR / f"landuse_probe_cell_bootstrap_{DATA_NAME}_ckpt.csv", index=False)
    print(cell_summary.to_string(index=False))

    print("\n=== Gepaarter Zellebenen-Bootstrap (trainiert - roh) ===")
    cell_diff = rlp.paired_diff(cell_samples, backbones=(DATA_NAME,))
    cell_diff.to_csv(OUT_DIR / f"landuse_probe_paired_diff_cell_{DATA_NAME}_ckpt.csv", index=False)
    print(cell_diff.to_string(index=False))

    print("\n=== Nebendiagnose: Stabilitaet des Zellebenen-Bootstraps (zweiter Startseed) ===")
    print("    (Hauptzahlen oben bleiben 1000x/Seed 2000 - KEINE Eskalation durch dieses Skript)")
    max_rd, stab = check_stability(fits, {"ae": cell_ids}, n_resamples=rlp.N_TEST_BOOTSTRAP)
    stab.to_csv(OUT_DIR / f"landuse_probe_cell_bootstrap_stability_{DATA_NAME}_ckpt.csv", index=False)
    print(stab.to_string(index=False, float_format=lambda x: f"{x:.5f}" if isinstance(x, float) else x))
    print(f"\nMaximale relative Abweichung der CI-Grenzen (Seed {rlp.TEST_BOOTSTRAP_SEED0} vs. 9000): {max_rd:.4%}")
    if max_rd > 0.01:
        print("STABILITAET_UEBER_1_PROZENT: Abweichung > 1 % - NICHT selbst eskalieren, stoppen und berichten.")
    else:
        print("STABILITAET_OK: Abweichung <= 1 %.")


if __name__ == "__main__":
    main()
