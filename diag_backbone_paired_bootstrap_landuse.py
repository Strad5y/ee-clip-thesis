"""
diag_backbone_paired_bootstrap_landuse.py  (2026-09-22, Jakob) - NUR DIAGNOSE

Gepaarter Zellebenen-Bootstrap fuer die BACKBONE-DIFFERENZ (TESSERA-dedup minus AE-dedup),
Land-Use-Probe, "roh" und "trainiert projiziert" (macro_f1, oa). Beide Backbones haben exakt
dieselben 167 Test-Zellen (verifiziert unten) - in jedem Resample wird DIESELBE gezogene
Zellmenge auf beide Backbones angewendet, damit die Differenz PAARWEISE ist (nicht zwei
unabhaengige CIs verglichen).

Refit bei dem bereits gewaehlten besten C je Backbone/Konfiguration (aus
landuse_probe_results_{tessera,ae}_dedup_ckpt.csv: TESSERA C=10.0, AE C=1.0 fuer roh/trainiert),
identische Methodik wie run_landuse_probe.py::run_probe()/build_configs() (unveraendert
importiert). Kein Refit von "zufaellig projiziert" (nicht Teil der angeforderten Backbone-
Kernaussage, random_head-Rauschen ist keine Modellaussage). Schreibt NUR eine Diagnosedatei,
ueberschreibt keine bestehende Ergebnisdatei.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
import run_landuse_probe as rlp
from eval_landuse import linear_probe

BEST_C = {"tessera_dedup": 10.0, "ae_dedup": 1.0}
CONFIGS = ["roh", "trainiert_projiziert"]
N_BOOTSTRAP = 2000
SEED = 3000
OUT_PATH = cfg.DATA / "processed" / "diag_backbone_paired_bootstrap_landuse.csv"


def main():
    import torch
    from model import ProjectionHead
    fits = {}
    for name in ("tessera_dedup", "ae_dedup"):
        df = rlp.load_clean(name)
        emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
        y = df["code_2021"].to_numpy()
        split = df["split"].to_numpy()

        import yaml
        backbone = name.split("_", 1)[0]
        yaml_path = cfg.ROOT / "configs" / f"{backbone}_v1_dedup.yaml"
        with open(yaml_path) as f:
            yaml_cfg = yaml.safe_load(f)
        yaml_mode, yaml_stats_path = rlp.resolve_normalization(yaml_cfg, str(yaml_path))
        if yaml_mode == rlp.ZSCORE:
            from normalize import apply_zscore, load_stats
            emb_raw = apply_zscore(emb, load_stats(yaml_stats_path))
        else:
            emb_raw = emb

        # Trainierter Kopf: dedup-Checkpoint desselben Namens (tessera_v1_dedup / ae_v1_dedup)
        ckpt_path = cfg.ROOT / "checkpoints" / f"{name.split('_')[0]}_v1_dedup" / "best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ee_dim = ckpt["cfg"]["ee_dim"]
        head = ProjectionHead(ee_dim, rlp.LATENT_DIM, rlp.HIDDEN_DIM, rlp.DROPOUT)
        state = {k.replace("ee_head.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("ee_head.")}
        head.load_state_dict(state)
        head.eval()

        X_by_kind = {"roh": emb_raw, "trainiert_projiziert": rlp.project(head, emb_raw)}
        cell_ids = rlp.load_test_pixel_cell_ids(name)
        train_m, val_m, test_m = split == "train", split == "val", split == "test"
        for kind in CONFIGS:
            X = X_by_kind[kind]
            print(f"Fitte {name}/{kind} (C={BEST_C[name]}, dim={X.shape[1]})...")
            r = linear_probe(X[train_m], y[train_m], X[val_m], y[val_m], X[test_m], y[test_m],
                              C=BEST_C[name], max_iter=1000, return_fit=True)
            fits[(name, kind)] = {"y_test": y[test_m], "pred_test": r["pred_test"], "cell_ids": cell_ids,
                                  "test_oa": r["test_oa"], "test_macro_f1": r["test_macro_f1"]}
            print(f"  test_oa={r['test_oa']:.4f} test_macro_f1={r['test_macro_f1']:.4f}")

    ct = fits[("tessera_dedup", "roh")]["cell_ids"]
    ca = fits[("ae_dedup", "roh")]["cell_ids"]
    common = np.intersect1d(np.unique(ct), np.unique(ca))
    print(f"\nGemeinsame Test-Zellen: {len(common)} (TESSERA {len(np.unique(ct))}, AE {len(np.unique(ca))})")
    assert len(common) == len(np.unique(ct)) == len(np.unique(ca)), "Zellmengen weichen ab - Bug/Annahme verletzt."

    rows = []
    rng = np.random.default_rng(SEED)
    resamples = [rng.choice(common, size=len(common), replace=True) for _ in range(N_BOOTSTRAP)]
    for kind in CONFIGS:
        ft = fits[("tessera_dedup", kind)]
        fa = fits[("ae_dedup", kind)]
        idx_t = {c: np.where(ft["cell_ids"] == c)[0] for c in common}
        idx_a = {c: np.where(fa["cell_ids"] == c)[0] for c in common}
        point_diff_oa = ft["test_oa"] - fa["test_oa"]
        point_diff_f1 = ft["test_macro_f1"] - fa["test_macro_f1"]
        diffs_oa = np.empty(N_BOOTSTRAP)
        diffs_f1 = np.empty(N_BOOTSTRAP)
        for i, cells in enumerate(resamples):
            it = np.concatenate([idx_t[c] for c in cells])
            ia = np.concatenate([idx_a[c] for c in cells])
            oa_t = accuracy_score(ft["y_test"][it], ft["pred_test"][it])
            oa_a = accuracy_score(fa["y_test"][ia], fa["pred_test"][ia])
            f1_t = f1_score(ft["y_test"][it], ft["pred_test"][it], average="macro")
            f1_a = f1_score(fa["y_test"][ia], fa["pred_test"][ia], average="macro")
            diffs_oa[i] = oa_t - oa_a
            diffs_f1[i] = f1_t - f1_a
        for metric, point, diffs in (("oa", point_diff_oa, diffs_oa), ("macro_f1", point_diff_f1, diffs_f1)):
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            sig = bool(not (lo <= 0 <= hi))
            rows.append({"config": kind, "metric": metric, "tessera_dedup": (ft["test_oa"] if metric == "oa" else ft["test_macro_f1"]),
                        "ae_dedup": (fa["test_oa"] if metric == "oa" else fa["test_macro_f1"]),
                        "point_diff_tessera_minus_ae": point, "ci_low": lo, "ci_high": hi,
                        "significant": sig, "n_common_cells": len(common)})
            print(f"  [{kind}/{metric}] TESSERA-AE = {point:+.4f}  95%-CI=[{lo:+.4f}, {hi:+.4f}]  signifikant={sig}")

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
