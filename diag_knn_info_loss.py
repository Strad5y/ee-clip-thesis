"""
diag_knn_info_loss.py

Diagnose-Aufgabe (2026-09-18, Jakob), Teil 4: k-NN-Klassifikation (k=10) der Land-Use-Labels
im ROHEN Raum (Arm A) vs. im projizierten 256-d-Raum (Arm C, trainierter Kopf), auf DENSELBEN
Testpixeln, fuer beide Backbones - prueft, ob der trainierte Kopf bei TESSERA relativ zu AE
mehr Information verliert (unabhaengig vom linearen Probe-Klassifikator aus run_landuse_probe.py:
k-NN reagiert auf lokale Nachbarschaftsstruktur, nicht nur auf lineare Trennbarkeit).

Aus Rechenzeitgruenden (kNN gegen 1,5+ Mio. Referenzpunkte ist auf CPU nicht praktikabel):
Referenz(Train)- und Query(Test)-Menge werden mit fixem Seed subsampled, IDENTISCH fuer roh
und projiziert (gleiche Testpixel, wie gefordert). Euklidische Distanz fuer beide Raeume
(projiziert ist ohnehin L2-normalisiert, dort ist euklidisch <-> Kosinus monoton aequivalent;
TESSERA roh ist z-score-normalisiert, AE roh ist bereits unit-norm - Euklidisch ist in beiden
Faellen die natuerliche Wahl, keine gesonderte Kosinus-Variante noetig).

Nur lesen/berichten, kein Neutraining der CLIP-/SigLIP-Gewichte.
"""
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
from normalize import ZSCORE, apply_zscore, load_stats
from run_landuse_probe import (
    EXCLUDED_CLASS, load_backbone_yaml_config, load_trained_head, project, resolve_normalization,
)

K = 10
N_REF = 50_000
N_QUERY = 20_000
SEED = 0


def load_and_project(backbone: str):
    path = cfg.DATA / "processed" / f"landuse_pairs_{backbone}.parquet"
    df = pd.read_parquet(path, columns=["embedding", "code_2021", "split"])
    df = df[(df["code_2021"] != EXCLUDED_CLASS) & (df["split"] != "missing")].reset_index(drop=True)
    emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    y = df["code_2021"].to_numpy()
    split = df["split"].to_numpy()

    yaml_cfg = load_backbone_yaml_config(backbone)
    mode, stats_path = resolve_normalization(yaml_cfg, f"configs/{backbone}_v1.yaml")
    emb_raw = apply_zscore(emb, load_stats(stats_path)) if mode == ZSCORE else emb

    head, ckpt_cfg = load_trained_head(backbone)
    ckpt_mode, ckpt_stats_path = resolve_normalization(ckpt_cfg, f"Checkpoint-Config ({backbone})")
    assert (ckpt_mode, ckpt_stats_path) == (mode, stats_path), "Normalisierung config vs. checkpoint weicht ab"
    emb_proj = project(head, emb_raw)

    return emb_raw, emb_proj, y, split


def run_knn(backbone: str, rng: np.random.Generator) -> dict:
    print(f"\n=== {backbone} ===")
    emb_raw, emb_proj, y, split = load_and_project(backbone)
    print(f"  {len(y)} Zeilen nach Filter, roh dim={emb_raw.shape[1]}, projiziert dim={emb_proj.shape[1]}")

    train_idx_all = np.where(split == "train")[0]
    test_idx_all = np.where(split == "test")[0]
    ref_idx = rng.choice(train_idx_all, size=min(N_REF, len(train_idx_all)), replace=False)
    query_idx = rng.choice(test_idx_all, size=min(N_QUERY, len(test_idx_all)), replace=False)
    print(f"  Referenz (Train): {len(ref_idx)}, Query (Test): {len(query_idx)} - IDENTISCH fuer roh/projiziert")

    result = {"backbone": backbone, "n_ref": len(ref_idx), "n_query": len(query_idx)}
    for space_name, emb in (("roh", emb_raw), ("projiziert", emb_proj)):
        t0 = time.time()
        clf = KNeighborsClassifier(n_neighbors=K, metric="euclidean", n_jobs=-1)
        clf.fit(emb[ref_idx], y[ref_idx])
        pred = clf.predict(emb[query_idx])
        oa = accuracy_score(y[query_idx], pred)
        f1 = f1_score(y[query_idx], pred, average="macro", zero_division=0)
        dt = time.time() - t0
        print(f"  {space_name}: oa={oa:.4f} macro_f1={f1:.4f} ({dt:.1f}s)")
        result[f"{space_name}_oa"] = oa
        result[f"{space_name}_macro_f1"] = f1
    result["delta_oa_proj_minus_roh"] = result["projiziert_oa"] - result["roh_oa"]
    result["delta_macro_f1_proj_minus_roh"] = result["projiziert_macro_f1"] - result["roh_macro_f1"]
    return result


def main():
    rng = np.random.default_rng(SEED)
    rows = [run_knn(bb, rng) for bb in ("tessera", "ae")]
    df = pd.DataFrame(rows)
    out_path = cfg.DATA / "processed" / "diag_knn_info_loss.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\nGespeichert: {out_path}")


if __name__ == "__main__":
    main()
