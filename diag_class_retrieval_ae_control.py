"""
diag_class_retrieval_ae_control.py  (2026-09-21, Jakob) - NUR DIAGNOSE, ueberschreibt keine Ergebnisdatei

Frage: warum liegen die Class-Retrieval-Metriken von ae_v1_dedup auf der neuen Galerie niedriger
als die von ae_v1 auf der alten? Trennt Checkpoint-Effekt und Galerie-Effekt (2 Galerien x 3
Checkpoints), mit DENSELBEN 185 Queries und derselben Metrikfunktion wie run_class_retrieval.py
(unveraendert importiert; Query-Texte per Qwen3-Encoder einmal berechnet).

  Galerien : alt = Test-Split pixels_ae.parquet (777.608), neu = Test-Split pixels_ae_dedup.parquet (728.588)
  Checkpoints: ae_v1 (alt, unseeded), ae_v1_predup_control (seed 0, alte Daten), ae_v1_dedup (seed 0, neue Daten)

Zusaetzlich je Klasse: n_relevant in alter/neuer Galerie und relativer Verlust. Konsistenzcheck:
(ae_v1, alt) muss class_retrieval_ae.csv und (ae_v1_dedup, neu) class_retrieval_ae_v1_dedup.csv
reproduzieren. Ausgaben: data/processed/diag_class_retrieval_ae_control_{per_query,summary,
gallery_relevance}.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chunked_join_runner import LOG_DIR, _Tee
from compute_text_embeddings import load_text_encoder
from dataset import PairDataset
from run_class_retrieval import (DEVICE, MIN_RELEVANT, build_query_text, load_pixel_classes,
                                 metrics_for_ranking, project)
from run_retrieval_baselines import apply_checkpoint_normalization, load_model

P = Path("data/processed")
GALLERIES = {"alt": "data/processed/pixels_ae.parquet", "neu": "data/processed/pixels_ae_dedup.parquet"}
CKPTS = {"ae_v1": Path("checkpoints/ae_v1/best.pt"),
         "predup_control": Path("checkpoints/ae_v1_predup_control/best.pt"),
         "ae_v1_dedup": Path("checkpoints/ae_v1_dedup/best.pt")}
TEXT_EMB_PATH = "data/processed/pixel_text_embeddings.parquet"
METRICS = ["p_at_10", "p_at_50", "p_at_100", "r_precision", "average_precision"]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_DIR / f"diag_class_retrieval_ae_control_{time.strftime('%Y%m%d_%H%M%S')}.log", "a")
    sys.stdout, sys.stderr = _Tee(sys.stdout, log_file), _Tee(sys.stderr, log_file)

    # Queries: exakt die 185 Klassen der gespeicherten Ergebnisse (Reihenfolge wie dort)
    ref_old = pd.read_csv(P / "class_retrieval_ae.csv")
    ref_new = pd.read_csv(P / "class_retrieval_ae_v1_dedup.csv")
    ref_old, ref_new = (r[r["class_value"] != "MACRO_MEAN"].reset_index(drop=True) for r in (ref_old, ref_new))
    q_old, q_new = list(zip(ref_old.class_key, ref_old.class_value)), list(zip(ref_new.class_key, ref_new.class_value))
    assert set(q_old) == set(q_new) and len(q_old) == 185, "Query-Mengen alt/neu unterscheiden sich"
    queries = q_old
    print(f"{len(queries)} Queries (identisch in alt/neu)")

    text_model = load_text_encoder()
    query_raw = np.asarray(text_model.encode([build_query_text(k, v) for k, v in queries],
                                             normalize_embeddings=True), dtype=np.float32)
    del text_model
    torch.cuda.empty_cache()

    per_query, gallery_rel = [], {}
    for gname, gpath in GALLERIES.items():
        print(f"\n=== Galerie {gname}: {gpath} ===")
        px = load_pixel_classes(gpath)
        test_mask = (px["split"] == "test").to_numpy()
        classes_test = px.loc[test_mask, "classes"].reset_index(drop=True)
        n_g = int(test_mask.sum())
        qidx = {c: i for i, c in enumerate(queries)}
        relevant = np.zeros((len(queries), n_g), dtype=bool)
        for gi, cls_set in enumerate(classes_test):
            for c in cls_set:
                qi = qidx.get(c)
                if qi is not None:
                    relevant[qi, gi] = True
        n_rel = relevant.sum(axis=1)
        gallery_rel[gname] = (n_g, n_rel)
        print(f"  Galerie: {n_g} Pixel; Klassen mit < {MIN_RELEVANT} relevanten Pixeln: {int((n_rel < MIN_RELEVANT).sum())}")

        ds = PairDataset(gpath, TEXT_EMB_PATH)
        assert len(ds) == len(px) and (ds.split == px["split"].to_numpy()).all()
        ee_test_raw = ds.ee[test_mask].copy()
        del ds
        for cname, cpath in CKPTS.items():
            model, ckcfg = load_model(cpath)
            assert ckcfg.get("normalization", "none") == "none", "AE erwartet normalization=none"
            gallery_z = project(model.ee_head, ee_test_raw)
            query_z = project(model.text_head, query_raw)
            sim = torch.from_numpy(query_z).to(DEVICE) @ torch.from_numpy(gallery_z).to(DEVICE).T
            sim = sim.cpu().numpy()
            for qi, (key, value) in enumerate(queries):
                order = np.argsort(-sim[qi])
                m = metrics_for_ranking(relevant[qi][order], int(n_rel[qi]))
                per_query.append({"checkpoint": cname, "gallery": gname, "class_key": key, "class_value": value,
                                  "n_relevant": int(n_rel[qi]), "n_gallery": n_g,
                                  "random_baseline": n_rel[qi] / n_g, **m})
            print(f"  {cname}: AP macro={np.mean([r['average_precision'] for r in per_query if r['checkpoint']==cname and r['gallery']==gname]):.5f}")
            del model, gallery_z, sim
            torch.cuda.empty_cache()

    pq = pd.DataFrame(per_query)
    pq["ap_lift"] = pq["average_precision"] / pq["random_baseline"]
    pq.to_csv(P / "diag_class_retrieval_ae_control_per_query.csv", index=False)

    summ = (pq.groupby(["checkpoint", "gallery"])[METRICS + ["random_baseline", "ap_lift"]].mean().reset_index())
    summ.to_csv(P / "diag_class_retrieval_ae_control_summary.csv", index=False)

    n_g_old, rel_old = gallery_rel["alt"]
    n_g_new, rel_new = gallery_rel["neu"]
    gr = pd.DataFrame({"class_key": [k for k, _ in queries], "class_value": [v for _, v in queries],
                       "n_relevant_alt": rel_old, "n_relevant_neu": rel_new})
    gr["abs_loss"] = gr["n_relevant_alt"] - gr["n_relevant_neu"]
    gr["rel_loss"] = gr["abs_loss"] / gr["n_relevant_alt"]
    gr["prevalence_alt"] = gr["n_relevant_alt"] / n_g_old
    gr["prevalence_neu"] = gr["n_relevant_neu"] / n_g_new
    gr = gr.sort_values("rel_loss", ascending=False).reset_index(drop=True)
    gr.to_csv(P / "diag_class_retrieval_ae_control_gallery_relevance.csv", index=False)
    print(f"\nGalerie gesamt: alt {n_g_old} -> neu {n_g_new} ({100 * (n_g_old - n_g_new) / n_g_old:.2f}% weniger)")

    # Konsistenzcheck gegen gespeicherte Ergebnisdateien
    for (ck, g), ref in ((("ae_v1", "alt"), ref_old), (("ae_v1_dedup", "neu"), ref_new)):
        sub = pq[(pq.checkpoint == ck) & (pq.gallery == g)].set_index(["class_key", "class_value"])
        r = ref.set_index(["class_key", "class_value"])
        d = max(float((sub[m_] - r[m_.replace("p_at", "p_at")]).abs().max()) for m_ in ["p_at_10", "p_at_50", "r_precision", "average_precision"])
        print(f"Konsistenz ({ck}, {g}) vs. gespeicherte Datei: max|diff| ueber p@10/p@50/r_prec/AP = {d:.3e}")
    print("\n=== Makro-Mittel (185 Queries) ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
