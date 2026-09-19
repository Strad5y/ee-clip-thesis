"""
run_class_retrieval_tessera_v1dedup.py

Schritt 3 (2026-09-19, Jakob): klassenbasiertes Text->Pixel-Retrieval mit
checkpoints/tessera_v1_dedup/best.pt auf dem deduplizierten pixels_tessera_dedup.parquet -
identische Methodik wie run_class_retrieval.py (dessen Funktionen unveraendert importiert
werden), nur Checkpoint-/Datenpfad sind neu. Ersetzt nichts - neuer Output
class_retrieval_tessera_v1_dedup.csv.
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
from poi_classes import KEY_FRAME
from run_class_retrieval import (
    DEVICE, K_LIST, MIN_RELEVANT, RANDOM_SEED, build_query_text, load_pixel_classes,
    metrics_for_ranking, project,
)
from run_retrieval_baselines import apply_checkpoint_normalization, load_model

CKPT_PATH = Path("checkpoints/tessera_v1_dedup/best.pt")
PIXELS_PATH = "data/processed/pixels_tessera_dedup.parquet"
TEXT_EMB_PATH = "data/processed/pixel_text_embeddings.parquet"
OUT_PATH = Path("data/processed") / "class_retrieval_tessera_v1_dedup.csv"


def run() -> None:
    model, cfg_ckpt = load_model(CKPT_PATH)
    model.eval()

    print("\n--- Galerie: Klassenmenge je Pixel ---")
    px = load_pixel_classes(PIXELS_PATH)
    test_mask = (px["split"] == "test").to_numpy()
    n_test = int(test_mask.sum())
    print(f"  Test-Split: {n_test} Pixel")

    n_classes_arr = px.loc[test_mask, "n_classes"].to_numpy()
    print(f"  Klassenanzahl je Test-Pixel: median={np.median(n_classes_arr):.1f} "
          f"p90={np.percentile(n_classes_arr, 90):.1f} max={n_classes_arr.max()}")

    print("\n--- EO-Embeddings laden, normalisieren, projizieren ---")
    ds = PairDataset(PIXELS_PATH, TEXT_EMB_PATH)
    assert len(ds) == len(px)
    assert (ds.split == px["split"].to_numpy()).all()
    apply_checkpoint_normalization(ds, cfg_ckpt)

    ee_test = ds.ee[test_mask]
    gallery_z = project(model.ee_head, ee_test)
    print(f"  Galeriegroesse: {len(gallery_z)} Pixel, {gallery_z.shape[1]} Dim")

    classes_test = px.loc[test_mask, "classes"].reset_index(drop=True)
    n_classes_test = px.loc[test_mask, "n_classes"].reset_index(drop=True).to_numpy()

    print("\n--- Queries: Klassen mit >= 100 relevanten Test-Pixeln ---")
    class_counts: dict = {}
    for cls_set in classes_test:
        for c in cls_set:
            class_counts[c] = class_counts.get(c, 0) + 1
    all_classes = sorted(class_counts.items(), key=lambda kv: -kv[1])
    queries = [(c, n) for c, n in all_classes if n >= MIN_RELEVANT]
    print(f"  Klassen insgesamt (>=1 Pixel): {len(all_classes)}")
    print(f"  Queries (>= {MIN_RELEVANT} relevante Pixel): {len(queries)}")

    query_texts = [build_query_text(k, v) for (k, v), _ in queries]
    print("\n--- Query-Texte einbetten ---")
    t0 = time.time()
    text_model = load_text_encoder()
    query_raw = np.asarray(text_model.encode(query_texts, normalize_embeddings=True), dtype=np.float32)
    del text_model
    torch.cuda.empty_cache()
    print(f"  {len(query_texts)} Queries eingebettet ({time.time() - t0:.1f}s)")
    query_z = project(model.text_head, query_raw)

    print("\n--- Relevanz-Membership-Matrix ---")
    class_to_qidx = {c: i for i, (c, _) in enumerate(queries)}
    n_q, n_g = len(queries), len(gallery_z)
    relevant = np.zeros((n_q, n_g), dtype=bool)
    for gi, cls_set in enumerate(classes_test):
        for c in cls_set:
            qi = class_to_qidx.get(c)
            if qi is not None:
                relevant[qi, gi] = True

    print("\n--- Kosinus-Ranking ---")
    q_t = torch.from_numpy(query_z).to(DEVICE)
    g_t = torch.from_numpy(gallery_z).to(DEVICE)
    sim = (q_t @ g_t.T).cpu().numpy()

    rng = np.random.default_rng(RANDOM_SEED)
    random_perm = rng.permutation(n_g)
    single_mask = n_classes_test == 1
    multi_mask = n_classes_test > 1
    bucket_masks = {"single": single_mask, "multi": multi_mask}

    all_rows = []
    for qi, ((key, value), n_rel_total) in enumerate(queries):
        order = np.argsort(-sim[qi])
        row = {"class_key": key, "class_value": value, "n_relevant": n_rel_total,
               "n_gallery": n_g, "random_baseline": n_rel_total / n_g}
        row.update(metrics_for_ranking(relevant[qi][order], n_rel_total))
        m_rand = metrics_for_ranking(relevant[qi][random_perm], n_rel_total)
        for k, v in m_rand.items():
            row[f"{k}_random_actual"] = v
        for bname, bmask in bucket_masks.items():
            n_rel_b = int((relevant[qi] & bmask).sum())
            order_b = order[bmask[order]]
            mb = metrics_for_ranking(relevant[qi][order_b], n_rel_b)
            for k, v in mb.items():
                row[f"{k}_{bname}"] = v
            row[f"n_relevant_{bname}"] = n_rel_b
            row[f"n_gallery_{bname}"] = int(bmask.sum())
        all_rows.append(row)

    result = pd.DataFrame(all_rows)
    macro_cols = ["p_at_10", "p_at_50", "p_at_100", "r_precision", "average_precision",
                  "random_baseline", "p_at_10_random_actual", "p_at_50_random_actual",
                  "p_at_100_random_actual", "r_precision_random_actual", "average_precision_random_actual"]
    macro = result[macro_cols].mean(numeric_only=True)
    summary_row = {"class_key": "", "class_value": "MACRO_MEAN", "n_relevant": result["n_relevant"].sum(),
                   "n_gallery": n_g}
    summary_row.update(macro.to_dict())
    result_out = pd.concat([result, pd.DataFrame([summary_row])], ignore_index=True)
    result_out.to_csv(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH}")
    print(f"\n=== Makro-Mittelwerte ({len(queries)} Queries) ===")
    print(macro.to_string())


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"class_retrieval_tessera_v1_dedup_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print("=== run_class_retrieval_tessera_v1dedup.py gestartet ===")
    print(f"Log-Datei: {log_path}")
    print(f"Device: {DEVICE}")
    t0 = time.time()
    run()
    print(f"\nGesamtlaufzeit: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
