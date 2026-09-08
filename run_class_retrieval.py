"""
run_class_retrieval.py

Klassenbasiertes Text->Pixel-Retrieval ueber die volle Test-Galerie, als quantitative
Retrieval-Evaluation neben der bestehenden In-Batch-Diagnose (run_retrieval_baselines.py).

Wiederverwendet aus run_retrieval_baselines.py: load_model() (Architektur + Checkpoint aus
ckpt['cfg']), apply_checkpoint_normalization() (zscore/none je nach Checkpoint-Config).
Wiederverwendet aus poi_classes.py: dieselbe KEY_FRAME/FRAME_NOUN/_split_glieder-Basis wie
extract_primary_class() - Mehrfachklassen-Extraktion via extract_all_classes() (Nutzervorgabe
T6-Ergaenzung: Relevanz ist mengenwertig, ein Pixel kann fuer mehrere Query-Klassen relevant
sein; n_relevant je Klasse ist die Anzahl Pixel, deren Klassenmenge diese Klasse enthaelt -
Summe ueber alle Klassen > Galeriegroesse, das ist korrekt).

Queries: eine je Value-Klasse mit >=100 relevanten Galerie-Pixeln, Text nach demselben
Single-Tag-Template wie build_poi_text.py, aber OHNE Namen und OHNE Objektbeschreibung (reine
Klassenbeschreibung: "{Frame}: {value}."). Eingebettet mit dem echten Qwen3-Text-Encoder
(load_text_encoder(), dieselbe Pipeline wie compute_pixel_text_embeddings.py - KEIN
Cache-Lookup, da diese Query-Strings nicht Teil des Caches sind), durch model.text_head projiziert.

Galerie: alle Test-Split-Pixel, EO-Embedding durch model.ee_head projiziert. Buckets 'all' /
'single' (Pixel mit genau 1 Klasse) / 'multi' (>1 Klasse), analog zu den bestehenden
all/single/multi-Buckets in run_retrieval_baselines.py.

Metriken je Query, macro-gemittelt: P@10/50/100, R-Precision (P@n_relevant), Average Precision
(-> mAP). Baseline: Klassenpraevalenz n_relevant/n_gallery (exakter Erwartungswert bei
zufaelligem Ranking OHNE Zuruecklegen, unabhaengig von k - Standardresultat der hypergeometrischen
Verteilung), PLUS eine tatsaechliche Zufallspermutation (Seed 0, EINE gemeinsame Permutation
fuer alle Queries) als empirische Kontrolle.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as cfg
from chunked_join_runner import LOG_DIR, _Tee
from compute_text_embeddings import load_text_encoder
from dataset import PairDataset
from poi_classes import KEY_FRAME, extract_all_classes
from run_retrieval_baselines import apply_checkpoint_normalization, load_model

MIN_RELEVANT = 100
K_LIST = (10, 50, 100)
RANDOM_SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("data/processed")


def build_query_text(key: str, value: str) -> str:
    """Single-Tag-Template aus build_poi_text.py, OHNE Namen, OHNE Objektbeschreibung - reine
    Klassenbeschreibung. '{name} — {body}' Zweig entfaellt (kein Name), also greift
    _prefix_name()s No-Name-Zweig: erster Buchstabe von body gross."""
    frame = KEY_FRAME[key]
    body = f"{frame}: {value}."
    return body[0].upper() + body[1:]


def load_pixel_classes(pixels_path: str) -> pd.DataFrame:
    """Liest pixel_text/split, extrahiert die Klassenmenge JE EINDEUTIGEM Text (nicht pro
    Zeile - Millionen Pixel teilen sich wenige zehntausend eindeutige Texte) und broadcastet
    zurueck. Gibt eine Zeile je Pixel zurueck: split, classes (frozenset), n_classes,
    n_unparsed."""
    df = pd.read_parquet(pixels_path, columns=["pixel_text", "split"])
    unique_texts = df["pixel_text"].drop_duplicates().reset_index(drop=True)

    t0 = time.time()
    parsed = {}
    total_unparsed = 0
    for t in unique_texts:
        classes, n_bad = extract_all_classes(t)
        parsed[t] = (frozenset(classes), n_bad)
        total_unparsed += n_bad
    print(f"  {len(unique_texts)} eindeutige Texte geparst ({time.time() - t0:.1f}s), "
          f"{total_unparsed} unaufloesbare Einheiten insgesamt (siehe Bericht).")

    df["classes"] = df["pixel_text"].map(lambda t: parsed[t][0])
    df["n_classes"] = df["classes"].map(len)
    return df


@torch.no_grad()
def project(head, x: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
    head = head.to(DEVICE)
    out = np.empty((len(x), head.net[-1].out_features), dtype=np.float32)
    for i in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[i:i + batch_size]).to(DEVICE)
        out[i:i + batch_size] = head(batch).cpu().numpy()
    return out


def metrics_for_ranking(ranked_relevant: np.ndarray, n_relevant: int) -> dict:
    """ranked_relevant: bool-Array (n_gallery,), Galerie bereits nach Aehnlichkeit absteigend
    sortiert. Liefert P@k je k in K_LIST, R-Precision, Average Precision."""
    cum_hits = np.cumsum(ranked_relevant)
    ranks = np.arange(1, len(ranked_relevant) + 1)
    precision_at_rank = cum_hits / ranks

    out = {}
    for k in K_LIST:
        out[f"p_at_{k}"] = float(precision_at_rank[k - 1]) if k <= len(ranked_relevant) else np.nan

    if n_relevant > 0:
        kc = min(n_relevant, len(ranked_relevant))
        out["r_precision"] = float(precision_at_rank[kc - 1])
        hits = precision_at_rank[ranked_relevant]
        out["average_precision"] = float(hits.mean()) if hits.size > 0 else 0.0
    else:
        out["r_precision"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def run_backbone(backbone: str) -> None:
    print(f"\n{'=' * 70}\n=== Backbone: {backbone} ===\n{'=' * 70}")
    ckpt_path = Path(f"checkpoints/{backbone}_v1/best.pt")
    pixels_path = f"data/processed/pixels_{backbone}.parquet"
    text_emb_path = "data/processed/pixel_text_embeddings.parquet"

    model, cfg_ckpt = load_model(ckpt_path)
    model.eval()

    print("\n--- Galerie: Klassenmenge je Pixel ---")
    px = load_pixel_classes(pixels_path)
    test_mask = (px["split"] == "test").to_numpy()
    n_test = int(test_mask.sum())
    print(f"  Test-Split: {n_test} Pixel")

    n_classes_arr = px.loc[test_mask, "n_classes"].to_numpy()
    print(f"  Klassenanzahl je Test-Pixel: median={np.median(n_classes_arr):.1f} "
          f"p90={np.percentile(n_classes_arr, 90):.1f} max={n_classes_arr.max()}")
    frac_single = float((n_classes_arr == 1).mean())
    print(f"  Anteil Pixel mit genau 1 Klasse: {frac_single:.4%}")

    print("\n--- EO-Embeddings laden, normalisieren, projizieren ---")
    ds = PairDataset(pixels_path, text_emb_path)
    assert len(ds) == len(px), f"Laengen-Mismatch: PairDataset={len(ds)} vs. px={len(px)}"
    assert (ds.split == px["split"].to_numpy()).all(), \
        "Split-Reihenfolge zwischen PairDataset und direktem Parquet-Read weicht ab."
    apply_checkpoint_normalization(ds, cfg_ckpt)

    ee_test = ds.ee[test_mask]
    gallery_z = project(model.ee_head, ee_test)
    print(f"  Galeriegroesse: {len(gallery_z)} Pixel, {gallery_z.shape[1]} Dim")

    classes_test = px.loc[test_mask, "classes"].reset_index(drop=True)
    n_classes_test = px.loc[test_mask, "n_classes"].reset_index(drop=True).to_numpy()

    print("\n--- Queries: Klassen mit >= 100 relevanten Test-Pixeln ---")
    class_counts: dict[tuple[str, str], int] = {}
    for cls_set in classes_test:
        for c in cls_set:
            class_counts[c] = class_counts.get(c, 0) + 1
    all_classes = sorted(class_counts.items(), key=lambda kv: -kv[1])
    queries = [(c, n) for c, n in all_classes if n >= MIN_RELEVANT]
    n_excluded = len(all_classes) - len(queries)
    print(f"  Klassen insgesamt (>=1 Pixel): {len(all_classes)}")
    print(f"  Queries (>= {MIN_RELEVANT} relevante Pixel): {len(queries)}")
    print(f"  Ausgeschlossen (< {MIN_RELEVANT}): {n_excluded}")

    query_texts = [build_query_text(k, v) for (k, v), _ in queries]
    print("\n--- Query-Texte einbetten (Qwen3-Encoder, kein Cache-Lookup) ---")
    t0 = time.time()
    text_model = load_text_encoder()
    query_raw = np.asarray(text_model.encode(query_texts, normalize_embeddings=True), dtype=np.float32)
    del text_model
    torch.cuda.empty_cache()
    print(f"  {len(query_texts)} Queries eingebettet ({time.time() - t0:.1f}s)")
    query_z = project(model.text_head, query_raw)

    print("\n--- Relevanz-Membership-Matrix (Queries x Galerie) ---")
    class_to_qidx = {c: i for i, (c, _) in enumerate(queries)}
    n_q, n_g = len(queries), len(gallery_z)
    relevant = np.zeros((n_q, n_g), dtype=bool)
    for gi, cls_set in enumerate(classes_test):
        for c in cls_set:
            qi = class_to_qidx.get(c)
            if qi is not None:
                relevant[qi, gi] = True
    total_relevant_cells = int(relevant.sum())
    print(f"  Summe n_relevant ueber alle Queries: {total_relevant_cells} "
          f"(> Galeriegroesse {n_g} durch Mehrfachklassen, erwartet)")

    print("\n--- Kosinus-Ranking ---")
    t0 = time.time()
    q_t = torch.from_numpy(query_z).to(DEVICE)
    g_t = torch.from_numpy(gallery_z).to(DEVICE)
    sim = (q_t @ g_t.T).cpu().numpy()
    print(f"  Aehnlichkeitsmatrix {sim.shape} berechnet ({time.time() - t0:.1f}s)")

    rng = np.random.default_rng(RANDOM_SEED)
    random_perm = rng.permutation(n_g)

    single_mask = n_classes_test == 1
    multi_mask = n_classes_test > 1
    bucket_masks = {"all": np.ones(n_g, dtype=bool), "single": single_mask, "multi": multi_mask}

    all_rows = []
    for qi, ((key, value), n_rel_total) in enumerate(queries):
        order = np.argsort(-sim[qi])
        prevalence_all = n_rel_total / n_g

        row = {"class_key": key, "class_value": value, "n_relevant": n_rel_total,
               "n_gallery": n_g, "random_baseline": prevalence_all}

        ranked_rel_true = relevant[qi][order]
        m = metrics_for_ranking(ranked_rel_true, n_rel_total)
        row.update(m)

        ranked_rel_rand = relevant[qi][random_perm]
        m_rand = metrics_for_ranking(ranked_rel_rand, n_rel_total)
        for k, v in m_rand.items():
            row[f"{k}_random_actual"] = v

        for bucket_name, bmask in bucket_masks.items():
            if bucket_name == "all":
                continue
            n_rel_bucket = int((relevant[qi] & bmask).sum())
            order_b = order[bmask[order]]
            ranked_rel_b = relevant[qi][order_b]
            mb = metrics_for_ranking(ranked_rel_b, n_rel_bucket)
            for k, v in mb.items():
                row[f"{k}_{bucket_name}"] = v
            row[f"n_relevant_{bucket_name}"] = n_rel_bucket
            row[f"n_gallery_{bucket_name}"] = int(bmask.sum())

        all_rows.append(row)

    result = pd.DataFrame(all_rows)

    macro_cols = ["p_at_10", "p_at_50", "p_at_100", "r_precision", "average_precision",
                  "random_baseline", "p_at_10_random_actual", "p_at_50_random_actual",
                  "p_at_100_random_actual", "r_precision_random_actual",
                  "average_precision_random_actual"]
    macro = result[macro_cols].mean(numeric_only=True)
    summary_row = {"class_key": "", "class_value": "MACRO_MEAN", "n_relevant": result["n_relevant"].sum(),
                   "n_gallery": n_g}
    summary_row.update(macro.to_dict())
    result_out = pd.concat([result, pd.DataFrame([summary_row])], ignore_index=True)

    out_path = OUT_DIR / f"class_retrieval_{backbone}.csv"
    result_out.to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")

    print(f"\n=== Makro-Mittelwerte, {backbone} ({len(queries)} Queries) ===")
    print(macro.to_string())

    top10 = result.sort_values("average_precision", ascending=False).head(10)
    bottom10 = result.sort_values("average_precision", ascending=True).head(10)
    print("\n--- Top-10 AP ---")
    print(top10[["class_key", "class_value", "n_relevant", "average_precision"]].to_string(index=False))
    print("\n--- Bottom-10 AP ---")
    print(bottom10[["class_key", "class_value", "n_relevant", "average_precision"]].to_string(index=False))

    n_single = int(single_mask.sum())
    n_multi = int(multi_mask.sum())
    print(f"\n--- Single/Multi-Buckets (Galerie) ---")
    print(f"  single: {n_single} Pixel ({n_single / n_g:.4%})  multi: {n_multi} Pixel ({n_multi / n_g:.4%})")
    for bucket_name in ("single", "multi"):
        cols = [f"p_at_10_{bucket_name}", f"p_at_50_{bucket_name}", f"p_at_100_{bucket_name}",
                f"r_precision_{bucket_name}", f"average_precision_{bucket_name}"]
        macro_b = result[cols].mean(numeric_only=True)
        print(f"  {bucket_name}: {macro_b.to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Klassenbasiertes Text->Pixel-Retrieval")
    parser.add_argument("--backbone", type=str, required=True, choices=["ae", "tessera"])
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"class_retrieval_{args.backbone}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")
    import sys
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"=== run_class_retrieval.py gestartet ({args.backbone}) ===")
    print(f"Log-Datei: {log_path}")
    print(f"Device: {DEVICE}")

    t0 = time.time()
    run_backbone(args.backbone)
    print(f"\nGesamtlaufzeit: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
