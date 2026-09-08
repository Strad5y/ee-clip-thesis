"""
compute_pixel_text_embeddings.py

Phase 6 (CLAUDE_CODE_TASKS_phase4_7.md): Encoding fuer die eindeutigen pixel_text-Werte
(Phase 5, aggregate_pixel_pairs.py) - GETEILTER Cache ueber BEIDE Backbones (TESSERA + AE),
nicht zwei getrennte Laeufe: die Schnittmenge ihrer eindeutigen Texte ist mit 158.403 von
243.237/237.811 gross genug (65%/67%), dass ein gemeinsamer, textbasierter Cache echte
Rechenzeit spart statt sie zu verdoppeln.

Modell UNVERAENDERT (Qwen3-Embedding-4B, truncate_dim=1024, normalize_embeddings=True, kein
Praefix) - reine Wiederverwendung von compute_text_embeddings.py::load_text_encoder() und
compute_poi_text_embeddings.py::embed_unique_texts() (Dedup passiert dort bereits ueber ALLE
uebergebenen Texte, hier einfach beide Backbones vor dem Aufruf konkateniert).

NEUER Cache-Dateiname (pixel_text_embeddings.parquet) - ueberschreibt NICHT text_embeddings.
parquet (HILUCS-Zero-Shot-Prompts) oder einen etwaigen poi-text_embeddings.parquet-Stand.
Format: (text, embedding) - textbasiert, nicht poi_id/pixel-keyed, da BEIDE Backbones dieselbe
Datei nutzen sollen (ueber pixel_text mergen, nicht ueber poi_id/tile/row/col).
"""
import sys
import time

import numpy as np
import pandas as pd

import config as cfg
from chunked_join_runner import LOG_DIR, _Tee
from compute_poi_text_embeddings import embed_unique_texts
from compute_text_embeddings import MRL_DIM, load_text_encoder

PIXELS_PATHS = {
    "tessera": cfg.DATA / "processed" / "pixels_tessera.parquet",
    "ae": cfg.DATA / "processed" / "pixels_ae.parquet",
}
OUT = cfg.DATA / "processed" / "pixel_text_embeddings.parquet"


def report_token_lengths(union_texts: list[str]) -> None:
    """Vorab-Check MIT dem echten Tokenizer (kein Modell-Forward-Pass noetig) - Nutzervorgabe:
    bei Ueberschreitung der Modellgrenze stoppen, NICHT stillschweigend trunkieren."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")
    lens = np.array([len(tok.encode(t)) for t in union_texts])
    median, p99, mx = np.median(lens), np.percentile(lens, 99), lens.max()
    print(f"Tokenlaengen (Tokenizer-gemessen, {len(union_texts)} Texte): "
          f"median={median:.0f}, p99={p99:.0f}, max={mx}")

    model_max = tok.model_max_length
    n_over = int((lens > model_max).sum())
    print(f"Modellgrenze (tokenizer.model_max_length): {model_max}")
    if n_over > 0:
        raise RuntimeError(
            f"{n_over} Texte ueberschreiten die Modellgrenze ({model_max} Tokens) - "
            "Nutzervorgabe: stoppen, nicht trunkieren.")
    print("Keine Ueberschreitung - alle Texte innerhalb der Modellgrenze.")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pixel_text_embed_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    print("=== compute_pixel_text_embeddings.py gestartet ===")
    print(f"Log-Datei: {log_path}")

    texts_per_backbone = {}
    for name, path in PIXELS_PATHS.items():
        t = pd.read_parquet(path, columns=["pixel_text"])["pixel_text"]
        texts_per_backbone[name] = t
        print(f"{name}: {len(t)} Pixel, {t.nunique()} eindeutige pixel_text")

    combined = pd.concat(texts_per_backbone.values(), ignore_index=True)
    union_texts = combined.drop_duplicates().tolist()
    t_set = set(texts_per_backbone["tessera"].unique())
    a_set = set(texts_per_backbone["ae"].unique())
    print(f"\nSchnittmenge TESSERA/AE: {len(t_set & a_set)}")
    print(f"Vereinigung (= Groesse des gemeinsamen Caches): {len(union_texts)}")

    report_token_lengths(union_texts)

    t_load = time.time()
    model = load_text_encoder()
    print(f"\nModell geladen ({MRL_DIM}-dim, CUDA) in {time.time() - t_load:.1f}s")

    t_encode = time.time()
    text_embeddings = embed_unique_texts(combined, model)
    dt = time.time() - t_encode
    print(f"Encoding fertig in {dt / 60:.1f} min ({len(union_texts)} Texte, "
          f"{len(union_texts) / dt:.1f} Texte/s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text_embeddings.to_parquet(OUT)
    print(f"Gespeichert: {OUT} ({len(text_embeddings)} Texte, dim={MRL_DIM})")
    print("=== compute_pixel_text_embeddings.py fertig ===")


if __name__ == "__main__":
    main()
