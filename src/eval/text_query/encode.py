"""
encode.py

INSTRUCTIONS_maps_aether.md, Phase 1: Text-Query-Encoder fuer die AETHER-Vergleichskarten.

Kodiert Query-Strings mit demselben eingefrorenen Sentence-Transformer wie im Training
(compute_text_embeddings.py::load_text_encoder(), MODEL_NAME/MRL_DIM importiert - NICHT
dupliziert, identischer Encoder fuer beide Backbones), projiziert dann durch den
Text-Projektor (model.text_head) des jeweiligen kanonischen Dedup-Checkpoints
(checkpoints/{backbone}_v1_dedup/best.pt). ProjectionHead.forward() normiert bereits L2
(model.py, settled decision) - kein zusaetzlicher Normierungsschritt noetig, ebenso normiert
bereits text_model.encode(..., normalize_embeddings=True) die MRL-truncatete Stufe.

Korrektur zur urspruenglichen Instruktionsdatei (2026-09-23, Jakob/Kai-Rueckfrage): "Encoder-
Name ... kommt aus ckpt['cfg']" trifft NICHT zu - ckpt['cfg'] enthaelt keine Encoder-Kennung
(nur text_dim=1024, die Dimension NACH MRL-Truncation). Der Encoder-Name ist eine Modulkonstante
in compute_text_embeddings.py (MODEL_NAME/MRL_DIM), hier importiert statt dupliziert - identisch
zum bereits bestehenden Weg in run_class_retrieval.py. Der Projektor (model.text_head) kommt
weiterhin korrekt aus dem jeweiligen Checkpoint.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Repo-Root

import numpy as np
import pandas as pd
import torch

from compute_text_embeddings import MODEL_NAME, MRL_DIM, load_text_encoder
from run_retrieval_baselines import load_model

CKPT_TEMPLATE = "checkpoints/{backbone}_v1_dedup/best.pt"
TEXT_EMB_CACHE = "data/processed/pixel_text_embeddings.parquet"


def _check_text_dim(backbone: str, cfg: dict) -> None:
    """Pflichtpruefung (Nutzervorgabe 2026-09-23): ckpt['cfg']['text_dim'] muss mit MRL_DIM
    uebereinstimmen - sonst passen Encoder-Ausgabe und der vom Checkpoint erwartete
    Projektor-Input nicht zusammen. Klarer Abbruch statt stillschweigender Shape-Fehler
    spaeter beim Forward-Pass."""
    ckpt_text_dim = cfg.get("text_dim")
    if ckpt_text_dim != MRL_DIM:
        raise ValueError(
            f"{backbone}: checkpoints/{backbone}_v1_dedup/best.pt hat cfg['text_dim']="
            f"{ckpt_text_dim!r}, aber der Encoder ({MODEL_NAME}) liefert MRL_DIM={MRL_DIM} - "
            f"Encoder und Checkpoint passen nicht zusammen, Abbruch."
        )


def load_query_encoder():
    """Laedt den eingefrorenen Sentence-Transformer (dieselbe Instanz fuer beide Backbones -
    nur der Projektor unterscheidet sich pro Checkpoint). Aufrufer ist fuer torch.cuda.
    empty_cache() nach Gebrauch verantwortlich (grosses 4B-Modell, wie in load_text_encoder()
    dokumentiert)."""
    return load_text_encoder()


def encode_queries(texts: list[str], backbone: str, text_model=None,
                    device: str | None = None) -> np.ndarray:
    """Kodiert `texts` mit dem eingefrorenen Sentence-Transformer (MRL-Truncation 2560->1024,
    L2-normiert), projiziert dann durch model.text_head des kanonischen Dedup-Checkpoints fuer
    `backbone` (ebenfalls L2-normiert). Gibt (n, latent_dim) zurueck.

    text_model optional uebergebbar, um das 4B-Modell bei mehreren Aufrufen (z.B. beide
    Backbones nacheinander) nur einmal zu laden - wird NICHT hier freigegeben, das macht der
    Aufrufer."""
    ckpt_path = Path(CKPT_TEMPLATE.format(backbone=backbone))
    model, cfg = load_model(ckpt_path)
    _check_text_dim(backbone, cfg)
    model.eval()
    # load_model() (run_retrieval_baselines.py) verschiebt das Modell bereits auf
    # run_retrieval_baselines.DEVICE (cuda falls verfuegbar) - Eingabetensor muss auf
    # DASSELBE Geraet, sonst RuntimeError (mat1 on cpu, andere Tensoren auf cuda).
    if device is None:
        device = next(model.parameters()).device

    owns_text_model = text_model is None
    if owns_text_model:
        text_model = load_query_encoder()
    try:
        raw = np.asarray(text_model.encode(texts, normalize_embeddings=True), dtype=np.float32)
    finally:
        if owns_text_model:
            del text_model
            torch.cuda.empty_cache()
    assert raw.shape[1] == MRL_DIM, f"Encoder liefert {raw.shape[1]} Dim, erwartet {MRL_DIM}."

    with torch.no_grad():
        projected = model.text_head(torch.from_numpy(raw).to(device)).cpu().numpy()
    return projected


def verify_against_cache(backbone: str, seed: int = 0) -> dict:
    """Pflicht-Check (Phase 1): EIN Trainings-Text aus pixels_{backbone}_dedup.parquet frisch
    kodieren - NUR der Sentence-Transformer-Schritt (VOR dem Text-Projektor), da genau diese
    Stufe in pixel_text_embeddings.parquet gecacht ist (dataset.py::PairDataset nutzt diesen
    Cache 1:1 als Trainings-Text-Embedding, TEXT_COL='pixel_text'). Kosinus zum gecachten
    Embedding muss 1.0 sein."""
    pixels_path = f"data/processed/pixels_{backbone}_dedup.parquet"
    px = pd.read_parquet(pixels_path, columns=["pixel_text", "split"])
    train_texts = px.loc[px["split"] == "train", "pixel_text"]
    if len(train_texts) == 0:
        raise ValueError(f"{backbone}: keine Trainingspixel in {pixels_path} gefunden.")
    rng = np.random.default_rng(seed)
    text = str(train_texts.iloc[int(rng.integers(0, len(train_texts)))])

    cache = pd.read_parquet(TEXT_EMB_CACHE)
    row = cache[cache["text"] == text]
    if len(row) == 0:
        raise ValueError(f"{backbone}: Trainings-Text nicht im Cache {TEXT_EMB_CACHE} gefunden: "
                          f"{text!r}")
    cached_vec = np.asarray(row.iloc[0]["embedding"], dtype=np.float32)

    text_model = load_query_encoder()
    try:
        fresh_vec = np.asarray(text_model.encode([text], normalize_embeddings=True),
                                dtype=np.float32)[0]
    finally:
        del text_model
        torch.cuda.empty_cache()

    cos = float(np.dot(cached_vec, fresh_vec) /
                (np.linalg.norm(cached_vec) * np.linalg.norm(fresh_vec)))
    return {"backbone": backbone, "text": text, "cosine": cos, "ok": cos >= 1.0 - 1e-6}


def main():
    for backbone in ("tessera", "ae"):
        ckpt_path = Path(CKPT_TEMPLATE.format(backbone=backbone))
        _, cfg = load_model(ckpt_path)
        _check_text_dim(backbone, cfg)
        print(f"[{backbone}] text_dim-Check OK: ckpt['cfg']['text_dim']={cfg['text_dim']} "
              f"== MRL_DIM={MRL_DIM} ({MODEL_NAME})")

        r = verify_against_cache(backbone)
        print(f"[{backbone}] Cache-Verifikation: cosine={r['cosine']:.8f} ok={r['ok']} "
              f"text={r['text'][:80]!r}")


if __name__ == "__main__":
    main()
