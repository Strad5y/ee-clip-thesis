"""
compute_poi_text_embeddings.py

F2: Encoding-Skript fuer die POI-Texte - getrennt von compute_text_embeddings.py, das
weiterhin nur die HILUCS-Zero-Shot-Prompts abdeckt (siehe dortiger Docstring/Teil-B-Audit).
Liest poi_text.parquet (build_poi_text.py, ein Text pro poi_id), bettet die EINDEUTIGEN
Texte einmal mit Qwen3-Embedding-4B ein (truncate_dim=1024, Renormalisierung ueber
normalize_embeddings=True NACH der MRL-Truncation) und schreibt ein poi_id-keyed
Cache-Parquet in dem Format, das dataset.py::PairDataset erwartet (poi_id, embedding).

load_text_encoder() (aus compute_text_embeddings.py) laedt das echte Modell auf CUDA -
lauffaehig seit torch 2.13.0+cu130 + sentence-transformers auf dem DGX Spark.

Fortschrittsausgabe/Logging wie bei den Join-Laeufen (chunked_join_runner.py): Batches statt
Bloecke, aber dieselbe stdout->Logdatei-Spiegelung und derselbe Heartbeat-Mechanismus
(mind. alle 60s ein Lebenszeichen, unabhaengig von der tatsaechlichen Batch-Laufzeit).
"""
import sys
import threading
import time

import numpy as np
import pandas as pd

import config as cfg
from chunked_join_runner import HEARTBEAT_INTERVAL_S, LOG_DIR, _Tee
from compute_text_embeddings import MRL_DIM, load_text_encoder

POI_TEXT_IN = cfg.DATA / "interim" / "poi_text.parquet"
OUT = cfg.DATA / "processed" / "text_embeddings.parquet"
BATCH_SIZE = 500


def embed_unique_texts(texts: pd.Series, model, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Kodiert nur die EINDEUTIGEN Texte (viele POIs teilen sich denselben Text, z.B. bei
    identischen Tags ohne Namen) - vermeidet redundantes Encoding. In Batches mit
    Fortschrittsausgabe: pro Batch UND zusaetzlich per Heartbeat-Thread (mind. alle 60s),
    damit ein langsamer Batch nicht die Lebenszeichen-Garantie verletzt."""
    unique_texts = texts.drop_duplicates().reset_index(drop=True).tolist()
    n = len(unique_texts)
    n_batches = -(-n // batch_size)

    state = {"done": 0}
    lock = threading.Lock()
    t_start = time.time()
    stop_event = threading.Event()

    def _progress_line(done: int) -> str:
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (n - done) / rate if rate > 0 else float("nan")
        return (f"{done}/{n} Texte kodiert, verstrichen={elapsed / 60:.1f} min, "
                f"geschaetzte Restzeit={remaining / 60:.1f} min")

    def _heartbeat():
        while not stop_event.wait(HEARTBEAT_INTERVAL_S):
            with lock:
                done = state["done"]
            print(f"[Fortschritt] {_progress_line(done)}")

    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()

    all_vectors = []
    for i in range(0, n, batch_size):
        batch = unique_texts[i:i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True)
        all_vectors.append(np.asarray(vecs, dtype=np.float32))

        done = min(i + batch_size, n)
        with lock:
            state["done"] = done
        print(f"Batch {i // batch_size + 1}/{n_batches}: {_progress_line(done)}")

    stop_event.set()

    vectors = np.concatenate(all_vectors, axis=0)
    assert vectors.shape[1] == MRL_DIM, f"Erwartet MRL_DIM={MRL_DIM}, bekommen {vectors.shape[1]}"
    return pd.DataFrame({
        "text": unique_texts,
        "embedding": list(vectors),
    })


def build_poi_text_embeddings(poi_text: pd.DataFrame, text_embeddings: pd.DataFrame) -> pd.DataFrame:
    """Broadcast der eindeutigen Text-Embeddings zurueck auf poi_id - Output-Format ist
    (poi_id, embedding), damit dataset.py::PairDataset per poi_id mergen kann."""
    merged = poi_text[["poi_id", "text"]].merge(text_embeddings, on="text", how="left")
    n_missing = int(merged["embedding"].isna().sum())
    if n_missing > 0:
        raise ValueError(f"{n_missing} poi_id ohne Text-Embedding nach dem Merge - "
                          "sollte bei vollstaendigem unique-Text-Encoding nicht vorkommen.")
    return merged[["poi_id", "embedding"]]


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"poi_text_embed_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    print("=== compute_poi_text_embeddings.py gestartet ===")
    print(f"Log-Datei: {log_path}")

    poi_text = pd.read_parquet(POI_TEXT_IN, columns=["poi_id", "text"])
    n_unique = poi_text["text"].nunique()
    print(f"{len(poi_text)} POIs, {n_unique} eindeutige Texte "
          f"({100 * n_unique / len(poi_text):.1f}%)")

    t_load = time.time()
    model = load_text_encoder()
    print(f"Modell geladen ({MRL_DIM}-dim, CUDA) in {time.time() - t_load:.1f}s")

    t_encode = time.time()
    text_embeddings = embed_unique_texts(poi_text["text"], model)
    print(f"Encoding fertig in {(time.time() - t_encode) / 60:.1f} min")

    out = build_poi_text_embeddings(poi_text, text_embeddings)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"Gespeichert: {OUT} ({len(out)} poi_id, dim={MRL_DIM})")
    print("=== compute_poi_text_embeddings.py fertig ===")


def _smoke_test():
    """Kein echter Encoder verfuegbar hier gebraucht - ein Dummy-Modell prueft nur die
    Dedup-/Batch-/Broadcast-Mechanik (embed_unique_texts/build_poi_text_embeddings), kein
    echtes Encoding."""
    class DummyModel:
        def encode(self, texts, normalize_embeddings=True):
            rng = np.random.default_rng(0)
            vecs = rng.standard_normal((len(texts), MRL_DIM)).astype(np.float32)
            if normalize_embeddings:
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    poi_text = pd.DataFrame({
        "poi_id": [f"node/{i}" for i in range(6)],
        # POI 0 und 2 teilen sich denselben Text -> muss nur einmal kodiert werden
        "text": ["a bench.", "a cafe.", "a bench.", "a park.", "a museum.", "a park."],
    })

    model = DummyModel()
    # batch_size=2 erzwingt mehrere Batches trotz nur 4 eindeutiger Texte - prueft die
    # Batch-Grenzen-Logik, nicht nur den Trivialfall "alles in einem Batch"
    text_embeddings = embed_unique_texts(poi_text["text"], model, batch_size=2)
    assert len(text_embeddings) == poi_text["text"].nunique() == 4

    out = build_poi_text_embeddings(poi_text, text_embeddings)
    assert len(out) == len(poi_text)
    assert set(out["poi_id"]) == set(poi_text["poi_id"])

    emb_poi0 = out.loc[out["poi_id"] == "node/0", "embedding"].iloc[0]
    emb_poi2 = out.loc[out["poi_id"] == "node/2", "embedding"].iloc[0]
    assert np.allclose(emb_poi0, emb_poi2), "Gleicher Text muss auf denselben Embedding-Vektor gemappt werden"

    dims = out["embedding"].apply(len)
    assert (dims == MRL_DIM).all()

    print(f"Smoke-Test OK: {len(poi_text)} POIs -> {len(text_embeddings)} eindeutige Texte kodiert "
          f"(mehrere Batches), Broadcast auf poi_id korrekt, dim={MRL_DIM}.")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        main()
