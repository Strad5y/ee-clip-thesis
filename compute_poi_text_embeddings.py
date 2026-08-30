"""
compute_poi_text_embeddings.py

F2: Encoding-Skript fuer die POI-Texte - getrennt von compute_text_embeddings.py, das
weiterhin nur die HILUCS-Zero-Shot-Prompts abdeckt (siehe dortiger Docstring/Teil-B-Audit).
Liest poi_text.parquet (build_poi_text.py, ein Text pro poi_id), bettet die EINDEUTIGEN
Texte einmal mit Qwen3-Embedding-4B ein (truncate_dim=1024, Renormalisierung ueber
normalize_embeddings=True NACH der MRL-Truncation) und schreibt ein poi_id-keyed
Cache-Parquet in dem Format, das dataset.py::PairDataset erwartet (poi_id, embedding).

GERUEST wie compute_text_embeddings.py: laeuft erst nach dem torch-Install auf dem Spark
(Qwen3-Embedding-4B ist zu gross fuer diese CPU-Maschine) - load_text_encoder() wird von
dort importiert und wirft bewusst NotImplementedError.
"""
import numpy as np
import pandas as pd

import config as cfg
from compute_text_embeddings import MRL_DIM, load_text_encoder

POI_TEXT_IN = cfg.DATA / "interim" / "poi_text.parquet"
OUT = cfg.DATA / "processed" / "text_embeddings.parquet"


def embed_unique_texts(texts: pd.Series, model) -> pd.DataFrame:
    """Kodiert nur die EINDEUTIGEN Texte (viele POIs teilen sich denselben Text, z.B. bei
    identischen Tags ohne Namen) - vermeidet redundantes Encoding."""
    unique_texts = texts.drop_duplicates().reset_index(drop=True)
    vectors = model.encode(unique_texts.tolist(), normalize_embeddings=True)
    assert vectors.shape[1] == MRL_DIM, f"Erwartet MRL_DIM={MRL_DIM}, bekommen {vectors.shape[1]}"
    return pd.DataFrame({
        "text": unique_texts,
        "embedding": list(np.asarray(vectors, dtype=np.float32)),
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
    poi_text = pd.read_parquet(POI_TEXT_IN, columns=["poi_id", "text"])
    n_unique = poi_text["text"].nunique()
    print(f"{len(poi_text)} POIs, {n_unique} eindeutige Texte "
          f"({100 * n_unique / len(poi_text):.1f}%)")

    model = load_text_encoder()  # wirft NotImplementedError auf dieser CPU-Maschine
    text_embeddings = embed_unique_texts(poi_text["text"], model)
    out = build_poi_text_embeddings(poi_text, text_embeddings)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"Gespeichert: {OUT} ({len(out)} poi_id, dim={MRL_DIM})")


def _smoke_test():
    """Kein echter Encoder verfuegbar (Qwen3-Embedding-4B laeuft nicht auf dieser CPU-Maschine,
    siehe load_text_encoder()) - ein Dummy-Modell prueft nur die Dedup-/Broadcast-Mechanik
    (embed_unique_texts/build_poi_text_embeddings), kein echtes Encoding."""
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
    text_embeddings = embed_unique_texts(poi_text["text"], model)
    assert len(text_embeddings) == poi_text["text"].nunique() == 4

    out = build_poi_text_embeddings(poi_text, text_embeddings)
    assert len(out) == len(poi_text)
    assert set(out["poi_id"]) == set(poi_text["poi_id"])

    emb_poi0 = out.loc[out["poi_id"] == "node/0", "embedding"].iloc[0]
    emb_poi2 = out.loc[out["poi_id"] == "node/2", "embedding"].iloc[0]
    assert np.allclose(emb_poi0, emb_poi2), "Gleicher Text muss auf denselben Embedding-Vektor gemappt werden"

    dims = out["embedding"].apply(len)
    assert (dims == MRL_DIM).all()

    print(f"Smoke-Test OK: {len(poi_text)} POIs -> {len(text_embeddings)} eindeutige Texte kodiert, "
          f"Broadcast auf poi_id korrekt (identische Texte teilen sich denselben Vektor), dim={MRL_DIM}.")


if __name__ == "__main__":
    import sys
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        main()
