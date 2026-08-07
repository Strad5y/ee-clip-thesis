"""
compute_text_embeddings.py

GERUEST - Code vollstaendig, aber bewusst NICHT auf dieser CPU-Maschine ausgefuehrt:
Qwen3-Embedding-4B wird hier NICHT geladen oder heruntergeladen (zu gross fuer CPU-only,
laeuft spaeter auf Spark).

Berechnet Text-Embeddings fuer alle Prompt-Templates je HILUCS-Klasse, MRL-Truncation
2560 -> 1024 (settled decision, CLAUDE.md). Modell FROZEN.
"""
import numpy as np
import pandas as pd

import config as cfg

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
MRL_DIM = 1024  # Truncation 2560 -> 1024 (settled decision)

# TODO: Prompt-Templates sind thesis-spezifisch und hier nur ein Platzhalter-Geruest - die
# tatsaechlichen Templates (inkl. deutscher Varianten fuer eval_retrieval.py, "Niedersachsen")
# muessen noch final festgelegt werden.
PROMPT_TEMPLATES = [
    "{feinlabel}",
    "Ein Ort der Kategorie {feinlabel}.",
]


def build_prompts(crosswalk: pd.DataFrame, templates=PROMPT_TEMPLATES) -> pd.DataFrame:
    """Pro (hilucs_code, template) ein Prompt-String - Grundlage fuer die Embedding-Berechnung."""
    rows = []
    for _, r in crosswalk.dropna(subset=["hilucs_code"]).iterrows():
        for t in templates:
            rows.append({
                "hilucs_code": r["hilucs_code"],
                "template": t,
                "prompt": t.format(feinlabel=r["feinlabel"]),
            })
    return pd.DataFrame(rows)


def load_text_encoder():
    """TODO: NICHT auf dieser Maschine ausfuehren (CPU-only, kein Spark hier).

    Vorgesehener Ladepfad auf Spark/GPU:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME, truncate_dim=MRL_DIM)
    """
    raise NotImplementedError(
        f"{MODEL_NAME} wird bewusst nicht auf dieser CPU-Maschine geladen/heruntergeladen - "
        "laeuft auf Spark. Siehe Docstring von load_text_encoder() fuer den vorgesehenen Ladepfad."
    )


def compute_embeddings(prompts: pd.DataFrame, model) -> pd.DataFrame:
    vectors = model.encode(prompts["prompt"].tolist(), normalize_embeddings=True)
    assert vectors.shape[1] == MRL_DIM, f"Erwartet MRL_DIM={MRL_DIM}, bekommen {vectors.shape[1]}"
    prompts = prompts.copy()
    prompts["embedding"] = list(np.asarray(vectors, dtype=np.float32))
    return prompts


def main():
    # TODO: NICHT ausfuehren - siehe load_text_encoder().
    raise NotImplementedError(
        "compute_text_embeddings.py ist ein Geruest - laeuft auf Spark, nicht hier."
    )


if __name__ == "__main__":
    main()
