"""
compute_text_embeddings.py

load_text_encoder() ist auf dem DGX Spark (CUDA verfuegbar) lauffaehig und wird von
compute_poi_text_embeddings.py wiederverwendet. main() hier ist weiterhin GERUEST - nicht
wegen der Maschine, sondern weil PROMPT_TEMPLATES noch Platzhalter sind (siehe TODO unten).

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
    """Laedt Qwen3-Embedding-4B mit MRL-Truncation auf MRL_DIM=1024. Braucht CUDA - auf einer
    reinen CPU-Maschine waere die Laufzeit fuer ein 4B-Modell unpraktikabel, also lieber laut
    scheitern als still auf CPU zurueckfallen und eine falsche Laufzeitschaetzung liefern."""
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{MODEL_NAME} braucht CUDA fuer eine praktikable Laufzeit - "
            "torch.cuda.is_available() ist False auf dieser Maschine."
        )
    return SentenceTransformer(MODEL_NAME, truncate_dim=MRL_DIM, device="cuda")


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
