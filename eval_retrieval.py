"""
eval_retrieval.py

GERUEST - noch nicht ausfuehrbar: braucht ein trainiertes Modell (train.py-Checkpoint) und
Text-Embeddings (compute_text_embeddings.py), beides noch nicht vorhanden. Metriken
(CLAUDE.md): Recall@k, mAP@k, Zero-Shot-F1 - inkl. deutscher Queries (Niedersachsen-Region).
"""
import numpy as np
from sklearn.metrics import f1_score

import config as cfg

# TODO: liegt noch nicht vor - Checkpoint aus train.py
CHECKPOINT_PATH = cfg.ROOT / "checkpoints" / "model.pt"


def load_model(path=CHECKPOINT_PATH):
    """TODO: hier trainiertes TwoTower-Modell laden (siehe model.py). Noch kein Checkpoint
    vorhanden, train.py laeuft noch nicht gegen echte Daten."""
    raise NotImplementedError(f"Kein trainiertes Modell unter {path}.")


def recall_at_k(similarity: np.ndarray, ground_truth_idx: np.ndarray, k: int) -> float:
    """similarity: (n_queries, n_candidates). ground_truth_idx[i] = Index des korrekten
    Treffers fuer Query i."""
    topk = np.argsort(-similarity, axis=1)[:, :k]
    hits = (topk == ground_truth_idx[:, None]).any(axis=1)
    return float(hits.mean())


def map_at_k(similarity: np.ndarray, ground_truth_idx: np.ndarray, k: int) -> float:
    """mAP@k mit genau einem relevanten Treffer je Query (AP = 1/rank falls im Top-k, sonst 0)."""
    topk = np.argsort(-similarity, axis=1)[:, :k]
    ap = np.zeros(len(similarity))
    for i in range(len(similarity)):
        matches = np.where(topk[i] == ground_truth_idx[i])[0]
        if len(matches) > 0:
            ap[i] = 1.0 / (matches[0] + 1)
    return float(ap.mean())


def zero_shot_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro"))


def main():
    # TODO: NICHT ausfuehren - siehe load_model(). Retrieval-Queries (inkl. deutsche Varianten
    # fuer die Niedersachsen-Region) sind ausserdem noch zu definieren.
    load_model()
    raise NotImplementedError("Retrieval-Queries (inkl. deutsche Varianten) noch zu definieren.")


if __name__ == "__main__":
    main()
