"""
eval_socioeconomic.py

GERUEST - noch nicht ausfuehrbar: sozio-oekonomische Ground-Truth (UK: Census/IMD,
DE: Zensus 2022) liegt noch nicht im Repo, keine erfundenen Daten. Vergleicht die Verteilung
der gelernten Embeddings/Cluster mit sozio-oekonomischen Referenzverteilungen per
KL-Divergenz (CLAUDE.md).
"""
import numpy as np
from scipy.stats import entropy

import config as cfg

# TODO: liegt noch nicht im Repo - London: Census/IMD, Niedersachsen: Zensus 2022
SOCIOECONOMIC_GT_PATH = cfg.DATA / "raw" / "socioeconomic"


def load_ground_truth(path=SOCIOECONOMIC_GT_PATH):
    """TODO: hier Ground-Truth einlesen (Census/IMD fuer London, Zensus 2022 fuer
    Niedersachsen). Noch keine Daten vorhanden - NICHT mit erfundenen Werten fuellen."""
    raise NotImplementedError(f"Sozio-oekonomische GT fehlt noch unter {path}.")


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(P || Q) zwischen zwei diskreten Verteilungen (z.B. Modell-Cluster-Anteile vs.
    IMD-Dezil-Anteile)."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(entropy(p, q))


def evaluate(model_distribution: np.ndarray, gt_distribution: np.ndarray) -> dict:
    return {"kl_divergence": kl_divergence(model_distribution, gt_distribution)}


def main():
    # TODO: NICHT ausfuehren - siehe load_ground_truth(). Modell-Verteilung braucht zusaetzlich
    # ein trainiertes Modell (train.py) sowie einen Clustering-/Aggregationsschritt, der hier
    # noch fehlt.
    load_ground_truth()
    raise NotImplementedError("Modell-Verteilung (aus trainiertem Modell) noch zu erzeugen.")


if __name__ == "__main__":
    main()
