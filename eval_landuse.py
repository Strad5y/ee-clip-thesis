"""
eval_landuse.py

GERUEST - noch nicht ausfuehrbar: Urban-Atlas-Ground-Truth liegt noch nicht im Repo, keine
erfundenen Daten. Linear Probe auf den gefrorenen EE-Embeddings gegen Urban-Atlas-Landuse-
Klassen. Metriken (CLAUDE.md): Macro-F1, Overall Accuracy (OA), Label-Efficiency-Kurve
(Accuracy vs. Trainingsanteil).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import config as cfg

# TODO: liegt noch nicht im Repo - Urban Atlas Vektor/Raster je Studienregion
URBAN_ATLAS_PATH = cfg.DATA / "raw" / "urban_atlas"


def load_ground_truth(path=URBAN_ATLAS_PATH):
    """TODO: hier Ground-Truth einlesen (Urban-Atlas-Landuse-Klasse je Pixel/POI, aligned mit
    den EE-Embeddings aus pairs_tessera.parquet/pairs_ae.parquet). Noch keine Daten vorhanden -
    NICHT mit erfundenen Werten fuellen."""
    raise NotImplementedError(f"Urban-Atlas-GT fehlt noch unter {path}.")


def linear_probe(embeddings: np.ndarray, labels: np.ndarray, train_frac: float = 1.0, seed: int = 0) -> dict:
    """Logistic-Regression-Probe auf gefrorenen Embeddings, mit Trainingsanteil fuer die
    Label-Efficiency-Kurve."""
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    idx = rng.permutation(n)
    n_train = max(1, int(n * train_frac))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    if len(test_idx) == 0:
        test_idx = train_idx

    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings[train_idx], labels[train_idx])
    pred = clf.predict(embeddings[test_idx])

    return {
        "train_frac": train_frac,
        "n_train": n_train,
        "oa": accuracy_score(labels[test_idx], pred),
        "macro_f1": f1_score(labels[test_idx], pred, average="macro"),
    }


def label_efficiency_curve(embeddings, labels, fracs=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0)) -> list:
    return [linear_probe(embeddings, labels, f) for f in fracs]


def main():
    # TODO: NICHT ausfuehren - siehe load_ground_truth().
    embeddings, labels = load_ground_truth()
    for point in label_efficiency_curve(embeddings, labels):
        print(point)


if __name__ == "__main__":
    main()
