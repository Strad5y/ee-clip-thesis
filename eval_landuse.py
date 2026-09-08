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


def linear_probe(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray, C: float = 1.0, max_iter: int = 200,
                  return_fit: bool = False) -> dict:
    """Logistic-Regression-Probe auf gefrorenen Embeddings, raeumlich geblockter Split
    (train/val/test kommen vom Aufrufer, siehe build_landuse_pairs.py - NICHT mehr der
    zufaellige train_frac-Split von vorher, der bei pixelweisen Paaren voll leakage-behaftet
    waere, genau wie der urspruenglich zufaellige Zeilensplit im Training). Multinomial ergibt
    sich mit dem Default-Solver (lbfgs) automatisch ab sklearn >=0.22 fuer mehr als 2 Klassen -
    kein expliziter multi_class-Parameter mehr (in aktuellen sklearn-Versionen entfernt).
    Trainiert NUR auf X_train/y_train; C wird vom Aufrufer ueber val_macro_f1 gewaehlt.
    return_fit=True liefert zusaetzlich den Classifier + rohe Vorhersagen (fuer Bootstrap in
    run_landuse_probe.py, ohne dort neu fitten zu muessen)."""
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X_train, y_train)
    pred_val = clf.predict(X_val)
    pred_test = clf.predict(X_test)

    result = {
        "C": C,
        "val_oa": accuracy_score(y_val, pred_val),
        "val_macro_f1": f1_score(y_val, pred_val, average="macro"),
        "test_oa": accuracy_score(y_test, pred_test),
        "test_macro_f1": f1_score(y_test, pred_test, average="macro"),
    }
    if return_fit:
        result["clf"] = clf
        result["pred_val"] = pred_val
        result["pred_test"] = pred_test
    return result


def main():
    # TODO: NICHT ausfuehren - siehe load_ground_truth(). Der eigentliche Phase-3-Ablauf
    # (sechs Konfigurationen, raeumlich geblockter Split, C-Suche auf Val) steht in
    # run_landuse_probe.py, das linear_probe() von hier importiert.
    embeddings, labels = load_ground_truth()
    print(linear_probe(embeddings, labels, embeddings, labels, embeddings, labels))


if __name__ == "__main__":
    main()
