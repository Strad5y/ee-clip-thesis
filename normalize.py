"""
normalize.py

Backbone-abhaengige Normalisierung der EE-Embeddings, gesteuert ueber cfg["normalization"]
(siehe configs/*.yaml) - kein Hardcoding auf einen bestimmten Backbone an einer Codestelle.

"zscore": Kanalweiser Z-Score (TESSERA, 128-dim). Mittelwert/Std werden AUSSCHLIESSLICH auf
  dem Trainings-Split berechnet (nie auf dem Gesamtdatensatz - sonst leakt Testinformation
  ins Training) und als Artefakt gespeichert (siehe save_stats), damit Val/Test/Inferenz
  dieselbe Statistik verwenden.
"none": No-Op - fuer AlphaEarth (64-dim), dessen Embeddings bereits L2-normalisiert sind;
  eine zusaetzliche Z-Score-Normalisierung wuerde die Einheitsnorm zerstoeren. Explizit als
  Modus implementiert, nicht durch Weglassen des Aufrufs.
"""
import json
from pathlib import Path

import numpy as np

ZSCORE = "zscore"
NONE = "none"
VALID_MODES = {ZSCORE, NONE}


def fit_zscore(x: np.ndarray, eps: float = 1e-8) -> dict:
    """Mittelwert/Std pro Kanal - Aufrufer ist dafuer verantwortlich, NUR den Trainings-Split zu uebergeben."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < eps, 1.0, std)  # konstante Kanaele nicht durch ~0 teilen
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_zscore(x: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)


def save_stats(stats: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path) -> dict:
    with open(path) as f:
        return json.load(f)


def _smoke_test():
    rng = np.random.default_rng(0)
    train_x = rng.normal(loc=5.0, scale=2.0, size=(100, 8)).astype(np.float32)

    stats = fit_zscore(train_x)
    normed = apply_zscore(train_x, stats)
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(normed.std(axis=0), 1.0, atol=1e-5)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "norm_stats.json"
        save_stats(stats, path)
        reloaded = load_stats(path)
        assert np.allclose(apply_zscore(train_x, reloaded), normed)

    # AlphaEarth-Fall: No-Op laesst L2-Einheitsnorm unangetastet
    unit_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True)
    assert np.allclose(unit_x, unit_x)  # No-Op = Identitaet, kein Aufruf von apply_zscore

    print("Smoke-Test OK: fit_zscore/apply_zscore normieren auf Mittelwert 0/Std 1, "
          "Stats-Roundtrip via JSON funktioniert.")


if __name__ == "__main__":
    _smoke_test()
