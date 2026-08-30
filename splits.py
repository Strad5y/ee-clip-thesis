"""
splits.py

Liest den raeumlich geblockten POI-Split (siehe build_splits.py, data/splits/*.csv) und
mappt ihn auf Datensatz-Zeilenindizes ueber poi_id. Ersetzt den frueheren zufaelligen
Zeilensplit in train.py: Gitterzellen (nicht Zeilen oder POIs) wurden vorab zufaellig auf
train/val/test verteilt, POIs folgen ihrer Zelle - garantiert per Konstruktion, dass alle
Pixel-Zeilen eines poi_id im selben Split landen (kein Leakage durch Zeilen-Split mehr).
"""
import numpy as np
import pandas as pd

VALID_SPLITS = ("train", "val", "test")


def load_split_assignment(splits_path) -> pd.Series:
    df = pd.read_csv(splits_path)
    return pd.Series(df["split"].values, index=df["poi_id"].values)


def split_indices(poi_ids: np.ndarray, split_assignment: pd.Series) -> dict:
    """poi_ids: ds.poi_id-Array (eine Zeile pro Pixel-Paar). Gibt {'train': idx, 'val': idx,
    'test': idx} zurueck (np.ndarray Zeilenindizes in poi_ids). Prueft explizit, dass die
    poi_id-Mengen der drei Splits paarweise disjunkt sind (Kernaussage von F1) - bricht mit
    AssertionError ab, statt eine Leakage still durchzulassen."""
    assigned = split_assignment.reindex(poi_ids).to_numpy()
    n_missing = int(pd.isna(assigned).sum())
    if n_missing > 0:
        print(f"  Warnung: {n_missing} Zeilen ohne Split-Zuordnung "
              f"(poi_id nicht im Split-Artefakt) - ausgeschlossen.")

    idx = {s: np.where(assigned == s)[0] for s in VALID_SPLITS}

    train_pois = set(poi_ids[idx["train"]])
    val_pois = set(poi_ids[idx["val"]])
    test_pois = set(poi_ids[idx["test"]])
    overlap_tv = train_pois & val_pois
    overlap_tt = train_pois & test_pois
    overlap_vt = val_pois & test_pois
    assert not overlap_tv, f"Split-Leakage: {len(overlap_tv)} poi_id sowohl in Train als auch Val"
    assert not overlap_tt, f"Split-Leakage: {len(overlap_tt)} poi_id sowohl in Train als auch Test"
    assert not overlap_vt, f"Split-Leakage: {len(overlap_vt)} poi_id sowohl in Val als auch Test"

    return idx


def _smoke_test():
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(0)
    # 10 POIs, je 3 Pixel-Zeilen (typischer pixelweiser Fall) -> 30 Zeilen insgesamt
    poi_ids = [f"way/{i}" for i in range(10)]
    rows_poi_ids = np.repeat(poi_ids, 3)

    splits_df = pd.DataFrame({
        "poi_id": poi_ids,
        "cell_row": rng.integers(0, 5, 10),
        "cell_col": rng.integers(0, 5, 10),
        "split": rng.choice(["train", "val", "test"], size=10, p=[0.6, 0.2, 0.2]),
    })

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splits_dummy.csv"
        splits_df.to_csv(path, index=False)

        assignment = load_split_assignment(path)
        idx = split_indices(rows_poi_ids, assignment)

    assert sum(len(v) for v in idx.values()) == len(rows_poi_ids)
    train_pois = set(rows_poi_ids[idx["train"]])
    val_pois = set(rows_poi_ids[idx["val"]])
    assert not (train_pois & val_pois)

    print(f"Smoke-Test OK: {len(idx['train'])} train / {len(idx['val'])} val / {len(idx['test'])} test Zeilen, "
          f"poi_id-Schnittmenge Train/Val leer.")


if __name__ == "__main__":
    _smoke_test()
