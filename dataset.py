"""
dataset.py

PyTorch Dataset fuer die Two-Tower-Trainingspaare: EE-Pixel-Embedding (aus spatial_join.py) x
Text-Embedding (aus compute_text_embeddings.py), gejoint ueber poi_id. Sampling-Gewicht ist
coverage / n_pixels_poi (WeightedRandomSampler) - macht das Gewicht naeherungsweise unabhaengig
von der Polygongroesse. Der Loss (losses.py) ist ungewichtet.

poi_text_path (optional): liefert die tatsaechlichen POI-Texte (build_poi_text.py), um
text_id zu berechnen - eine Ganzzahl-Gruppe pro EINDEUTIGEM Text (pd.factorize, byte-
identischer Stringvergleich, kein Aehnlichkeitsmass). Zwei Zeilen teilen sich genau dann
denselben text_id-Wert, wenn ihr Text-String identisch ist - Grundlage fuer die pos_mask
in train.py (siehe dortiger Docstring: 54,8% der POIs teilen sich nur 46 Texte)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ID_COL = "poi_id"


class PairDataset(Dataset):
    def __init__(self, pairs_path, text_embeddings_path, poi_text_path=None, id_col: str = ID_COL):
        pairs = pd.read_parquet(pairs_path)
        text = pd.read_parquet(text_embeddings_path)

        merged = pairs.merge(text, on=id_col, suffixes=("_ee", "_text"))
        if len(merged) == 0:
            raise ValueError(
                f"Kein Overlap zwischen {pairs_path} und {text_embeddings_path} ueber '{id_col}'."
            )

        if poi_text_path is not None:
            poi_text = pd.read_parquet(poi_text_path, columns=[id_col, "text"])
            merged = merged.merge(poi_text, on=id_col, how="left")
            if merged["text"].isna().any():
                raise ValueError(
                    f"{int(merged['text'].isna().sum())} Zeilen ohne Text-String nach dem "
                    f"Merge mit {poi_text_path} - poi_text_path sollte dieselbe POI-Menge "
                    f"wie {text_embeddings_path} abdecken."
                )
            # pd.factorize: byte-identischer Stringvergleich, keine Aehnlichkeit/Schwellwert -
            # gleicher text_id-Wert <=> exakt derselbe Text-String
            self.text_id = pd.factorize(merged["text"].to_numpy())[0].astype(np.int64)
        else:
            self.text_id = None

        self.ee = np.stack(merged["embedding_ee"].to_numpy()).astype(np.float32)
        self.text = np.stack(merged["embedding_text"].to_numpy()).astype(np.float32)
        self.coverage = merged["coverage"].to_numpy(dtype=np.float32)
        self.n_pixels_poi = merged["n_pixels_poi"].to_numpy(dtype=np.float32)
        self.sample_weight = self.coverage / self.n_pixels_poi
        self.poi_id = merged[id_col].to_numpy()

    def __len__(self) -> int:
        return len(self.coverage)

    def __getitem__(self, idx):
        item = {
            "ee_embedding": torch.from_numpy(self.ee[idx]),
            "text_embedding": torch.from_numpy(self.text[idx]),
            "coverage": torch.tensor(self.coverage[idx]),
            "poi_id": self.poi_id[idx],
        }
        if self.text_id is not None:
            item["text_id"] = torch.tensor(self.text_id[idx])
        return item


def make_coverage_sampler(dataset: PairDataset) -> WeightedRandomSampler:
    """coverage / n_pixels_poi als Sampling-Gewicht: haelt den Gesamtzug pro POI naeherungsweise
    unabhaengig von dessen Pixelzahl (Polygongroesse), statt grosse Polygone zu ueberrepraesentieren."""
    weights = torch.from_numpy(dataset.sample_weight.copy())
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)


def _smoke_test():
    """Kein echtes Textembedding vorhanden (compute_text_embeddings.py ist nur ein Geruest) -
    Dummy-Parquets in einem temporaeren Verzeichnis pruefen nur den Join/DataLoader-Mechanismus."""
    rng = np.random.default_rng(0)
    n, ee_dim, text_dim = 20, 128, 1024
    poi_ids = [f"node/{i}" for i in range(n)]

    pairs = pd.DataFrame({
        ID_COL: poi_ids,
        "row": rng.integers(0, 100, n),
        "col": rng.integers(0, 100, n),
        "coverage": rng.random(n).astype(np.float32),
        "n_pixels_poi": rng.integers(1, 5, n).astype(np.float32),
        "embedding": list(rng.standard_normal((n, ee_dim)).astype(np.float32)),
    })
    text = pd.DataFrame({
        ID_COL: poi_ids,
        "embedding": list(rng.standard_normal((n, text_dim)).astype(np.float32)),
    })
    # poi_id 0 und 5 teilen sich denselben Text-String -> muessen denselben text_id bekommen
    texts = [f"text {i}" for i in range(n)]
    texts[5] = texts[0]
    poi_text = pd.DataFrame({ID_COL: poi_ids, "text": texts})

    with tempfile.TemporaryDirectory() as tmp:
        pairs_path = Path(tmp) / "pairs_dummy.parquet"
        text_path = Path(tmp) / "text_dummy.parquet"
        poi_text_path = Path(tmp) / "poi_text_dummy.parquet"
        pairs.to_parquet(pairs_path)
        text.to_parquet(text_path)
        poi_text.to_parquet(poi_text_path)

        ds = PairDataset(pairs_path, text_path, poi_text_path=poi_text_path)
        assert len(ds) == n
        assert ds.text_id[0] == ds.text_id[5], "identischer Text-String muss denselben text_id bekommen"
        assert ds.text_id[1] != ds.text_id[2], "unterschiedliche Texte muessen unterschiedliche text_id bekommen"

        sampler = make_coverage_sampler(ds)
        dl = DataLoader(ds, batch_size=4, sampler=sampler)
        batch = next(iter(dl))
        assert batch["ee_embedding"].shape == (4, ee_dim)
        assert batch["text_embedding"].shape == (4, text_dim)
        assert batch["coverage"].shape == (4,)
        assert batch["text_id"].shape == (4,)

        # Ohne poi_text_path bleibt das Verhalten wie zuvor (text_id fehlt im Batch)
        ds_no_text = PairDataset(pairs_path, text_path)
        assert ds_no_text.text_id is None
        assert "text_id" not in ds_no_text[0]

    print(f"Smoke-Test OK: PairDataset liefert Batches ee={tuple(batch['ee_embedding'].shape)}, "
          f"text={tuple(batch['text_embedding'].shape)}, coverage-Sampler funktioniert, "
          f"text_id markiert identische Texte korrekt (mit poi_text_path) bzw. fehlt sauber "
          f"(ohne poi_text_path).")


if __name__ == "__main__":
    _smoke_test()
