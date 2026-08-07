"""
dataset.py

PyTorch Dataset fuer die Two-Tower-Trainingspaare: EE-Pixel-Embedding (aus spatial_join.py) x
Text-Embedding (aus compute_text_embeddings.py), gejoint ueber poi_id. coverage wird sowohl als
Sampling-Gewicht (WeightedRandomSampler) als auch als Loss-Gewicht (siehe losses.py) vorgesehen.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ID_COL = "poi_id"


class PairDataset(Dataset):
    def __init__(self, pairs_path, text_embeddings_path, id_col: str = ID_COL):
        pairs = pd.read_parquet(pairs_path)
        text = pd.read_parquet(text_embeddings_path)

        merged = pairs.merge(text, on=id_col, suffixes=("_ee", "_text"))
        if len(merged) == 0:
            raise ValueError(
                f"Kein Overlap zwischen {pairs_path} und {text_embeddings_path} ueber '{id_col}'."
            )

        self.ee = np.stack(merged["embedding_ee"].to_numpy()).astype(np.float32)
        self.text = np.stack(merged["embedding_text"].to_numpy()).astype(np.float32)
        self.coverage = merged["coverage"].to_numpy(dtype=np.float32)
        self.poi_id = merged[id_col].to_numpy()

    def __len__(self) -> int:
        return len(self.coverage)

    def __getitem__(self, idx):
        return {
            "ee_embedding": torch.from_numpy(self.ee[idx]),
            "text_embedding": torch.from_numpy(self.text[idx]),
            "coverage": torch.tensor(self.coverage[idx]),
            "poi_id": self.poi_id[idx],
        }


def make_coverage_sampler(dataset: PairDataset) -> WeightedRandomSampler:
    """coverage als Sampling-Gewicht: Paare mit hoher POI-Pixel-Ueberdeckung werden haeufiger gezogen."""
    weights = torch.from_numpy(dataset.coverage.copy())
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
        "embedding": list(rng.standard_normal((n, ee_dim)).astype(np.float32)),
    })
    text = pd.DataFrame({
        ID_COL: poi_ids,
        "embedding": list(rng.standard_normal((n, text_dim)).astype(np.float32)),
    })

    with tempfile.TemporaryDirectory() as tmp:
        pairs_path = Path(tmp) / "pairs_dummy.parquet"
        text_path = Path(tmp) / "text_dummy.parquet"
        pairs.to_parquet(pairs_path)
        text.to_parquet(text_path)

        ds = PairDataset(pairs_path, text_path)
        assert len(ds) == n

        sampler = make_coverage_sampler(ds)
        dl = DataLoader(ds, batch_size=4, sampler=sampler)
        batch = next(iter(dl))
        assert batch["ee_embedding"].shape == (4, ee_dim)
        assert batch["text_embedding"].shape == (4, text_dim)
        assert batch["coverage"].shape == (4,)

    print(f"Smoke-Test OK: PairDataset liefert Batches ee={tuple(batch['ee_embedding'].shape)}, "
          f"text={tuple(batch['text_embedding'].shape)}, coverage-Sampler funktioniert.")


if __name__ == "__main__":
    _smoke_test()
