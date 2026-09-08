"""
dataset.py

Phase 7 (CLAUDE_CODE_TASKS_phase4_7.md): PyTorch Dataset fuer die Two-Tower-Trainingspaare auf
PIXEL-Ebene (Phase 5, aggregate_pixel_pairs.py) - EE-Pixel-Embedding x Text-Embedding, gejoint
ueber 'pixel_text' (nicht mehr poi_id: ein Pixel ist jetzt bereits die kleinste Einheit, kein
Join mehr ueber mehrere Tag-Zeilen noetig).

Sampling-Gewicht ist NICHT mehr coverage/n_pixels_poi (das war die alte poi_id-Pairs-Logik,
die grosse Polygone mit vielen Pixel-Zeilen pro POI kompensieren musste). Auf Pixel-Ebene gibt
es diese Verzerrung strukturell nicht mehr - jedes Pixel ist bereits genau eine Zeile,
unabhaengig von der Flaeche des/der beitragenden POI(s). weight=1.0 durchgehend (Phase-5-
Vorgabe), coverage/n_pixels_poi bleiben nur als Diagnosespalten erhalten, gehen in kein
Gewicht ein.

split kommt direkt aus der Pixel-Datei (Phase-5-Nachtrag, build_splits_pixels.py hat die
Split-Zuordnung schon eingetragen) - kein separater CSV-Merge ueber poi_id mehr noetig
(splits.py/load_split_assignment sind fuer diesen Pfad nicht mehr im Einsatz).

text_id (fuer die pos_mask in train.py) kommt jetzt direkt aus pd.factorize auf 'pixel_text' -
kein separater poi_text_path-Merge mehr noetig, der String steht schon in der Pixel-Datei.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from build_poi_text import FRAME_NOUN, KEY_FRAME

TEXT_COL = "pixel_text"

# Reverse-Lookup: Frame-Text -> Key, fuer extract_primary_key(). Laengste zuerst, damit ein
# Praefix-Frame (z.B. falls zwei Frames denselben Anfang teilen wuerden) nicht vorzeitig matcht.
_KEY_FRAME_BY_LEN = sorted(KEY_FRAME.items(), key=lambda kv: -len(kv[1]))
_FRAME_NOUN_BY_LEN = sorted(FRAME_NOUN.items(), key=lambda kv: -len(kv[1]))


def extract_primary_key(text: str) -> str | None:
    """Rekonstruiert den primaeren OSM-Key (den der Top-Coverage-POI, siehe
    aggregate_pixel_pairs.py-Sortierung) direkt aus dem TEMPLATE-generierten pixel_text -
    kein zusaetzlicher Join gegen die Rohdaten noetig, weil build_poi_text.py/
    aggregate_pixel_pairs.py den Text deterministisch aus genau diesen Frame-Vokabularen
    zusammensetzen. Heuristisch (Substring-Suche), aber zuverlaessig GENAU WEIL die Texte
    Template-Ausgaben sind, keine freie Sprache."""
    t = text
    if t.startswith("A location containing "):
        t = t[len("A location containing "):]
    t = t.split("; ")[0]           # nur der Top-Coverage-POI-Teil (vor dem ersten Trenner)
    if " — " in t:
        t = t.split(" — ", 1)[1]   # Namensprefix abtrennen, falls vorhanden
    t_lower = t.lower()

    for key, frame in _KEY_FRAME_BY_LEN:                # Single-Tag-Form "{frame}: {value}."
        if t_lower.startswith(frame.lower() + ":"):
            return key

    best_key, best_pos = None, None                     # Multi-Tag-Form "a {value} {noun} and ..."
    for key, noun in _FRAME_NOUN_BY_LEN:
        pos = t_lower.find(noun.lower())
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_key, best_pos = key, pos
    return best_key


class PairDataset(Dataset):
    def __init__(self, pixels_path, text_embeddings_path):
        pixels = pd.read_parquet(pixels_path)
        text = pd.read_parquet(text_embeddings_path)

        # pixel_text_embeddings.parquet (compute_pixel_text_embeddings.py) benutzt die Spalte
        # 'text', die Pixel-Datei 'pixel_text' - unterschiedliche Namen, gleicher Inhalt.
        merged = pixels.merge(text, left_on=TEXT_COL, right_on="text", how="left",
                               suffixes=("_ee", "_text"))
        n_missing = int(merged["embedding_text"].isna().sum())
        if n_missing > 0:
            raise ValueError(
                f"{n_missing} Pixel ohne Text-Embedding nach dem Merge - {text_embeddings_path} "
                f"sollte alle '{TEXT_COL}'-Werte aus {pixels_path} abdecken."
            )

        # pd.factorize: byte-identischer Stringvergleich auf pixel_text, keine Aehnlichkeit -
        # gleicher text_id-Wert <=> exakt derselbe Pixel-Text-String (analog zur alten poi_id-
        # Variante, nur jetzt direkt auf der bereits vorhandenen Textspalte).
        self.text_id = pd.factorize(merged[TEXT_COL].to_numpy())[0].astype(np.int64)

        self.ee = np.stack(merged["embedding_ee"].to_numpy()).astype(np.float32)
        self.text = np.stack(merged["embedding_text"].to_numpy()).astype(np.float32)
        self.coverage = merged["coverage"].to_numpy(dtype=np.float32)          # Diagnose, kein Gewicht
        self.n_pixels_poi = merged["n_pixels_poi"].to_numpy(dtype=np.float32)  # Diagnose, kein Gewicht
        self.sample_weight = merged["weight"].to_numpy(dtype=np.float32)       # =1.0 durchgehend
        self.split = merged["split"].to_numpy()
        self.n_pois = merged["n_pois"].to_numpy(dtype=np.int64)
        self.pixel_key = (merged["tile"].astype(str) + "/" + merged["row"].astype(str) +
                           "/" + merged["col"].astype(str)).to_numpy()

        # primary_key NUR einmal pro EINDEUTIGEM Text parsen, nicht pro Zeile (Millionen Pixel
        # teilen sich wenige zehntausend eindeutige Texte, siehe Phase-5/6-Berichte) - danach
        # per text_id zurueckbroadcasten.
        # Sentinel "" statt None: default_collate() im DataLoader kann keine None-Werte in
        # einer Batch-Liste verarbeiten (bricht mit TypeError ab). "" ist kein gueltiger Key
        # (alle 13 Keys sind nicht-leer) - train.py::class_recall_at_k() ueberspringt ""
        # explizit statt sie faelschlich als Klasse zu werten.
        unique_texts = merged[TEXT_COL].drop_duplicates().reset_index(drop=True)
        key_by_text = {t: (extract_primary_key(t) or "") for t in unique_texts}
        self.primary_key = merged[TEXT_COL].map(key_by_text).to_numpy()

    def __len__(self) -> int:
        return len(self.coverage)

    def __getitem__(self, idx):
        return {
            "ee_embedding": torch.from_numpy(self.ee[idx]),
            "text_embedding": torch.from_numpy(self.text[idx]),
            "coverage": torch.tensor(self.coverage[idx]),
            "n_pois": torch.tensor(self.n_pois[idx]),
            "primary_key": self.primary_key[idx],
            "pixel_key": self.pixel_key[idx],
            "text_id": torch.tensor(self.text_id[idx]),
        }


def _smoke_test():
    """Kein echtes Textembedding vorhanden - Dummy-Parquets in einem temporaeren Verzeichnis
    pruefen nur den Join/DataLoader-Mechanismus (Merge ueber pixel_text statt poi_id)."""
    rng = np.random.default_rng(0)
    n, ee_dim, text_dim = 20, 128, 1024

    texts = [f"pixel text {i}" for i in range(n)]
    texts[5] = texts[0]  # Pixel 0 und 5 teilen sich denselben pixel_text -> gleicher text_id

    pixels = pd.DataFrame({
        "tile": [f"t{i % 3}.tif" for i in range(n)],
        "row": rng.integers(0, 100, n),
        "col": rng.integers(0, 100, n),
        "embedding": list(rng.standard_normal((n, ee_dim)).astype(np.float32)),
        "pixel_text": texts,
        "weight": np.ones(n, dtype=np.float32),
        "n_pois": rng.integers(1, 4, n),
        "coverage": rng.random(n).astype(np.float32),
        "n_pixels_poi": rng.integers(1, 5, n).astype(np.float32),
        "poi_ids": [f"way/{i}" for i in range(n)],
        "split": rng.choice(["train", "val", "test"], size=n, p=[0.6, 0.2, 0.2]),
    })
    unique_texts = pixels["pixel_text"].drop_duplicates().reset_index(drop=True)
    text_emb = pd.DataFrame({
        "text": unique_texts,
        "embedding": list(rng.standard_normal((len(unique_texts), text_dim)).astype(np.float32)),
    })

    with tempfile.TemporaryDirectory() as tmp:
        pixels_path = Path(tmp) / "pixels_dummy.parquet"
        text_path = Path(tmp) / "text_dummy.parquet"
        pixels.to_parquet(pixels_path)
        text_emb.to_parquet(text_path)

        ds = PairDataset(pixels_path, text_path)
        assert len(ds) == n
        assert ds.text_id[0] == ds.text_id[5], "identischer pixel_text muss denselben text_id bekommen"
        assert ds.text_id[1] != ds.text_id[2], "unterschiedliche Texte muessen unterschiedliche text_id bekommen"
        assert np.all(ds.sample_weight == 1.0), "weight muss durchgehend 1.0 sein"

        dl = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(dl))
        assert batch["ee_embedding"].shape == (4, ee_dim)
        assert batch["text_embedding"].shape == (4, text_dim)
        assert batch["coverage"].shape == (4,)
        assert batch["text_id"].shape == (4,)
        assert batch["n_pois"].shape == (4,)
        assert len(batch["primary_key"]) == 4

    # extract_primary_key: Single-Tag (Doppelpunkt-Form) und Multi-Tag (Nominalphrasen-Form)
    assert extract_primary_key("A building: house. building usually intended for living in") == "building"
    assert extract_primary_key("Greggs — a public amenity or service: fast food.") == "amenity"
    assert extract_primary_key("A pitch leisure spot and a soccer sports facility.") == "leisure"
    assert extract_primary_key(
        "A location containing Waitrose — a retail building and a supermarket shop.; "
        "a building: house."
    ) == "building"

    print(f"Smoke-Test OK: PairDataset (pixel_text-Merge) liefert Batches "
          f"ee={tuple(batch['ee_embedding'].shape)}, text={tuple(batch['text_embedding'].shape)}, "
          f"weight durchgehend 1.0, text_id markiert identische pixel_text korrekt, "
          f"extract_primary_key() erkennt Single- und Multi-Tag-Form korrekt.")


if __name__ == "__main__":
    _smoke_test()
