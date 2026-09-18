"""
tessera_dedup_utils.py

Hauptaufgabe (2026-09-18, Jakob): TESSERA vollstaendig deduplizieren. Gemeinsame Bausteine,
wiederverwendet von build_pairs_tessera_dedup.py, build_pixels_tessera_dedup.py,
build_landuse_pairs_tessera_dedup.py - EIN Ort fuer die Maskenlogik, nicht pro Skript
dupliziert.

Grundsatz (Jakob): jeder Weltort existiert in jedem TESSERA-Datensatz genau einmal. Regel:
die bereits implementierte Eindeutigkeitsmaske (naechster Kachelmittelpunkt gewinnt, Tie-Break
lexikographisch kleinerer Dateiname, src/eval/text_query/masks.py) PLUS die AOI-Polygon-Maske -
beide bereits als GeoTIFF unter data/interim/text_query_masks/tessera/ gespeichert
(Text-Query-Aufgabe, Phase 2), im AOI-zugeschnittenen Arm-C-Grid. Fuer die Pixel-/Paar-
Tabellen (die im VOLLEN Rohkachel-Grid indiziert sind) wird dieselbe Maske an der exakten
Fensterposition eingebettet, die build_sdm_projected_field.py beim Erzeugen des Arm-C-Feldes
verwendet hat (build_sdm_pooled_embeddings_dedup.py::load_mask_for_tile, hier importiert
statt dupliziert - keine erneute Berechnung der Eindeutigkeitsregel).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

import config as cfg
import spatial_join as sj
from build_sdm_pooled_embeddings_dedup import load_mask_for_tile
from check_tile_overlap_side_effects import find_duplicate_rows

AOI_PATH = cfg.AOI_DIR / "london.gpkg"


def build_full_mask_lookup(backbone: str = "tessera", region: str = "london") -> dict:
    """tile_name -> volles bool-Array im Rohkachel-Grid (Eindeutigkeit UND AOI-Polygon)."""
    aoi_gpd = gpd.read_file(AOI_PATH)
    lookup = {}
    for raw_path in sj.get_backbones(region)[backbone]:
        with rasterio.open(raw_path) as src:
            shape = src.shape
        mask = load_mask_for_tile(backbone, raw_path, raw_path, "a", aoi_gpd, shape)
        lookup[raw_path.name] = mask
    return lookup


def mask_keep(df: pd.DataFrame, mask_lookup: dict) -> np.ndarray:
    """Vektorisiert: True, wo (tile,row,col) laut Maske gueltig/eindeutig dieser Kachel gehoert."""
    keep = np.zeros(len(df), dtype=bool)
    tiles = df["tile"].to_numpy()
    rows = df["row"].to_numpy()
    cols = df["col"].to_numpy()
    for tile_name, mask in mask_lookup.items():
        sel = tiles == tile_name
        if not sel.any():
            continue
        r = rows[sel]
        c = cols[sel]
        in_bounds = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
        vals = np.zeros(int(sel.sum()), dtype=bool)
        vals[in_bounds] = mask[r[in_bounds], c[in_bounds]]
        keep[sel] = vals
    n_unmatched_tile = int((~np.isin(tiles, list(mask_lookup.keys()))).sum())
    if n_unmatched_tile > 0:
        raise ValueError(f"{n_unmatched_tile} Zeilen mit einer Kachel ohne Maske - "
                          f"unerwartet, alle TESSERA-Kacheln sollten eine Maske haben.")
    return keep


def resolve_residual_duplicates(df: pd.DataFrame, meta: dict, candidates: list,
                                 max_rounds: int = 5) -> tuple:
    """Zweiter Auflösungsschritt NACH der primären Maskierung (mask_keep()): die statischen
    Pro-Kachel-Masken werden UNABHAENGIG voneinander gebaut (jede Kachel sieht nur ihre eigenen
    Pixelzentren, nicht die der Nachbarkachel) - an der exakten mathematischen Grenzlinie
    zweier NICHT phasengleicher 10-m-Gitter kann das dazu fuehren, dass beide Seiten ihr
    jeweils eigenes (real existierendes, aber nur ~5-7m auseinanderliegendes) Pixel
    unabhaengig als "naeher am eigenen Zentrum" werten - ein duenner Grenzlinien-Rest bleibt
    dann in BEIDEN Masken als True stehen (verifiziert: 2.083 von 8,75 Mio. TESSERA-Pixeln,
    0,024%, alle exakt auf Kachelgrenzen). find_duplicate_rows() rechnet die Eindeutigkeitsregel
    DIREKT auf dem Paar nach (ueber Weltkoordinaten, ohne die Vorab-Diskretisierung der
    statischen Masken) und liefert einen definitiven Gewinner je Fundstelle - dessen
    Verlierer-Zeilen werden hier entfernt. Iteriert bis zum Fixpunkt (max_rounds als
    Sicherheitsnetz), meldet aber bereits nach der ersten Runde nahezu immer 0 Rest."""
    total_removed = 0
    for round_i in range(max_rounds):
        uniq = df[["tile", "row", "col"]].drop_duplicates().assign(split="n/a")
        dup = find_duplicate_rows(uniq, meta, candidates)
        if len(dup) == 0:
            return df, total_removed, round_i
        loser_keys = set()
        for _, r in dup.iterrows():
            loser_tile = r["loser_tile"]
            if loser_tile == r["tile_a"]:
                loser_keys.add((r["tile_a"], r["row_a"], r["col_a"]))
            else:
                loser_keys.add((r["tile_b"], r["row_b"], r["col_b"]))
        row_keys = list(zip(df["tile"], df["row"], df["col"]))
        drop_mask = np.array([k in loser_keys for k in row_keys])
        total_removed += int(drop_mask.sum())
        df = df[~drop_mask].reset_index(drop=True)
    raise RuntimeError(f"resolve_residual_duplicates: nach {max_rounds} Runden immer noch "
                        f"Dopplungen vorhanden - kein Fixpunkt erreicht, echter Bug.")
