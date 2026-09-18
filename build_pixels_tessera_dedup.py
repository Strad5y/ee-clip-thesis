"""
build_pixels_tessera_dedup.py

Hauptaufgabe (2026-09-18, Jakob), Schritt 1: baut pixels_tessera neu aus dem deduplizierten
pairs_tessera_full_dedup.parquet (build_pairs_tessera_dedup.py) - wiederverwendet
src/aggregate_pixel_pairs.py::aggregate() UNVERAENDERT (identische Text-Komposition/Split-
Logik), nur der Input-Pfad ist neu. AlphaEarth unveraendert. Splits kommen weiterhin aus dem
AKTUELLEN data/splits/splits_london.csv (der Split-Neubau+Vergleich ist ein separater,
nachgelagerter Schritt - check_splits_rebuild_dedup.py).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd

import config as cfg
from aggregate_pixel_pairs import aggregate, build_cell_to_split, report, validate
from spatial_join import BACKBONES
from check_tile_overlap_side_effects import _tile_meta, find_overlap_candidates, find_duplicate_rows

PAIRS_DEDUP_PATH = cfg.DATA / "processed" / "pairs_tessera_full_dedup.parquet"
POI_TEXT_PATH = cfg.DATA / "interim" / "poi_text_full.parquet"
ORIGINAL_PIXELS_PATH = cfg.DATA / "processed" / "pixels_tessera.parquet"
OUT_PATH = cfg.DATA / "processed" / "pixels_tessera_dedup.parquet"


def main():
    print("Lade Split-Zellen-Referenz (unveraendert, aktuelles splits_london.csv)...")
    cell_to_split, utm_crs, (minx, miny) = build_cell_to_split()
    print(f"  {len(cell_to_split)} Zellen mit bekanntem Split")

    poi_text = pd.read_parquet(POI_TEXT_PATH, columns=["poi_id", "text", "body"])
    print(f"poi_text_full.parquet: {len(poi_text)} POIs geladen")

    tile_paths = BACKBONES["tessera"]
    out, stats = aggregate("tessera_dedup", tile_paths, PAIRS_DEDUP_PATH, poi_text,
                            cell_to_split, utm_crs, minx, miny)
    validate(out, "tessera_dedup")
    report(out, stats, "tessera_dedup")

    n_before_orig = len(pd.read_parquet(ORIGINAL_PIXELS_PATH, columns=["tile"]))
    print(f"\nZeilen vorher (bestehendes pixels_tessera.parquet): {n_before_orig}")
    print(f"Zeilen nachher (dedupliziert): {len(out)}")
    print(f"Entfernt: {n_before_orig - len(out)} "
          f"({100 * (n_before_orig - len(out)) / n_before_orig:.2f}%)")

    print("\nVerifikation: verbleibende Dopplungen (Soll: 0)...")
    meta = _tile_meta("london", "tessera")
    candidates = find_overlap_candidates(meta)
    dup = find_duplicate_rows(out[["tile", "row", "col", "split"]], meta, candidates)
    print(f"  Verbleibende Dopplungs-Weltorte: {len(dup)} (Soll: 0)")
    assert len(dup) == 0, f"{len(dup)} Dopplungen verbleiben nach Maskierung - Bug."

    out[["tile", "row", "col", "embedding", "pixel_text", "weight", "n_pois",
         "coverage", "n_pixels_poi", "poi_ids", "split"]].to_parquet(OUT_PATH)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
