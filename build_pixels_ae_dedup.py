"""
build_pixels_ae_dedup.py

AlphaEarth-Dedup (2026-09-21, Jakob), Schritt 1: baut pixels_ae neu aus dem deduplizierten
pairs_ae_full_dedup.parquet (build_pairs_ae_dedup.py) - wiederverwendet
src/aggregate_pixel_pairs.py::aggregate() UNVERAENDERT (analog build_pixels_tessera_dedup.py).
Splits kommen weiterhin aus dem AKTUELLEN data/splits/splits_london.csv (der Split-Neubau+
Vergleich ist separat: check_splits_rebuild_dedup_ae.py). Zerlegt zusaetzlich die aus
pixels_ae entfernten Zeilen nach Ursache (Eindeutigkeit / AOI / Restaufloesung).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

import config as cfg
import spatial_join as sj
from aggregate_pixel_pairs import aggregate, build_cell_to_split, report, validate
from ae_dedup_utils import (BACKBONE, breakdown_counts, pixel_key, primary_and_aoi_flags,
                            print_breakdown)
from check_tile_overlap_side_effects import _tile_meta, find_duplicate_rows, find_overlap_candidates
from diag_removal_rate_breakdown import build_aoi_only_mask_lookup
from tessera_dedup_utils import build_full_mask_lookup

PAIRS_DEDUP_PATH = cfg.DATA / "processed" / "pairs_ae_full_dedup.parquet"
POI_TEXT_PATH = cfg.DATA / "interim" / "poi_text_full.parquet"
ORIGINAL_PIXELS_PATH = cfg.DATA / "processed" / "pixels_ae.parquet"
OUT_PATH = cfg.DATA / "processed" / "pixels_ae_dedup.parquet"


def main():
    print("Lade Split-Zellen-Referenz (unveraendert, aktuelles splits_london.csv)...")
    cell_to_split, utm_crs, (minx, miny) = build_cell_to_split()
    print(f"  {len(cell_to_split)} Zellen mit bekanntem Split")

    poi_text = pd.read_parquet(POI_TEXT_PATH, columns=["poi_id", "text", "body"])
    print(f"poi_text_full.parquet: {len(poi_text)} POIs geladen")

    tile_paths = sj.BACKBONES[BACKBONE]
    out, stats = aggregate("ae_dedup", tile_paths, PAIRS_DEDUP_PATH, poi_text,
                            cell_to_split, utm_crs, minx, miny)
    validate(out, "ae_dedup")
    report(out, stats, "ae_dedup")

    orig = pd.read_parquet(ORIGINAL_PIXELS_PATH, columns=["tile", "row", "col", "split"])
    n_before_orig = len(orig)
    print(f"\nZeilen vorher (bestehendes pixels_ae.parquet): {n_before_orig}")
    print(f"Zeilen nachher (dedupliziert): {len(out)}")
    print(f"Entfernt: {n_before_orig - len(out)} "
          f"({100 * (n_before_orig - len(out)) / n_before_orig:.2f}%)")

    print("\nUrsachen-Zerlegung der aus pixels_ae entfernten Zeilen...")
    tile_names = [p.name for p in tile_paths]
    combined = build_full_mask_lookup(BACKBONE)
    aoi_only = build_aoi_only_mask_lookup(BACKBONE)
    primary_ok, aoi_ok = primary_and_aoi_flags(orig, combined, aoi_only)
    kept_final = np.isin(pixel_key(orig, tile_names), pixel_key(out, tile_names))
    assert int(kept_final.sum()) == len(out), "Zeilenzahl der Schluesselmenge passt nicht zum Ergebnis"
    print_breakdown("pixels_ae", breakdown_counts(primary_ok, aoi_ok, kept_final))

    meta = _tile_meta("london", BACKBONE)
    candidates = find_overlap_candidates(meta)
    dup_orig = find_duplicate_rows(orig[["tile", "row", "col", "split"]], meta, candidates)
    print(f"\nKontrolle: Dopplungs-Weltorte im ORIGINAL pixels_ae (vor jeder Maske): {len(dup_orig)}")

    print("\nVerifikation: verbleibende Dopplungen (Soll: 0)...")
    dup = find_duplicate_rows(out[["tile", "row", "col", "split"]], meta, candidates)
    print(f"  Verbleibende Dopplungs-Weltorte: {len(dup)} (Soll: 0)")
    assert len(dup) == 0, f"{len(dup)} Dopplungen verbleiben nach Maskierung - Bug."

    out[["tile", "row", "col", "embedding", "pixel_text", "weight", "n_pois",
         "coverage", "n_pixels_poi", "poi_ids", "split"]].to_parquet(OUT_PATH)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
