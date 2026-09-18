"""
build_landuse_pairs_tessera_dedup.py

Hauptaufgabe (2026-09-18, Jakob), Schritt 1: baut landuse_pairs_tessera neu, mit der
Eindeutigkeits+AOI-Maske EINGEFUEGT VOR der bestehenden Coverage-Dedup/Sampling-Logik aus
build_landuse_pairs.py (load_and_clip/deduplicate_pixels/assign_pixel_splits werden
UNVERAENDERT wiederverwendet, nicht neu geschrieben). Nur TESSERA - AlphaEarth unveraendert.

Reihenfolge wichtig: die Kachel-Maske filtert Weltort-Dopplungen (mehrere Kacheln fuer
dieselbe Position) - eine ANDERE, unabhaengige Ursache als deduplicate_pixels() (mehrere
Urban-Atlas-POLYGONE treffen dasselbe Pixel EINER Kachel). Maske zuerst, dann die bestehende
Coverage-Dedup, dann Sampling - genau die Reihenfolge, die auch bei einem kompletten Neubau
ohne die spaetere Kachel-Ueberlappungs-Erkenntnis entstanden waere.
"""
import numpy as np
import pandas as pd

import config as cfg
import spatial_join as sj
from build_landuse_pairs import (
    CARRY, ID_COL, MAX_PIXELS_PER_POLYGON, SAMPLE_SEED, assign_pixel_splits, deduplicate_pixels,
    load_and_clip,
)
from tessera_dedup_utils import build_full_mask_lookup, mask_keep, resolve_residual_duplicates
from check_tile_overlap_side_effects import _tile_meta, find_overlap_candidates, find_duplicate_rows

ORIGINAL_PATH = cfg.DATA / "processed" / "landuse_pairs_tessera.parquet"
OUT_PATH = cfg.DATA / "processed" / "landuse_pairs_tessera_dedup.parquet"


def main():
    print("Lade/klippe Urban Atlas (identisch zu build_landuse_pairs.py)...")
    ua = load_and_clip()

    tile_paths_list = [p for p in sj.BACKBONES["tessera"] if p.exists()]
    print(f"\n{len(tile_paths_list)} TESSERA-Kacheln")

    print("Join gegen alle Kacheln (spatial_join.join_pois_to_tiles, wie im Original)...")
    pairs = sj.join_pois_to_tiles(ua, tile_paths_list, id_col=ID_COL, carry=CARRY,
                                   dequantize=sj.DEQUANTIZE.get("tessera"))
    n_before_mask = len(pairs)
    print(f"  Pixel-Polygon-Zeilen vor Maskierung: {n_before_mask}")

    print("\nWende Eindeutigkeits+AOI-Maske an...")
    mask_lookup = build_full_mask_lookup("tessera")
    keep = mask_keep(pairs, mask_lookup)
    pairs = pairs[keep].reset_index(drop=True)
    n_after_mask = len(pairs)
    print(f"  Pixel-Polygon-Zeilen nach primaerer Maskierung: {n_after_mask} "
          f"(entfernt: {n_before_mask - n_after_mask}, "
          f"{100 * (n_before_mask - n_after_mask) / n_before_mask:.2f}%)")

    print("\nZweiter Auflösungsschritt (Grenzlinien-Rest zwischen unabhaengigen Masken)...")
    meta = _tile_meta("london", "tessera")
    candidates = find_overlap_candidates(meta)
    pairs, n_removed_residual, n_rounds = resolve_residual_duplicates(pairs, meta, candidates)
    print(f"  {n_removed_residual} Zeilen in {n_rounds + 1} Runde(n) zusaetzlich entfernt")

    pairs, _dedup_stats = deduplicate_pixels(pairs)  # bestehende Max-Coverage-Dedup, andere Ursache

    rng = np.random.default_rng(SAMPLE_SEED)
    parts = [g.sample(n=min(len(g), MAX_PIXELS_PER_POLYGON), random_state=rng)
             for _, g in pairs.groupby(ID_COL)]
    sampled = pd.concat(parts, ignore_index=True)
    print(f"  Pixel-Zeilen nach Sampling (max {MAX_PIXELS_PER_POLYGON}/Polygon): {len(sampled)}")

    tile_paths = {p.name: p for p in tile_paths_list}
    split = assign_pixel_splits(sampled, tile_paths)
    sampled["split"] = split.to_numpy()
    n_missing = int((sampled["split"] == "missing").sum())
    if n_missing > 0:
        print(f"  {n_missing} Pixel ohne Split-Zuordnung (split='missing')")

    n_before_orig = len(pd.read_parquet(ORIGINAL_PATH, columns=["tile"]))
    print(f"\nZeilen vorher (bestehendes landuse_pairs_tessera.parquet): {n_before_orig}")
    print(f"Zeilen nachher (dedupliziert): {len(sampled)}")
    print(f"Differenz: {n_before_orig - len(sampled)} "
          f"({100 * (n_before_orig - len(sampled)) / n_before_orig:.2f}%)")

    print("\nVerifikation: verbleibende Dopplungen (Soll: 0)...")
    meta = _tile_meta("london", "tessera")
    candidates = find_overlap_candidates(meta)
    dup = find_duplicate_rows(sampled[["tile", "row", "col", "split"]], meta, candidates)
    print(f"  Verbleibende Dopplungs-Weltorte: {len(dup)} (Soll: 0)")
    assert len(dup) == 0, f"{len(dup)} Dopplungen verbleiben nach Maskierung - Bug."

    sampled.to_parquet(OUT_PATH)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
