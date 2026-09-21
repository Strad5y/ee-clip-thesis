"""
build_landuse_pairs_ae_dedup.py

AlphaEarth-Dedup (2026-09-21, Jakob), Schritt 1: baut landuse_pairs_ae neu, mit der
Eindeutigkeits+AOI-Maske (plus Restaufloesung) EINGEFUEGT VOR der bestehenden Coverage-Dedup/
Sampling-Logik aus build_landuse_pairs.py - exakt die Reihenfolge von
build_landuse_pairs_tessera_dedup.py. Zerlegt zusaetzlich die durch die Masken entfernten
Pixel-Polygon-Zeilen nach Ursache (vor Coverage-Dedup/Sampling).
"""
import numpy as np
import pandas as pd

import config as cfg
import spatial_join as sj
from ae_dedup_utils import (BACKBONE, breakdown_counts, pixel_key, primary_and_aoi_flags,
                            print_breakdown)
from build_landuse_pairs import (
    CARRY, ID_COL, MAX_PIXELS_PER_POLYGON, SAMPLE_SEED, assign_pixel_splits, deduplicate_pixels,
    load_and_clip,
)
from check_tile_overlap_side_effects import _tile_meta, find_duplicate_rows, find_overlap_candidates
from diag_removal_rate_breakdown import build_aoi_only_mask_lookup
from tessera_dedup_utils import build_full_mask_lookup, resolve_residual_duplicates

ORIGINAL_PATH = cfg.DATA / "processed" / "landuse_pairs_ae.parquet"
OUT_PATH = cfg.DATA / "processed" / "landuse_pairs_ae_dedup.parquet"


def main():
    print("Lade/klippe Urban Atlas (identisch zu build_landuse_pairs.py)...")
    ua = load_and_clip()

    tile_paths_list = [p for p in sj.BACKBONES[BACKBONE] if p.exists()]
    print(f"\n{len(tile_paths_list)} AlphaEarth-Kacheln")

    print("Join gegen alle Kacheln (spatial_join.join_pois_to_tiles, wie im Original)...")
    pairs = sj.join_pois_to_tiles(ua, tile_paths_list, id_col=ID_COL, carry=CARRY,
                                   dequantize=sj.DEQUANTIZE.get(BACKBONE))
    n_before_mask = len(pairs)
    print(f"  Pixel-Polygon-Zeilen vor Maskierung: {n_before_mask}")

    print("\nWende Eindeutigkeits+AOI-Maske an...")
    mask_lookup = build_full_mask_lookup(BACKBONE)
    aoi_lookup = build_aoi_only_mask_lookup(BACKBONE)
    primary_ok, aoi_ok = primary_and_aoi_flags(pairs, mask_lookup, aoi_lookup)
    tile_names = [p.name for p in tile_paths_list]
    key_before = pixel_key(pairs, tile_names)
    pairs = pairs[primary_ok].reset_index(drop=True)
    n_after_mask = len(pairs)
    print(f"  Pixel-Polygon-Zeilen nach primaerer Maskierung: {n_after_mask} "
          f"(entfernt: {n_before_mask - n_after_mask}, "
          f"{100 * (n_before_mask - n_after_mask) / n_before_mask:.2f}%)")

    print("\nZweiter Auflösungsschritt (Grenzlinien-Rest zwischen unabhaengigen Masken)...")
    meta = _tile_meta("london", BACKBONE)
    candidates = find_overlap_candidates(meta)
    pairs, n_removed_residual, n_rounds = resolve_residual_duplicates(pairs, meta, candidates)
    print(f"  {n_removed_residual} Zeilen in {n_rounds + 1} Runde(n) zusaetzlich entfernt")

    kept_final = np.isin(key_before, pixel_key(pairs, tile_names))
    print_breakdown("landuse Pixel-Polygon-Zeilen vor Coverage-Dedup/Sampling",
                    breakdown_counts(primary_ok, aoi_ok, kept_final))

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
    print(f"\nZeilen vorher (bestehendes landuse_pairs_ae.parquet): {n_before_orig}")
    print(f"Zeilen nachher (dedupliziert): {len(sampled)}")
    print(f"Differenz: {n_before_orig - len(sampled)} "
          f"({100 * (n_before_orig - len(sampled)) / n_before_orig:.2f}%)")

    print("\nVerifikation: verbleibende Dopplungen (Soll: 0)...")
    dup = find_duplicate_rows(sampled[["tile", "row", "col", "split"]], meta, candidates)
    print(f"  Verbleibende Dopplungs-Weltorte: {len(dup)} (Soll: 0)")
    assert len(dup) == 0, f"{len(dup)} Dopplungen verbleiben nach Maskierung - Bug."

    sampled.to_parquet(OUT_PATH)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
