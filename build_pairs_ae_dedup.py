"""
build_pairs_ae_dedup.py

AlphaEarth-Dedup (2026-09-21, Jakob), Schritt 1: dedupliziert pairs_ae_full.parquet ueber die
Eindeutigkeits+AOI-Maske plus Restaufloesung ueber Weltkoordinaten - EXAKT dieselbe Prozedur wie
build_pairs_tessera_dedup.py (Logik aus tessera_dedup_utils.py, backbone='ae'). Nichts
ueberschrieben: neue Datei pairs_ae_full_dedup.parquet, altes pairs_ae_full.parquet bleibt.
"""
import numpy as np
import pandas as pd

import config as cfg
import spatial_join as sj
from ae_dedup_utils import (BACKBONE, breakdown_counts, pixel_key, primary_and_aoi_flags,
                            print_breakdown)
from check_tile_overlap_side_effects import _tile_meta, find_duplicate_rows, find_overlap_candidates
from diag_removal_rate_breakdown import build_aoi_only_mask_lookup
from tessera_dedup_utils import build_full_mask_lookup, mask_keep, resolve_residual_duplicates

IN_PATH = cfg.DATA / "processed" / "pairs_ae_full.parquet"
OUT_PATH = cfg.DATA / "processed" / "pairs_ae_full_dedup.parquet"


def main():
    print(f"Lade {IN_PATH}...")
    df = pd.read_parquet(IN_PATH)
    n_before = len(df)
    print(f"  {n_before} Zeilen")
    tile_names = [p.name for p in sj.get_backbones("london")[BACKBONE]]

    print("Baue volle Kachel-Masken (Eindeutigkeit + AOI-Polygon) und reine AOI-Maske...")
    mask_lookup = build_full_mask_lookup(BACKBONE)
    aoi_lookup = build_aoi_only_mask_lookup(BACKBONE)

    primary_ok, aoi_ok = primary_and_aoi_flags(df, mask_lookup, aoi_lookup)
    assert (primary_ok == mask_keep(df, mask_lookup)).all()
    df_dedup = df[primary_ok].reset_index(drop=True)
    print(f"  Nach primaerer Maskierung: {len(df_dedup)} Zeilen")

    print("\nZweiter Aufloesungsschritt (Grenzlinien-Rest zwischen unabhaengigen Masken)...")
    meta = _tile_meta("london", BACKBONE)
    candidates = find_overlap_candidates(meta)
    df_dedup, n_removed_residual, n_rounds = resolve_residual_duplicates(df_dedup, meta, candidates)
    print(f"  {n_removed_residual} Zeilen in {n_rounds + 1} Runde(n) zusaetzlich entfernt")
    n_after = len(df_dedup)

    print(f"\nZeilen vorher: {n_before}")
    print(f"Zeilen nachher: {n_after}")
    print(f"Entfernt gesamt: {n_before - n_after} ({100 * (n_before - n_after) / n_before:.2f}%)")

    kept_final = np.isin(pixel_key(df, tile_names), pixel_key(df_dedup, tile_names))
    assert int(kept_final.sum()) == n_after, "Zeilenzahl der Schluesselmenge passt nicht zum Ergebnis"
    print_breakdown("pairs_ae_full", breakdown_counts(primary_ok, aoi_ok, kept_final))

    print("\nVerifikation: verbleibende Dopplungen (Soll: 0)...")
    uniq = df_dedup[["tile", "row", "col"]].drop_duplicates().assign(split="n/a")
    dup = find_duplicate_rows(uniq, meta, candidates)
    print(f"  Verbleibende Dopplungs-Weltorte: {len(dup)} (Soll: 0)")
    assert len(dup) == 0, f"{len(dup)} Dopplungen verbleiben nach Maskierung - Bug."

    df_dedup.to_parquet(OUT_PATH)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
