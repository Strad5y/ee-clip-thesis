"""
build_pairs_tessera_dedup.py

Hauptaufgabe (2026-09-18, Jakob), Schritt 1: dedupliziert pairs_tessera_full.parquet (POI-
Pixel-Join, vor der Pixel-Aggregation) ueber die Eindeutigkeits+AOI-Maske - jeder Weltort
(tile,row,col) gehoert danach genau einer Kachel. AlphaEarth unveraendert, nichts
ueberschrieben (neue Datei, altes pairs_tessera_full.parquet bleibt).
"""
import pandas as pd

import config as cfg
from tessera_dedup_utils import build_full_mask_lookup, mask_keep, resolve_residual_duplicates
from check_tile_overlap_side_effects import _tile_meta, find_overlap_candidates, find_duplicate_rows

IN_PATH = cfg.DATA / "processed" / "pairs_tessera_full.parquet"
OUT_PATH = cfg.DATA / "processed" / "pairs_tessera_full_dedup.parquet"


def main():
    print(f"Lade {IN_PATH}...")
    df = pd.read_parquet(IN_PATH)
    n_before = len(df)
    print(f"  {n_before} Zeilen")

    print("Baue volle Kachel-Masken (Eindeutigkeit + AOI-Polygon)...")
    mask_lookup = build_full_mask_lookup("tessera")

    keep = mask_keep(df, mask_lookup)
    n_after_mask = int(keep.sum())
    df_dedup = df[keep].reset_index(drop=True)
    print(f"  Nach primaerer Maskierung: {n_after_mask} Zeilen")

    print("\nZweiter Auflösungsschritt (Grenzlinien-Rest zwischen unabhaengigen Masken)...")
    meta = _tile_meta("london", "tessera")
    candidates = find_overlap_candidates(meta)
    df_dedup, n_removed_residual, n_rounds = resolve_residual_duplicates(df_dedup, meta, candidates)
    print(f"  {n_removed_residual} Zeilen in {n_rounds + 1} Runde(n) zusaetzlich entfernt "
          f"(Grenzlinien-Rest, siehe tessera_dedup_utils.py-Docstring)")
    n_after = len(df_dedup)

    print(f"\nZeilen vorher: {n_before}")
    print(f"Zeilen nachher: {n_after}")
    print(f"Entfernt gesamt: {n_before - n_after} ({100 * (n_before - n_after) / n_before:.2f}%)")

    print("\nVerifikation: verbleibende Dopplungen (Soll: 0)...")
    uniq = df_dedup[["tile", "row", "col"]].drop_duplicates().assign(split="n/a")
    dup = find_duplicate_rows(uniq, meta, candidates)
    print(f"  Verbleibende Dopplungs-Weltorte: {len(dup)} (Soll: 0)")
    assert len(dup) == 0, f"{len(dup)} Dopplungen verbleiben nach Maskierung - Bug."

    df_dedup.to_parquet(OUT_PATH)
    print(f"\nGespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
