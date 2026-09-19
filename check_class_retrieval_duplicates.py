"""
check_class_retrieval_duplicates.py

Schritt 3 (2026-09-19, Jakob), Vorab-Check fuer run_class_retrieval.py: Galerie UND Ground
Truth kommen bei run_class_retrieval.py aus DERSELBEN Quelle - dem Test-Split von
pixels_{backbone}.parquet (Galerie = EO-Embeddings dieser Zeilen, Ground Truth = die aus
pixel_text derselben Zeilen geparsten Klassen). Die Dopplungsfrage reduziert sich damit auf:
wie viele TEST-Zeilen von pixels_tessera.parquet sind Dopplungs-Weltorte (siehe
check_tile_overlap_side_effects.py)? Nur lesen/berichten, kein Fix hier.
"""
import pandas as pd

import config as cfg
from check_tile_overlap_side_effects import _tile_meta, find_duplicate_rows, find_overlap_candidates

PIXELS_PATH = cfg.DATA / "processed" / "pixels_tessera.parquet"


def main():
    df = pd.read_parquet(PIXELS_PATH, columns=["tile", "row", "col", "split"])
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"pixels_tessera.parquet Test-Split (= run_class_retrieval.py Galerie/Ground-Truth-Quelle): "
          f"{len(test)} Zeilen")

    meta = _tile_meta("london", "tessera")
    candidates = find_overlap_candidates(meta)
    dup = find_duplicate_rows(test[["tile", "row", "col", "split"]], meta, candidates)
    n_pairs = len(dup)
    n_rows_affected = 2 * n_pairs
    print(f"\nDopplungs-Weltorte im Test-Split: {n_pairs} Paare "
          f"({n_rows_affected} betroffene Zeilen von {len(test)}, "
          f"{100 * n_rows_affected / len(test):.2f}%)")
    print(f"Davon wuerden bei Dedup {n_pairs} Zeilen entfernt "
          f"({100 * n_pairs / len(test):.2f}% der Galerie)")

    if n_pairs > 0:
        n_same_split = int(dup["same_split"].sum())
        print(f"Im selben Split (beide Kopien im Test, wie erwartet da beide aus demselben "
              f"Testfilter kommen): {n_same_split}/{n_pairs}")


if __name__ == "__main__":
    main()
