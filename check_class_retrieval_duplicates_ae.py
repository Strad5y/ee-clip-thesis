"""
check_class_retrieval_duplicates_ae.py

Schritt 3 AlphaEarth (2026-09-21, Jakob): AE-Variante von check_class_retrieval_duplicates.py
(das ist fest auf pixels_tessera.parquet/'tessera' verdrahtet und bleibt unveraendert).

Galerie UND Ground Truth kommen bei run_class_retrieval.py aus DERSELBEN Quelle - dem Test-Split
von pixels_<backbone>.parquet. Die Dopplungsfrage reduziert sich damit auf: wie viele TEST-Zeilen
sind Dopplungs-Weltorte (check_tile_overlap_side_effects.find_duplicate_rows)? Berichtet fuer die
ALTE Galerie (pixels_ae) und belegt fuer die NEUE (pixels_ae_dedup) 0 Dopplungen (assert).
"""
import pandas as pd

import config as cfg
from check_tile_overlap_side_effects import _tile_meta, find_duplicate_rows, find_overlap_candidates

BACKBONE = "ae"
PIXELS_PATHS = {"alt": cfg.DATA / "processed" / "pixels_ae.parquet",
                "neu": cfg.DATA / "processed" / "pixels_ae_dedup.parquet"}


def main():
    meta = _tile_meta("london", BACKBONE)
    candidates = find_overlap_candidates(meta)
    for label, path in PIXELS_PATHS.items():
        df = pd.read_parquet(path, columns=["tile", "row", "col", "split"])
        test = df[df["split"] == "test"].reset_index(drop=True)
        print(f"[{label}] {path.name} Test-Split (= Galerie/Ground-Truth-Quelle von "
              f"run_class_retrieval): {len(test)} Zeilen")
        dup = find_duplicate_rows(test[["tile", "row", "col", "split"]], meta, candidates)
        n_pairs = len(dup)
        print(f"[{label}] Dopplungs-Weltort-Paare im Test-Split: {n_pairs} "
              f"({2 * n_pairs} betroffene Zeilen von {len(test)}, "
              f"{100 * 2 * n_pairs / len(test):.3f}%)")
        if label == "neu":
            assert n_pairs == 0, f"{n_pairs} Dopplungen in der Neu-Galerie - Bug."
            print("[neu] 0 verbleibende Dopplungs-Weltorte in der Galerie: belegt.")


if __name__ == "__main__":
    main()
