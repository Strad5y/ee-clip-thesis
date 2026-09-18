"""
diag_removal_rate_breakdown.py

Klaerung vor Schritt 2 (2026-09-18, Jakob): warum unterscheiden sich die Entfernungsraten
zwischen pixels_tessera (11,97%), pairs_tessera_full (10,20%) und der SDM-Poolungs-
Ausschlussrate (7,89%)? Zerlegt die entfernten pixels_tessera-Zeilen nach Ursache -
Eindeutigkeitsmaske (innerhalb AOI, aber nicht naechste Kachel), AOI-Polygon-Maske
(ausserhalb der echten Greater-London-Grenze), Restaufloesung (zweiter Schritt,
tessera_dedup_utils.py::resolve_residual_duplicates) - und prueft direkt, ob die ALTEN
Trainingspaare teilweise ausserhalb Londons gebildet wurden. Nur lesen/berichten.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

import config as cfg
import spatial_join as sj
from tessera_dedup_utils import build_full_mask_lookup

AOI_PATH = cfg.AOI_DIR / "london.gpkg"
ORIGINAL_PATH = cfg.DATA / "processed" / "pixels_tessera.parquet"
DEDUP_PATH = cfg.DATA / "processed" / "pixels_tessera_dedup.parquet"


def build_aoi_only_mask_lookup(backbone: str = "tessera", region: str = "london") -> dict:
    """NUR die AOI-Polygon-Bedingung (Pixelzentrum in Greater London), OHNE Eindeutigkeit -
    isoliert die zweite Maskenkomponente von masks.py::build_backbone_masks() fuer den
    Ursachen-Vergleich unten."""
    aoi = gpd.read_file(AOI_PATH)
    lookup = {}
    for raw_path in sj.get_backbones(region)[backbone]:
        with rasterio.open(raw_path) as src:
            shape = src.shape
            transform = src.transform
            crs = src.crs
        aoi_in_crs = aoi.to_crs(crs)
        mask = rasterize([(aoi_in_crs.geometry.iloc[0], 1)], out_shape=shape,
                          transform=transform, fill=0, dtype="uint8").astype(bool)
        lookup[raw_path.name] = mask
    return lookup


def lookup_bool(df: pd.DataFrame, mask_lookup: dict) -> np.ndarray:
    out = np.zeros(len(df), dtype=bool)
    tiles = df["tile"].to_numpy()
    rows = df["row"].to_numpy()
    cols = df["col"].to_numpy()
    for tile_name, mask in mask_lookup.items():
        sel = tiles == tile_name
        if not sel.any():
            continue
        r, c = rows[sel], cols[sel]
        in_bounds = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
        vals = np.zeros(int(sel.sum()), dtype=bool)
        vals[in_bounds] = mask[r[in_bounds], c[in_bounds]]
        out[sel] = vals
    return out


def main():
    print("Lade Original pixels_tessera.parquet...")
    orig = pd.read_parquet(ORIGINAL_PATH, columns=["tile", "row", "col"])
    n_orig = len(orig)
    print(f"  {n_orig} Zeilen")

    print("Lade dedupliziertes pixels_tessera_dedup.parquet...")
    dedup = pd.read_parquet(DEDUP_PATH, columns=["tile", "row", "col"])
    dedup_keys = set(zip(dedup["tile"], dedup["row"], dedup["col"]))
    print(f"  {len(dedup)} Zeilen")

    print("\nBaue Masken (kombiniert: Eindeutigkeit+AOI; separat: nur AOI-Polygon)...")
    combined_lookup = build_full_mask_lookup("tessera")
    aoi_only_lookup = build_aoi_only_mask_lookup("tessera")

    primary_ok = lookup_bool(orig, combined_lookup)
    aoi_ok = lookup_bool(orig, aoi_only_lookup)

    orig_keys = list(zip(orig["tile"], orig["row"], orig["col"]))
    in_dedup = np.array([k in dedup_keys for k in orig_keys])

    n_kept = int(in_dedup.sum())
    n_removed_total = n_orig - n_kept
    print(f"\nGesamt entfernt: {n_removed_total} ({100 * n_removed_total / n_orig:.2f}%)")

    # Kategorien (disjunkt per Konstruktion):
    #  - primary_ok=True aber nicht im Dedup-File  -> nur durch die Restaufloesung entfernt
    #  - primary_ok=False, aoi_ok=False             -> AOI-Polygon-Ausschluss (erklaert allein)
    #  - primary_ok=False, aoi_ok=True              -> reiner Eindeutigkeitsverlust (AOI war ok)
    cat_residual = primary_ok & ~in_dedup
    cat_aoi = (~primary_ok) & (~aoi_ok)
    cat_uniqueness = (~primary_ok) & aoi_ok

    n_residual = int(cat_residual.sum())
    n_aoi = int(cat_aoi.sum())
    n_uniqueness = int(cat_uniqueness.sum())
    n_check = n_residual + n_aoi + n_uniqueness
    print(f"\nZerlegung der {n_removed_total} entfernten Zeilen:")
    print(f"  Eindeutigkeitsmaske (innerhalb AOI, aber nicht naechste Kachel): {n_uniqueness} "
          f"({100 * n_uniqueness / n_orig:.2f}% von allen, "
          f"{100 * n_uniqueness / n_removed_total:.1f}% der Entfernten)")
    print(f"  AOI-Polygon-Maske (ausserhalb Greater London): {n_aoi} "
          f"({100 * n_aoi / n_orig:.2f}% von allen, {100 * n_aoi / n_removed_total:.1f}% der Entfernten)")
    print(f"  Restaufloesung (Grenzlinien-Rest, zweiter Schritt): {n_residual} "
          f"({100 * n_residual / n_orig:.2f}% von allen, "
          f"{100 * n_residual / n_removed_total:.1f}% der Entfernten)")
    print(f"  Summe Kategorien: {n_check} (Soll: {n_removed_total})")
    assert n_check == n_removed_total, "Kategorien summieren nicht auf die Gesamtzahl - Bug."

    print("\n*** Waren die ALTEN Trainingspaare teilweise ausserhalb Londons? ***")
    n_outside_aoi_total = int((~aoi_ok).sum())
    print(f"  Zeilen im URSPRUENGLICHEN pixels_tessera.parquet ausserhalb des AOI-Polygons "
          f"(unabhaengig von der Eindeutigkeit): {n_outside_aoi_total} "
          f"({100 * n_outside_aoi_total / n_orig:.2f}%)")


if __name__ == "__main__":
    main()
