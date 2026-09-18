"""
check_splits_rebuild_dedup.py

Hauptaufgabe (2026-09-18, Jakob), Schritt 1: baut die 1-km-Split-Zellen (SEED=0, identischer
Algorithmus wie build_splits_pixels.py - assign_splits() importiert, nicht dupliziert) auf dem
NEUEN Pixelbestand (pixels_tessera_dedup.parquet fuer TESSERA, unveraendertes
pixels_ae.parquet fuer AlphaEarth) frisch und vergleicht sie gegen data/splits/splits_london.csv.
Berichtet Abweichungen, LOEST SIE NICHT AUF, ueberschreibt die bestehende Datei NICHT.

Erwartung (explizit, nicht ueberraschend): build_splits.py::assign_splits() zieht EINE globale
Permutation ueber ALLE eindeutigen Zell-IDs (np.random.default_rng(seed).permutation(array)).
Aendert sich die Zell-MENGE (weil dedupliziertes TESSERA einzelne, nur von der verlierenden
Kachel beruehrte Zellen verliert), aendert sich fuer denselben Seed die Permutation JEDER
verbleibenden Zelle - ein grossflaechiger Split-Unterschied ist deshalb strukturell erwartbar,
kein Fehler.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import geopandas as gpd
import pandas as pd

import config as cfg
from aggregate_pixel_pairs import compute_pixel_cells
from build_splits import SEED, SPLIT_RATIOS, assign_splits
from spatial_join import BACKBONES

AOI_PATH = cfg.AOI_DIR / "london.gpkg"
CURRENT_SPLITS = cfg.DATA / "splits" / "splits_london.csv"
PIXELS_PATHS = {
    "tessera": cfg.DATA / "processed" / "pixels_tessera_dedup.parquet",
    "ae": cfg.DATA / "processed" / "pixels_ae.parquet",  # unveraendert
}


def get_aoi_utm_reference():
    aoi = gpd.read_file(AOI_PATH)
    utm_crs = aoi.estimate_utm_crs()
    aoi_utm = aoi.to_crs(utm_crs)
    minx, miny, _, _ = aoi_utm.total_bounds
    return utm_crs, minx, miny


def main():
    utm_crs, minx, miny = get_aoi_utm_reference()
    print(f"AOI-UTM-CRS: {utm_crs}, minx={minx:.1f}, miny={miny:.1f}")

    all_cells = []
    for name, path in PIXELS_PATHS.items():
        px = pd.read_parquet(path, columns=["tile", "row", "col"])
        cells = compute_pixel_cells(px, BACKBONES[name], utm_crs, minx, miny)
        all_cells.append(cells[["cell_row", "cell_col"]])
        print(f"  {name}: {len(px)} Pixel -> "
              f"{cells[['cell_row', 'cell_col']].drop_duplicates().shape[0]} Zellen")

    union_cells = pd.concat(all_cells, ignore_index=True).drop_duplicates().reset_index(drop=True)
    union_cells["cell_id"] = union_cells["cell_row"].astype(str) + "_" + union_cells["cell_col"].astype(str)
    print(f"\nVereinigung (neu): {len(union_cells)} distinkte Zellen (SEED={SEED})")

    cell_to_split = assign_splits(union_cells["cell_id"].to_numpy(), SPLIT_RATIOS, SEED)
    union_cells["split_new"] = union_cells["cell_id"].map(cell_to_split)

    current = pd.read_csv(CURRENT_SPLITS)
    current["cell_id"] = current["cell_row"].astype(str) + "_" + current["cell_col"].astype(str)
    print(f"Bestehende Datei: {len(current)} Zellen")
    current_map = current.drop_duplicates("cell_id").set_index("cell_id")["split"]

    union_cells["split_old"] = union_cells["cell_id"].map(current_map)

    n_only_new = int(union_cells["split_old"].isna().sum())
    n_only_old = int((~current_map.index.isin(union_cells["cell_id"])).sum())
    both = union_cells.dropna(subset=["split_old"])
    n_diff = int((both["split_new"] != both["split_old"]).sum())

    print(f"\nZellen nur in NEUER Vereinigung (nicht in altem splits_london.csv): {n_only_new}")
    print(f"Zellen nur in ALTER splits_london.csv (im neuen Pixelbestand nicht mehr vorhanden): {n_only_old}")
    print(f"Zellen in BEIDEN, aber mit ABWEICHENDEM Split: {n_diff}/{len(both)} "
          f"({100 * n_diff / len(both):.2f}%)")
    print(f"Zellen in BEIDEN mit GLEICHEM Split: {len(both) - n_diff}/{len(both)}")

    out_path = cfg.DATA / "processed" / "diag_splits_rebuild_dedup_comparison.csv"
    union_cells.to_csv(out_path, index=False)
    print(f"\nGespeichert (nur zur Dokumentation, ersetzt data/splits/splits_london.csv NICHT): {out_path}")


if __name__ == "__main__":
    main()
