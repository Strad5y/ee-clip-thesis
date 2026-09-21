"""
check_splits_rebuild_dedup_ae.py

AlphaEarth-Dedup (2026-09-21, Jakob), Schritt 1: baut die 1-km-Split-Zellen (SEED=0, identischer
Algorithmus wie build_splits_pixels.py, assign_splits() importiert) auf dem Pixelbestand NACH
BEIDEN Deduplizierungen (pixels_tessera_dedup + pixels_ae_dedup) neu und vergleicht sie gegen
data/splits/splits_london.csv. Berichtet, LOEST NICHTS AUF, ueberschreibt splits_london.csv nicht.
Nachfolger von check_splits_rebuild_dedup.py (dort: tessera_dedup + UNVERAENDERTES pixels_ae).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd

import config as cfg
from aggregate_pixel_pairs import compute_pixel_cells
from build_splits import SEED, SPLIT_RATIOS, assign_splits
from check_splits_rebuild_dedup import CURRENT_SPLITS, get_aoi_utm_reference
from spatial_join import BACKBONES

PIXELS_PATHS = {
    "tessera": cfg.DATA / "processed" / "pixels_tessera_dedup.parquet",
    "ae": cfg.DATA / "processed" / "pixels_ae_dedup.parquet",
}


def main():
    utm_crs, minx, miny = get_aoi_utm_reference()
    print(f"AOI-UTM-CRS: {utm_crs}, minx={minx:.1f}, miny={miny:.1f}")

    per_backbone = {}
    for name, path in PIXELS_PATHS.items():
        px = pd.read_parquet(path, columns=["tile", "row", "col"])
        cells = compute_pixel_cells(px, BACKBONES[name], utm_crs, minx, miny)
        cells = cells[["cell_row", "cell_col"]].drop_duplicates()
        per_backbone[name] = cells
        print(f"  {name}: {len(px)} Pixel -> {len(cells)} Zellen")

    union_cells = pd.concat(per_backbone.values(), ignore_index=True).drop_duplicates().reset_index(drop=True)
    union_cells["cell_id"] = union_cells["cell_row"].astype(str) + "_" + union_cells["cell_col"].astype(str)
    print(f"\nVereinigung (neu, beide dedupliziert): {len(union_cells)} distinkte Zellen (SEED={SEED})")

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
    print(f"\nZellen nur in NEUER Vereinigung: {n_only_new}")
    print(f"Zellen nur in ALTER splits_london.csv: {n_only_old}")
    print(f"Zellen in BEIDEN mit ABWEICHENDEM Split: {n_diff}/{len(both)} "
          f"({100 * n_diff / len(both):.2f}%)")
    print(f"Zellen in BEIDEN mit GLEICHEM Split: {len(both) - n_diff}/{len(both)}")

    # Zellmengen je Backbone gegen die Referenz (welche Zellen fehlen wem?)
    ref = set(current["cell_id"])
    for name, cells in per_backbone.items():
        ids = set(cells["cell_row"].astype(str) + "_" + cells["cell_col"].astype(str))
        print(f"  {name}: {len(ids)} Zellen, davon nicht in splits_london.csv: {len(ids - ref)}, "
              f"in splits_london.csv aber ohne {name}-Pixel: {len(ref - ids)}")

    out_path = cfg.DATA / "processed" / "diag_splits_rebuild_dedup_ae_comparison.csv"
    union_cells.to_csv(out_path, index=False)
    print(f"\nGespeichert (nur Dokumentation, ersetzt splits_london.csv NICHT): {out_path}")


if __name__ == "__main__":
    main()
