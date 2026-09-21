"""
build_landuse_pixel_cells_ae_dedup.py

Schritt 3 AlphaEarth (2026-09-21, Jakob): Begleitartefakt fuer den Zellebenen-Bootstrap,
AE-Variante von build_landuse_pixel_cells_tessera_dedup.py (bleibt unveraendert): Zellzuordnung
je Pixel von landuse_pairs_ae_dedup.parquet, gleiche Zellformel wie das Original
(compute_world_xy + CELL_SIZE_M), Kachelgeometrie des Backbones 'ae'.
"""
import numpy as np
import pandas as pd

import config as cfg
from build_sdm_lsoa_splits import CELL_SIZE_M, get_aoi_utm_reference
from diag_landuse_distance_accuracy import compute_world_xy

DATA_NAME = "ae_dedup"       # -> landuse_pairs_ae_dedup.parquet
TILE_BACKBONE = "ae"         # fuer die Kachel-Geometrie-Lookups (compute_world_xy)
IN_PATH = cfg.DATA / "processed" / f"landuse_pairs_{DATA_NAME}.parquet"
OUT_PATH = cfg.DATA / "processed" / f"landuse_pixel_cells_{DATA_NAME}.parquet"


def main():
    df = pd.read_parquet(IN_PATH, columns=["tile", "row", "col"])
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"{n_before} Zeilen -> {len(df)} eindeutige (tile,row,col)-Pixel")

    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    xy = compute_world_xy(df, TILE_BACKBONE)
    n_nan = int(np.isnan(xy).any(axis=1).sum())
    if n_nan > 0:
        print(f"  WARNUNG: {n_nan} Pixel ohne Weltkoordinate - deren Zellzuordnung wird NaN.")

    df["cell_row"] = np.floor((xy[:, 0] - minx) / CELL_SIZE_M)
    df["cell_col"] = np.floor((xy[:, 1] - miny) / CELL_SIZE_M)
    df["cell_row"] = df["cell_row"].astype("Int64")
    df["cell_col"] = df["cell_col"].astype("Int64")

    df.to_parquet(OUT_PATH, index=False)
    print(f"Gespeichert: {OUT_PATH} ({len(df)} Zeilen, "
          f"{df[['cell_row', 'cell_col']].drop_duplicates().shape[0]} eindeutige Zellen)")


if __name__ == "__main__":
    main()
