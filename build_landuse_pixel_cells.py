"""
build_landuse_pixel_cells.py

Nachtrag zu INSTRUCTIONS_landuse_cell_bootstrap.md, Phase 2: persistiert die Zellzuordnung
(cell_row, cell_col) je Pixel in landuse_pairs_{ae,tessera}.parquet EINMAL, statt sie bei
jedem Bootstrap-Lauf transient neu zu berechnen - wird fuer die Retrieval-Auswertung wieder
gebraucht (Nutzervorgabe). Mechanismus identisch zu diag_landuse_distance_accuracy.py::
compute_world_xy() (Kachel-Affintransformation je Pixel, Reprojektion 31N->30N) und zur
Zellformel aus build_splits.py/build_splits_pixels.py (cell_col=floor((x-minx)/1000),
cell_row=floor((y-miny)/1000), Ursprung aus der AOI-UTM-CRS).

WICHTIG: landuse_pairs_{backbone}.parquet selbst wird NICHT veraendert (keine neue Spalte
reingeschrieben) - das koennte die von verify_provenance.py bestaetigte Split-Konsistenz-
Pruefung stoeren bzw. den Dateihash aendern, auf den sich MANIFEST.md-Eintraege beziehen.
Stattdessen ein separates, ueber (tile,row,col) joinbares Begleitartefakt.
"""
import numpy as np
import pandas as pd

import config as cfg
from build_sdm_lsoa_splits import CELL_SIZE_M, get_aoi_utm_reference
from diag_landuse_distance_accuracy import compute_world_xy

OUT_TEMPLATE = "data/processed/landuse_pixel_cells_{name}.parquet"


def build_for_backbone(name: str) -> pd.DataFrame:
    path = cfg.DATA / "processed" / f"landuse_pairs_{name}.parquet"
    df = pd.read_parquet(path, columns=["tile", "row", "col"])
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  {name}: {n_before} Zeilen -> {len(df)} eindeutige (tile,row,col)-Pixel")

    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    xy = compute_world_xy(df, name)
    n_nan = int(np.isnan(xy).any(axis=1).sum())
    if n_nan > 0:
        print(f"    WARNUNG: {n_nan} Pixel ohne Weltkoordinate (Kachel nicht gefunden?) - "
              f"deren Zellzuordnung wird NaN.")

    cell_col = np.floor((xy[:, 0] - minx) / CELL_SIZE_M)
    cell_row = np.floor((xy[:, 1] - miny) / CELL_SIZE_M)
    df["cell_row"] = cell_row
    df["cell_col"] = cell_col
    # Int64 (nullable) statt int, falls NaN vorkommen sollten
    df["cell_row"] = df["cell_row"].astype("Int64")
    df["cell_col"] = df["cell_col"].astype("Int64")
    return df


def main():
    for name in ["ae", "tessera"]:
        print(f"\n=== {name} ===")
        df = build_for_backbone(name)
        out_path = cfg.ROOT / OUT_TEMPLATE.format(name=name)
        df.to_parquet(out_path, index=False)
        print(f"  Gespeichert: {out_path} ({len(df)} Zeilen, "
              f"{df[['cell_row','cell_col']].drop_duplicates().shape[0]} eindeutige Zellen)")


if __name__ == "__main__":
    main()
