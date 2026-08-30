"""
build_splits.py

Raeumlich geblockter Split gegen Leakage (F1, ersetzt den frueheren zeilenweisen
Zufallssplit in train.py): Gitter aus quadratischen Zellen ueber der AOI, Zellen werden
zufaellig auf train/val/test verteilt, POIs folgen ihrer Zelle - garantiert automatisch,
dass alle Pixel-Zeilen eines poi_id im selben Split landen (kein Split auf Zeilenebene
mehr, siehe splits.py fuer die Konsum-Seite).

Wird EINMAL ausgefuehrt, das Ergebnis ist ein versioniertes Artefakt unter data/splits/ -
nicht bei jedem Trainingslauf neu gewuerfelt.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import config as cfg

CELL_SIZE_M = 1000.0  # Startwert (1 km Kantenlaenge), per --cell-size ueberschreibbar
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 0

POIS_IN = cfg.DATA / "interim" / "pois_london_test_filtered.parquet"
AOI_IN = cfg.AOI_DIR / "london.gpkg"
OUT = cfg.DATA / "splits" / "splits_london.csv"


def assign_cells(gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, cell_size_m: float) -> pd.DataFrame:
    """Reprojiziert auf das metrische UTM-CRS der AOI, ordnet jedem POI (ueber einen
    Repraesentativpunkt) eine Gitterzelle (cell_row, cell_col) per Ganzzahl-Division zu -
    kein tatsaechliches Polygon-Gitter noetig, nur die Zellindizes."""
    utm_crs = aoi.estimate_utm_crs()
    aoi_utm = aoi.to_crs(utm_crs)
    gdf_utm = gdf.to_crs(utm_crs)

    minx, miny, _, _ = aoi_utm.total_bounds
    points = gdf_utm.geometry.representative_point()  # bleibt garantiert im Polygon (anders als centroid)

    cell_col = np.floor((points.x.to_numpy() - minx) / cell_size_m).astype(int)
    cell_row = np.floor((points.y.to_numpy() - miny) / cell_size_m).astype(int)

    out = gdf[["poi_id"]].copy()
    out["cell_row"] = cell_row
    out["cell_col"] = cell_col
    out["cell_id"] = [f"{r}_{c}" for r, c in zip(cell_row, cell_col)]
    return out


def assign_splits(cell_ids: np.ndarray, ratios: dict, seed: int) -> dict:
    """Gitterzellen (NICHT einzelne POIs) zufaellig auf train/val/test verteilen -
    das ist der eigentliche raeumliche Block; POIs erben den Split ihrer Zelle."""
    unique_cells = np.unique(cell_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_cells)

    n = len(shuffled)
    n_train = int(round(ratios["train"] * n))
    n_val = int(round(ratios["val"] * n))
    train_cells = shuffled[:n_train]
    val_cells = shuffled[n_train:n_train + n_val]
    test_cells = shuffled[n_train + n_val:]  # Rest, faengt Rundungsfehler auf

    cell_to_split = {}
    cell_to_split.update({c: "train" for c in train_cells})
    cell_to_split.update({c: "val" for c in val_cells})
    cell_to_split.update({c: "test" for c in test_cells})
    return cell_to_split


def build(gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, cell_size_m: float, seed: int) -> pd.DataFrame:
    cells = assign_cells(gdf, aoi, cell_size_m)
    cell_to_split = assign_splits(cells["cell_id"].to_numpy(), SPLIT_RATIOS, seed)
    cells["split"] = cells["cell_id"].map(cell_to_split)
    return cells


def verify_no_leakage(cells: pd.DataFrame) -> None:
    """Explizite Abnahme (F1): Schnittmenge der poi_id zwischen den Splits muss leer sein."""
    sets = {s: set(cells.loc[cells["split"] == s, "poi_id"]) for s in ["train", "val", "test"]}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        overlap = sets[a] & sets[b]
        status = "OK" if not overlap else "FEHLER"
        print(f"  Verifikation poi_id-Schnittmenge {a}/{b}: {len(overlap)} ({status})")
        assert not overlap, f"Split-Leakage zwischen {a} und {b}: {len(overlap)} poi_id betroffen"


def main():
    parser = argparse.ArgumentParser(description="Raeumlich geblockten POI-Split ueber Gitterzellen bauen.")
    parser.add_argument("--pois", default=str(POIS_IN))
    parser.add_argument("--aoi", default=str(AOI_IN))
    parser.add_argument("--cell-size", type=float, default=CELL_SIZE_M, help="Zellkantenlaenge in Metern")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    gdf = gpd.read_parquet(args.pois)
    gdf = gdf.drop_duplicates(subset="poi_id").reset_index(drop=True)  # Long-Format -> eindeutige Geometrien
    aoi = gpd.read_file(args.aoi)

    cells = build(gdf, aoi, args.cell_size, args.seed)

    print(f"{len(cells)} POIs, {cells['cell_id'].nunique()} Gitterzellen "
          f"(Kantenlaenge {args.cell_size:.0f} m)")
    print(cells.groupby("split")["poi_id"].nunique())
    print()
    verify_no_leakage(cells)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cells[["poi_id", "cell_row", "cell_col", "split"]].to_csv(out_path, index=False)
    print(f"\nGespeichert: {out_path}")


def _smoke_test():
    """Synthetische POIs (Zufallspunkte) statt echter Daten - pois_london_test_filtered.parquet
    existiert noch nicht (Paarbau ist in dieser Session bewusst nicht gestartet)."""
    from shapely.geometry import Point, box

    rng = np.random.default_rng(0)
    n = 500
    # 5km x 5km Testflaeche in EPSG:4326 um einen Punkt in London, genug Zellen bei 1km Gitter
    minx, miny = -0.10, 51.50
    xs = minx + rng.random(n) * 0.07   # ca. 5 km in Laengengrad-Naeherung
    ys = miny + rng.random(n) * 0.045  # ca. 5 km in Breitengrad-Naeherung

    gdf = gpd.GeoDataFrame(
        {"poi_id": [f"node/{i}" for i in range(n)]},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs="EPSG:4326",
    )
    aoi = gpd.GeoDataFrame(geometry=[box(minx, miny, minx + 0.07, miny + 0.045)], crs="EPSG:4326")

    cells = build(gdf, aoi, cell_size_m=1000.0, seed=0)
    assert cells["cell_id"].nunique() > 1, "Testflaeche zu klein fuer mehr als eine Gitterzelle"
    verify_no_leakage(cells)

    counts = cells["split"].value_counts(normalize=True)
    print(f"\nSmoke-Test OK: {len(cells)} synthetische POIs, {cells['cell_id'].nunique()} Zellen, "
          f"Split-Anteile ~{dict(counts.round(2))}, keine poi_id-Ueberlappung zwischen den Splits.")


if __name__ == "__main__":
    import sys
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        main()
