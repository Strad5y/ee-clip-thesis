"""
spatial_join.py

Kern-Join: POI-Geometrie x EE-Pixel -> Trainingspaare.
Settled Decision (CLAUDE.md): jedes EE-Pixel, das eine POI-Geometrie schneidet, wird ein
Trainingspaar; coverage in [0,1] ist der Flaechenanteil des Pixels, den die Geometrie abdeckt
(Punkte zaehlen immer als coverage=1.0).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract

import config as cfg

ID_COL = "poi_id"
CARRY = []  # zusaetzliche POI-Spalten fuer den Output; bewusst leer -
            # Tag-Verknuepfung (key/value) laeuft spaeter separat ueber poi_id,
            # sonst wuerde das Long-Format (mehrere Tags je poi_id) die Pixel-Paare vervielfachen

POIS_IN = cfg.DATA / "interim" / "pois_london_test_filtered.parquet"
BACKBONES = {
    "tessera": cfg.DATA / "ee" / "tessera_london.tif",
    "ae": cfg.DATA / "ee" / "alphaearth_london.tif",  # noch nicht erzeugt
}
OUT_DIR = cfg.DATA / "processed"


def _join_points(points: gpd.GeoDataFrame, raster_path, id_col, carry) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        xs = points.geometry.x.to_numpy()
        ys = points.geometry.y.to_numpy()
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        in_bounds = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        n_out = int((~in_bounds).sum())
        if n_out > 0:
            print(f"  Points: {n_out} außerhalb der Rasterbounds verworfen")

        out = pd.DataFrame({
            id_col: points[id_col].to_numpy()[in_bounds],
            "row": rows[in_bounds],
            "col": cols[in_bounds],
            "coverage": 1.0,
        })
        for c in carry:
            out[c] = points[c].to_numpy()[in_bounds]
        return out


def _join_polygons(polys: gpd.GeoDataFrame, raster_path, id_col, carry) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        width = src.width

    cols_to_include = [id_col] + list(carry)
    res = exact_extract(
        str(raster_path), polys, ["cell_id", "coverage"],
        include_cols=cols_to_include, output="pandas",
    )

    # cell_id/coverage sind pro Band identisch (gleiches Grid) -> Band 1 genuegt
    records = []
    for _, row in res.iterrows():
        cell_ids = np.asarray(row["band_1_cell_id"])
        coverages = np.asarray(row["band_1_coverage"])
        if len(cell_ids) == 0:
            continue
        rows_ = cell_ids // width
        cols_ = cell_ids % width
        n = len(cell_ids)
        rec = {
            id_col: np.full(n, row[id_col]),
            "row": rows_,
            "col": cols_,
            "coverage": coverages,
        }
        for c in carry:
            rec[c] = np.full(n, row[c])
        records.append(pd.DataFrame(rec))

    if not records:
        return pd.DataFrame(columns=[id_col, "row", "col", "coverage", *carry])
    return pd.concat(records, ignore_index=True)


def join_pois_to_raster(gdf: gpd.GeoDataFrame, raster_path, id_col=ID_COL, carry=CARRY) -> pd.DataFrame:
    """POIs x Raster -> DataFrame mit einer Zeile pro (poi, geschnittenem Pixel)."""
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        n_bands = src.count

    gdf = gdf.to_crs(raster_crs)

    points = gdf[gdf.geometry.geom_type == "Point"]
    polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    parts = []
    if len(points) > 0:
        parts.append(_join_points(points, raster_path, id_col, carry))
    if len(polys) > 0:
        parts.append(_join_polygons(polys, raster_path, id_col, carry))

    pairs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(pairs) == 0:
        return pairs

    # Embedding-Werte je Pixel nachschlagen (ein voller Array-Read; fuer die Mini-Region
    # vertretbar, fuer grosse Regionen spaeter durch windowed reads ersetzen)
    with rasterio.open(raster_path) as src:
        arr = src.read()  # (bands, H, W)

    band_vals = arr[:, pairs["row"].to_numpy(), pairs["col"].to_numpy()]  # (bands, n_pairs)
    pairs["embedding"] = list(band_vals.T.astype(np.float32))
    pairs.attrs["n_bands"] = n_bands
    return pairs


def main():
    print(f"Lese POIs: {POIS_IN}")
    gdf = gpd.read_parquet(POIS_IN)
    print(f"  {len(gdf)} POIs, Typen: {dict(gdf.geometry.geom_type.value_counts())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, raster_path in BACKBONES.items():
        print(f"\n=== Backbone: {name} ===")
        if not raster_path.exists():
            print(f"  Raster nicht vorhanden ({raster_path}) - übersprungen")
            continue

        pairs = join_pois_to_raster(gdf, raster_path)
        if len(pairs) == 0:
            print("  Keine Paare erzeugt.")
            continue

        emb_dim = len(pairs["embedding"].iloc[0])
        n_unique_pois = pairs[ID_COL].nunique()
        cov = pairs["coverage"]
        print(f"  Paare: {len(pairs)}  (aus {n_unique_pois} von {len(gdf)} POIs)")
        print(f"  Embedding-Dim: {emb_dim}")
        print(f"  Coverage: min={cov.min():.4f} max={cov.max():.4f} mean={cov.mean():.4f}")
        print(f"  NaN in embeddings: {sum(np.isnan(e).any() for e in pairs['embedding'])}")

        out_path = OUT_DIR / f"pairs_{name}.parquet"
        pairs.to_parquet(out_path)
        print(f"  Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
