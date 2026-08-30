import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
import numpy as np
from geotessera import GeoTessera

import config as cfg

OUT = "data/ee/tessera_london.tif"


def default_bbox() -> tuple:
    """WGS84-BBox aus der London-AOI-Geometrie (data/aoi/london.gpkg, siehe src/extract_aoi.py)
    abgeleitet - keine hartcodierte Bounding Box mehr."""
    aoi_path = cfg.AOI_DIR / "london.gpkg"
    gdf = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return (minx, miny, maxx, maxy)


def progress(current, total, status):
    print(f"\r  Tessera-Download: {current}/{total} Kacheln ({status})", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Tessera-Mosaik für eine BBox als GeoTIFF speichern.")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                         metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                         help="Default: aus data/aoi/london.gpkg abgeleitet (siehe default_bbox())")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--crs", default="EPSG:32630", help="Ziel-CRS (default: UTM 30N für London)")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox is not None else default_bbox()
    gt = GeoTessera()

    n_tiles = gt.embeddings_count(bbox, year=args.year)
    print(f"BBox {bbox}: {n_tiles} Kacheln für Jahr {args.year}")

    arr, transform, crs = gt.fetch_mosaic_for_region(
        bbox=bbox, year=args.year, target_crs=args.crs, progress_callback=progress
    )
    print()

    # Achsenreihenfolge absichern: rasterio erwartet (bands, H, W)
    print("rohe shape:", arr.shape)
    if arr.shape[-1] < arr.shape[0]:      # (H, W, bands) -> umsortieren
        arr = np.transpose(arr, (2, 0, 1))
    print("shape für GeoTIFF:", arr.shape)

    nb, H, W = arr.shape
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff", height=H, width=W, count=nb,
                       dtype=arr.dtype, crs=crs, transform=transform, compress="lzw") as dst:
        dst.write(arr)
    print("Gespeichert:", out_path)


if __name__ == "__main__":
    main()