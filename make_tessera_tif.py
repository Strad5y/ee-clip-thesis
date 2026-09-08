import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
import numpy as np
from geotessera import GeoTessera

import config as cfg

OUT = "data/ee/tessera_london.tif"

REGION_CRS = {"london": "EPSG:32630", "niedersachsen": "EPSG:32632"}  # CLAUDE.md: Niedersachsen = 32N


def region_bbox(region: str) -> tuple:
    """WGS84-BBox aus der Region-AOI-Geometrie (data/aoi/{region}.gpkg, siehe src/extract_aoi.py)
    abgeleitet - keine hartcodierte Bounding Box."""
    aoi_path = cfg.AOI_DIR / f"{region}.gpkg"
    gdf = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return (minx, miny, maxx, maxy)


def progress(current, total, status):
    print(f"\r  Tessera-Download: {current}/{total} Kacheln ({status})", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Tessera-Mosaik für eine Region als GeoTIFF speichern.")
    parser.add_argument("--region", type=str, default="london", choices=sorted(REGION_CRS),
                         help="AOI aus data/aoi/{region}.gpkg, Ziel-CRS aus REGION_CRS.")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                         metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                         help="Ueberschreibt die aus --region abgeleitete BBox.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--crs", default=None, help="Ziel-CRS (default: aus REGION_CRS[--region]).")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cache-dir", default=None,
                         help="geotessera-Cache-Verzeichnis (default: data/raw/tessera/{region}) - "
                              "hier landen die einzelnen Kacheln, die spatial_join.py liest.")
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox is not None else region_bbox(args.region)
    crs = args.crs or REGION_CRS[args.region]
    out = args.out or f"data/ee/tessera_{args.region}.tif"
    cache_dir = args.cache_dir or str(cfg.DATA / "raw" / "tessera" / args.region)

    print(f"Region: {args.region}  BBox: {bbox}  CRS: {crs}  Cache: {cache_dir}")
    # cache_dir und embeddings_dir sind ZWEI GETRENNTE GeoTessera-Parameter - cache_dir
    # betrifft nur Registry-Metadaten (v1/manifest.parquet etc.), die eigentlichen
    # heruntergeladenen Kacheln landen in embeddings_dir, das ohne explizite Angabe auf
    # das aktuelle Arbeitsverzeichnis faellt (nicht auf cache_dir!). Beide explizit
    # gleich setzen, sonst landen Kacheln unbemerkt im Projekt-Root.
    gt = GeoTessera(cache_dir=cache_dir, embeddings_dir=cache_dir)

    n_tiles = gt.embeddings_count(bbox, year=args.year)
    print(f"BBox {bbox}: {n_tiles} Kacheln für Jahr {args.year}")

    arr, transform, out_crs = gt.fetch_mosaic_for_region(
        bbox=bbox, year=args.year, target_crs=crs, progress_callback=progress
    )
    print()

    # Achsenreihenfolge absichern: rasterio erwartet (bands, H, W)
    print("rohe shape:", arr.shape)
    if arr.shape[-1] < arr.shape[0]:      # (H, W, bands) -> umsortieren
        arr = np.transpose(arr, (2, 0, 1))
    print("shape für GeoTIFF:", arr.shape)

    nb, H, W = arr.shape
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff", height=H, width=W, count=nb,
                       dtype=arr.dtype, crs=out_crs, transform=transform, compress="lzw") as dst:
        dst.write(arr)
    print("Gespeichert:", out_path)


if __name__ == "__main__":
    main()
