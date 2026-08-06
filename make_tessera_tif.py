import argparse
from pathlib import Path

import rasterio
import numpy as np
from geotessera import GeoTessera

# Greater London als Standard-BBox (w, s, e, n) — per --bbox überschreibbar,
# z.B. für einen Mini-Testlauf auf einer kleinen Teilregion.
BBOX = (-0.51, 51.28, 0.33, 51.69)
OUT = "data/ee/tessera_london.tif"


def progress(current, total, status):
    print(f"\r  Tessera-Download: {current}/{total} Kacheln ({status})", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Tessera-Mosaik für eine BBox als GeoTIFF speichern.")
    parser.add_argument("--bbox", type=float, nargs=4, default=list(BBOX),
                         metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--crs", default="EPSG:32630", help="Ziel-CRS (default: UTM 30N für London)")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    bbox = tuple(args.bbox)
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