"""
download_tessera_tiles.py

Paralleler TESSERA-Kachel-Downloader (Ersatz fuer den sequenziellen Weg ueber
GeoTessera.fetch_mosaic_for_region, der bei ~1.500 Kacheln auf >30h hochrechnet).
Nutzt dieselbe Kachelliste wie fetch_mosaic_for_region (ueber gt.registry), laedt aber
mit einem ThreadPoolExecutor parallel (Muster wie build_ae_index.py). Bereits
gecachte Kacheln werden uebersprungen (download_tile default refresh=False) - resume-faehig,
falls unterbrochen.

WICHTIG: cache_dir UND embeddings_dir muessen beide explizit gesetzt werden (siehe
make_tessera_tif.py) - sonst landen Kacheln im aktuellen Arbeitsverzeichnis statt im
Zielordner.
"""
import argparse
import concurrent.futures as cf
import time

import geopandas as gpd
from geotessera import GeoTessera

import config as cfg
from make_tessera_tif import REGION_CRS, region_bbox


def main():
    parser = argparse.ArgumentParser(description="Paralleler TESSERA-Kachel-Download je Region.")
    parser.add_argument("--region", type=str, default="niedersachsen", choices=sorted(REGION_CRS))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    bbox = region_bbox(args.region)
    cache_dir = args.cache_dir or str(cfg.DATA / "raw" / "tessera" / args.region)
    print(f"Region: {args.region}  BBox: {bbox}  Jahr: {args.year}  Cache/Embeddings-Dir: {cache_dir}")

    gt = GeoTessera(cache_dir=cache_dir, embeddings_dir=cache_dir)

    blocks = gt.registry.load_blocks_for_region(bbox, args.year)
    tiles = [(lon, lat) for (yr, lon, lat) in blocks]
    print(f"{len(tiles)} Kacheln fuer BBox {bbox}, Jahr {args.year}")

    t0 = time.time()
    done = 0
    failed = []

    def work(tile):
        lon, lat = tile
        ok = gt.download_tile(lon, lat, args.year)
        return tile, ok

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for tile, ok in ex.map(work, tiles):
            done += 1
            if not ok:
                failed.append(tile)
            if done % 20 == 0 or done == len(tiles):
                dt = time.time() - t0
                print(f"  {done}/{len(tiles)} ({dt:.0f}s, {done/dt*60:.1f} Kacheln/min)", flush=True)

    dt = time.time() - t0
    print(f"Fertig: {done - len(failed)}/{len(tiles)} ok in {dt/60:.1f} min, {len(failed)} FAIL")
    if failed:
        print("Fehlgeschlagene Kacheln:", failed)


if __name__ == "__main__":
    main()
