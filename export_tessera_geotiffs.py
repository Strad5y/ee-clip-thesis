"""
export_tessera_geotiffs.py

Wandelt bereits gecachte TESSERA-Kacheln (download_tessera_tiles.py - rohe quantisierte
.npy + .npy_scales) in georeferenzierte, dequantisierte GeoTIFFs um. Das ist ein separater
Schritt von download_tessera_tiles.py, weil GeoTessera.download_tile() NUR die Rohdaten
laedt - die Dequantisierung + GeoTIFF-Erzeugung uebernimmt export_embedding_geotiffs()
(reine CPU-Arbeit, kein Netzwerk, nutzt den lokalen Cache).

WICHTIG: export_embedding_geotiffs(tiles, output_dir=X) erzeugt selbst die Unterstruktur
X/global_0.1_degree_representation/{year}/grid_{lon}_{lat}/grid_{lon}_{lat}_{year}.tiff -
output_dir muss also der REGION-Ordner sein (z.B. data/raw/tessera/niedersachsen), NICHT
schon .../global_0.1_degree_representation/{year} (das haengt die Funktion selbst an).
Genau dieses Layout erwartet spatial_join.py::tessera_dir() (Glob "*/*.tiff").

Resume-faehig: bereits existierende {tile}_{year}.tiff werden uebersprungen (der Export
selbst braucht pro Kachel mehrere Sekunden CPU-Zeit, bei >1000 Kacheln lohnt sich das).
"""
import argparse

from geotessera import GeoTessera

import config as cfg
from make_tessera_tif import REGION_CRS, region_bbox


def main():
    parser = argparse.ArgumentParser(description="Dequantisierte GeoTIFFs aus gecachten TESSERA-Rohkacheln.")
    parser.add_argument("--region", type=str, default="niedersachsen", choices=sorted(REGION_CRS))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    bbox = region_bbox(args.region)
    cache_dir = args.cache_dir or str(cfg.DATA / "raw" / "tessera" / args.region)
    out_dir = cache_dir  # export_embedding_geotiffs haengt global_0.1_degree_representation/{year}/... selbst an

    gt = GeoTessera(cache_dir=cache_dir, embeddings_dir=cache_dir)
    blocks = gt.registry.load_blocks_for_region(bbox, args.year)
    print(f"{len(blocks)} Kacheln fuer {args.region}, Jahr {args.year}")

    rep_dir = cfg.DATA / "raw" / "tessera" / args.region / "global_0.1_degree_representation" / str(args.year)
    todo = []
    for (year, lon, lat) in blocks:
        tag = f"grid_{lon}_{lat}"
        tiff_path = rep_dir / tag / f"{tag}_{year}.tiff"
        if not tiff_path.exists():
            todo.append((year, lon, lat))
    print(f"  {len(blocks) - len(todo)} bereits vorhanden, {len(todo)} zu exportieren")

    if todo:
        done = 0

        def progress(current, total, status=""):
            nonlocal done
            done = current
            if current % 20 == 0 or current == total:
                print(f"  Export: {current}/{total} ({status})", flush=True)

        gt.export_embedding_geotiffs(todo, out_dir, progress_callback=progress)

    print("Fertig.")


if __name__ == "__main__":
    main()
