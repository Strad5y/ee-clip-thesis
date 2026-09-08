"""
download_ae_tiles.py

Laedt die in needed_{region}.txt gelisteten AlphaEarth-Kacheln ueber den oeffentlichen
HTTPS-Spiegel (storage.googleapis.com) - kein gsutil/gcloud-Billing-Setup noetig, verifiziert
per direktem rasterio-Read (Header UND Pixel-Daten lesbar ohne Auth). Ueberspringt bereits
vorhandene Dateien (Resume-faehig).
"""
import argparse
import time
from pathlib import Path

import requests

import config as cfg


def gs_to_https(gs_url: str) -> str:
    return gs_url.replace("gs://", "https://storage.googleapis.com/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    args = parser.parse_args()

    # Region-eigenes Unterverzeichnis (siehe spatial_join.py ae_tiles_dir()) - ein flaches,
    # regionsloses Verzeichnis wuerde Kacheln verschiedener Regionen vermischen (Kachelnamen
    # sind opake Earth-Engine-Hashes ohne Regionsinfo im Dateinamen).
    tiledir = cfg.DATA / "raw" / "alphaearth" / "tiles" / args.region
    needed_path = cfg.DATA / "raw" / "alphaearth" / f"needed_{args.region}.txt"
    urls = [l.strip() for l in open(needed_path) if l.strip()]
    tiledir.mkdir(parents=True, exist_ok=True)
    print(f"{len(urls)} Kacheln fuer {args.region} zu laden nach {tiledir}")

    for i, gs_url in enumerate(urls, 1):
        fname = gs_url.rsplit("/", 1)[-1]
        out_path = tiledir / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [{i}/{len(urls)}] {fname}: existiert bereits ({out_path.stat().st_size/1e9:.2f} GB) - skip")
            continue
        url = gs_to_https(gs_url)
        t0 = time.time()
        tmp_path = out_path.with_suffix(".tiff.part")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        tmp_path.rename(out_path)
        dt = time.time() - t0
        size_gb = out_path.stat().st_size / 1e9
        print(f"  [{i}/{len(urls)}] {fname}: {size_gb:.2f} GB in {dt:.1f}s ({size_gb/dt*1000:.0f} MB/s)")

    print("Fertig.")


if __name__ == "__main__":
    main()
