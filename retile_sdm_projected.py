"""
retile_sdm_projected.py

Bugfix-Nachtrag zu Phase 4 (siehe build_sdm_projected_field.py-Kommentar und MANIFEST.md):
die urspruenglich geschriebenen Phase-4-GeoTIFFs (data/interim/sdm_projected/*/*.tif) waren
ungetilte Streifen mit interleave='pixel' (GDAL-Default) - fuer exact_extract() (Phase 5)
katastrophal langsam, weil jede Polygonabfrage trotz "nur Band 1 noetig" alle 256 Baender pro
Zeile dekomprimieren musste. Werte sind korrekt, nur das Speicherlayout ist falsch - dieses
Skript konvertiert verlustfrei (read+rewrite mit neuem Profil: tiled=True, blockxsize/
blockysize=256, interleave='band'), OHNE die Projektion neu zu rechnen.

Zeilenweise Chunks beim Lesen/Schreiben (dieselbe Vorsicht wie in build_sdm_projected_field.py -
die AlphaEarth-Dateien sind einzeln bis zu 19 GB gross).
"""
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

import config as cfg

PROJECTED_DIR = cfg.DATA / "interim" / "sdm_projected"
ROW_CHUNK = 500


def retile_one(path: Path) -> None:
    tmp_path = path.with_suffix(".tmp.tif")
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        if profile.get("tiled") and profile.get("interleave", "").lower() == "band":
            print(f"  {path.name}: bereits tiled+band-interleave, uebersprungen")
            return
        h, w = src.height, src.width
        profile.update(tiled=True, blockxsize=256, blockysize=256, interleave="band",
                        BIGTIFF="YES")
        t0 = time.time()
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for r0 in range(0, h, ROW_CHUNK):
                r1 = min(r0 + ROW_CHUNK, h)
                chunk = src.read(window=Window(0, r0, w, r1 - r0))
                dst.write(chunk, window=Window(0, r0, w, r1 - r0))
    tmp_path.replace(path)
    print(f"  {path.name}: retiled in {time.time() - t0:.1f}s")


def main():
    files = sorted(PROJECTED_DIR.glob("*/*.tif"))
    print(f"{len(files)} Dateien gefunden")
    for i, p in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {p}")
        retile_one(p)


if __name__ == "__main__":
    main()
