"""
build_greencover_valid.py

Phase-0-Retrieval-Evaluation, Nachtrag 2026-09-23 (Jakob): repariert die Green-Cover-2024-
Referenz (509 von 519 Polygonen topologisch ungueltig, siehe MANIFEST.md) EINMALIG mit
shapely.make_valid() und speichert das Ergebnis, damit die Reparatur nicht bei jedem
Rasterisierungslauf neu passieren muss. Liest weiterhin nur aus dem Original-Zip
(data/raw/greencover/Green_Cover_2024.zip, unveraendert, nicht entpackt).
"""
import sys
sys.path.insert(0, '.')
import zipfile
import geopandas as gpd
import pandas as pd
import shapely

import config as cfg

ZIP_PATH = cfg.DATA / "raw" / "greencover" / "Green_Cover_2024.zip"
OUT_PATH = cfg.DATA / "interim" / "greencover_2024_valid.parquet"


def main():
    with zipfile.ZipFile(ZIP_PATH) as z:
        shp_files = sorted(n for n in z.namelist() if n.endswith(".shp"))
    print(f"{len(shp_files)} Shapefile-Kacheln im Archiv")

    parts = [gpd.read_file(f"/vsizip/{ZIP_PATH}/{s}") for s in shp_files]
    gc = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:27700")
    n_before = len(gc)
    n_invalid = int((~gc.geometry.is_valid).sum())
    print(f"{n_before} Features geladen, {n_invalid} davon topologisch ungueltig "
          f"({100 * n_invalid / n_before:.1f}%)")

    gc["geometry"] = shapely.make_valid(gc.geometry.to_numpy())
    n_still_invalid = int((~gc.geometry.is_valid).sum())
    assert n_still_invalid == 0, f"{n_still_invalid} Geometrien nach make_valid() weiterhin ungueltig."
    print("Nach make_valid(): 0 ungueltig.")

    gc["ref_id"] = range(len(gc))
    gc = gc[["ref_id", "tile_num", "geometry"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gc.to_parquet(OUT_PATH)
    print(f"Gespeichert: {OUT_PATH} ({len(gc)} Zeilen)")

    check = gpd.read_parquet(OUT_PATH)
    assert len(check) == n_before
    assert (~check.geometry.is_valid).sum() == 0
    assert str(check.crs) == "EPSG:27700"
    print("Rueck-Lese-Kontrolle OK: Zeilenzahl, CRS und Gueltigkeit stimmen.")


if __name__ == "__main__":
    main()
