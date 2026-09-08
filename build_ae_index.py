"""
build_ae_index.py

Baut einen raeumlichen Index der AlphaEarth-Kacheln aus den COG-Headern (kein Bild-Read,
nur Metadaten - Laufzeit wenige Minuten). Die Kacheldateinamen sind opake Earth-Engine-Hashes
ohne Geoinformation, der Index muss also aus den COG-Bounds selbst kommen.

Kritisch (verifiziert an 30N): die Kacheln sind bottom-up geschrieben (y-Aufloesung
positiv), d.h. src.bounds.bottom > src.bounds.top ist moeglich. Bounds werden deshalb IMMER
ueber min()/max() normalisiert, nie direkt aus left/right/top/bottom uebernommen - sonst
entstehen invertierte Boxen und jeder raeumliche Filter liefert still null Treffer.
"""
import argparse
import concurrent.futures as cf
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

import config as cfg

ZONES_BY_REGION = {
    "london": ["30N", "31N"],          # 0-Meridian, siehe CLAUDE.md
    "niedersachsen": ["32N"],          # eine einzelne UTM-Zone, siehe CLAUDE.md
}
TILEDIR = cfg.DATA / "raw" / "alphaearth"


def header(gs_url: str) -> dict | None:
    url = gs_url.replace("gs://", "https://storage.googleapis.com/")
    try:
        with rasterio.open(url) as src:
            b = src.bounds
            return {
                "gs_url": gs_url,
                "crs": str(src.crs),
                "minx": min(b.left, b.right), "maxx": max(b.left, b.right),
                "miny": min(b.bottom, b.top), "maxy": max(b.bottom, b.top),
                "width": src.width, "height": src.height,
                "bands": src.count, "dtype": src.dtypes[0],
            }
    except Exception as e:
        print("FAIL", gs_url, e)
        return None


def build_zone_index(zone: str, tiledir: Path = TILEDIR, max_workers: int = 16) -> gpd.GeoDataFrame:
    urls = [l.strip() for l in open(tiledir / f"tiles_{zone}.txt") if l.strip()]
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = [r for r in ex.map(header, urls) if r]

    if not rows:
        raise RuntimeError(f"{zone}: kein einziger Header lesbar - {len(urls)} URLs versucht")

    gdf = gpd.GeoDataFrame(
        rows,
        geometry=[box(r["minx"], r["miny"], r["maxx"], r["maxy"]) for r in rows],
        crs=rows[0]["crs"],
    )
    n_fail = len(urls) - len(rows)
    print(f"{zone}: {len(gdf)}/{len(urls)} Kacheln indiziert (CRS={gdf.crs}), {n_fail} FAIL")
    return gdf


def main():
    parser = argparse.ArgumentParser(description="Raeumlicher Index der AlphaEarth-Kacheln je Region")
    parser.add_argument("--region", type=str, default="london", choices=sorted(ZONES_BY_REGION))
    args = parser.parse_args()

    for zone in ZONES_BY_REGION[args.region]:
        out = TILEDIR / f"ae_index_{zone}.gpkg"
        if out.exists():
            print(f"  {out} existiert bereits - ueberspringe (Index wird nicht neu gebaut).")
            continue
        gdf = build_zone_index(zone)
        gdf.to_file(out, driver="GPKG")
        print(f"  Gespeichert: {out}")


if __name__ == "__main__":
    main()
