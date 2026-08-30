"""
extract_aoi.py

Extrahiert die Verwaltungsgrenze (AOI) fuer eine Studienregion aus dem lokalen OSM-PBF-Extrakt
(data/raw/osm/) statt einer hartcodierten Bounding Box - reproduzierbar, zitierbar, echtes
Polygon statt Rechteck (relevant fuer Kachelauswahl und Flaechenangaben).

Greater London: boundary=administrative, admin_level=5.
Niedersachsen:  boundary=administrative, admin_level=4.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))   # damit config gefunden wird

import geopandas as gpd
from pyrosm import OSM

import config as cfg

REGIONS = {
    "london": {
        "pbf": cfg.PBF_LONDON,
        "admin_level": "5",
        "name": "Greater London",
    },
    "niedersachsen": {
        "pbf": cfg.PBF_NIEDERSACHSEN,
        "admin_level": "4",
        "name": "Niedersachsen",
    },
}


def extract_aoi(pbf_path: Path, admin_level: str, name: str) -> gpd.GeoDataFrame:
    osm = OSM(str(pbf_path))
    boundaries = osm.get_boundaries(boundary_type="administrative", extra_attributes=["admin_level", "name"])
    match = boundaries[(boundaries["admin_level"] == admin_level) & (boundaries["name"] == name)]
    if len(match) != 1:
        raise ValueError(
            f"Erwartet genau 1 Treffer fuer name={name!r} admin_level={admin_level!r} in {pbf_path.name}, "
            f"gefunden: {len(match)}"
        )
    return match.to_crs("EPSG:4326")


def main():
    cfg.AOI_DIR.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        "# AOI-Quellen (siehe src/extract_aoi.py)",
        "#",
        "# Snapshot-Datum ist die Dateisystem-mtime der .osm.pbf-Datei (Download-Zeitpunkt),",
        "# NICHT der exakte OSM-Replication-Timestamp aus dem PBF-Header - dafuer war kein",
        "# osmium/pyosmium im eeclip-Env verfuegbar. Fuer die Thesis-Zitation ggf. den echten",
        "# Header-Timestamp separat nachtragen.",
        "",
    ]

    for key, spec in REGIONS.items():
        print(f"=== {key} ===")
        pbf_path = Path(spec["pbf"])
        gdf = extract_aoi(pbf_path, spec["admin_level"], spec["name"])

        area_km2 = gdf.to_crs(gdf.estimate_utm_crs()).area.iloc[0] / 1e6
        mtime = pd_mtime(pbf_path)
        print(f"  Quelle: {pbf_path.name}, admin_level={spec['admin_level']}, name={spec['name']!r}")
        print(f"  Datei-mtime (Näherung Snapshot-Datum): {mtime}")
        print(f"  Fläche: {area_km2:.1f} km²")

        out_path = cfg.AOI_DIR / f"{key}.gpkg"
        gdf.to_file(out_path, driver="GPKG")
        print(f"  Gespeichert: {out_path}")

        manifest_lines.append(
            f"{key}: {pbf_path.name} (mtime {mtime}), admin_level={spec['admin_level']}, "
            f"name={spec['name']!r}, area={area_km2:.1f} km²"
        )

    manifest_path = cfg.AOI_DIR / "SOURCES.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nManifest-Notiz gespeichert: {manifest_path}")


def pd_mtime(path: Path) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
