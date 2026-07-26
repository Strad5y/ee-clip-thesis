"""
download_osm_extracts.py

Lädt die OSM-Rohdaten (.osm.pbf) für die beiden Studienregionen
von Geofabrik herunter. Wird noch NICHT automatisch ausgeführt -
nur zum Ablegen im Projekt, Aufruf erfolgt manuell später.
"""

import requests
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Region:
    name: str
    url: str


REGIONS = [
    Region(
        name="greater_london",
        url="https://download.geofabrik.de/europe/united-kingdom/england/greater-london-latest.osm.pbf",
    ),
    Region(
        name="niedersachsen",
        url="https://download.geofabrik.de/europe/germany/niedersachsen-latest.osm.pbf",
    ),
]

OUTPUT_DIR = Path("data/raw/osm")


def download_extract(region: Region, output_dir: Path, chunk_size: int = 1024 * 1024) -> Path:
    """
    Lädt eine .osm.pbf-Datei mit Streaming herunter (kein Laden in RAM),
    mit einfachem Fortschrittsbalken.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"{region.name}-latest.osm.pbf"

    if target_path.exists():
        print(f"{region.name}: Datei existiert bereits unter {target_path}, überspringe.")
        return target_path

    print(f"{region.name}: Starte Download von {region.url}")

    with requests.get(region.url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"\r{region.name}: {pct:.1f}% ({downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB)", end="")

    print(f"\n{region.name}: fertig, gespeichert unter {target_path}")
    return target_path


def main():
    for region in REGIONS:
        try:
            download_extract(region, OUTPUT_DIR)
        except requests.exceptions.RequestException as e:
            print(f"{region.name}: Fehler beim Download - {e}")


if __name__ == "__main__":
    main()