from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
YEAR = 2025

# Testartefakt: nur fuer kleine Mini-Testlaeufe (z.B. src/extract_pois.py), NICHT fuer
# echte Region-Downloads verwenden - dafuer die AOI-Geometrien aus data/aoi/ (siehe
# src/extract_aoi.py).
BBOX_TEST = (-0.12, 51.50, -0.08, 51.53)

PBF_LONDON = DATA / "raw" / "osm" / "greater_london-latest.osm.pbf"
PBF_NIEDERSACHSEN = DATA / "raw" / "osm" / "niedersachsen-latest.osm.pbf"
AOI_DIR = DATA / "aoi"
TESSERA_DIR = DATA / "raw" / "tessera_london_test" / "global_0.1_degree_representation" / str(YEAR)
OSM_KEYS = ["amenity", "shop", "tourism", "leisure"]  # landuse bewusst nicht aktiv - HILUCS-
                                                        # Crosswalk und sozioökonomische Kartierung
                                                        # sind gestrichen, Eval-Ground-Truth ist
                                                        # Copernicus Urban Atlas
CROSSWALK = ROOT / "crosswalk" / "osm_hilucs_crosswalk.csv"