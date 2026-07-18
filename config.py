from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
YEAR = 2024

# Kleine Testregion in Central London
BBOX_TEST = (-0.12, 51.50, -0.08, 51.53)

PBF_LONDON = DATA / "raw" / "greater-london-260710.osm.pbf"
TESSERA_DIR = DATA / "raw" / "tessera_london_test" / "global_0.1_degree_representation" / str(YEAR)
OSM_KEYS = ["amenity", "shop", "tourism", "leisure", "landuse"]
CROSSWALK = ROOT / "crosswalk" / "osm_hilucs_crosswalk.csv"