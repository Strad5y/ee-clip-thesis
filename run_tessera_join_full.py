"""
run_tessera_join_full.py

Validierungslauf Phase 3 (CLAUDE_CODE_TASKS_umbau.md): TESSERA x volle POI-Menge mit dem
13-Key-Satz - NICHT die produktive pairs_tessera.parquet ueberschreiben, solange dieser Lauf
nicht validiert ist (Pixeldichte-Check). Eigener chunk_dir/out_path/pois_in je Region, damit
nichts mit dem alten 4-Key-Lauf kollidiert.

Nutzt join_pois_to_tiles_by_tile() (Kachel-Schleife AUSSEN), NICHT chunked_join_runner.run()
(POI-Block-Schleife aussen, wie run_ae_join_full.py) - TESSERA hat viele kleine Kacheln (z.B.
1474 fuer Niedersachsen), bei blockweisem Aufruf wuerde jede Kachel einmal PRO POI-BLOCK
geoeffnet (gemessene Hochrechnung: >16h). AlphaEarth hat dagegen wenige riesige Kacheln, dort
bleibt der blockweise Ansatz (chunked_join_runner.run()) richtig - siehe run_ae_join_full.py.
Beide Ansaetze liefern nachweislich (spatial_join.py::_smoke_test_by_tile_equivalence)
dieselbe Paarmenge, nur die Durchlaufreihenfolge unterscheidet sich.
"""
import argparse
import time

import geopandas as gpd

import config as cfg
from spatial_join import DEQUANTIZE, ID_COL, POIS_IN_BY_REGION, ZONES_BY_REGION, get_backbones, join_pois_to_tiles_by_tile


def main():
    parser = argparse.ArgumentParser(description="Validierungslauf TESSERA (13-Key-Satz)")
    parser.add_argument("--region", type=str, default="london", choices=sorted(ZONES_BY_REGION))
    args = parser.parse_args()

    suffix = "" if args.region == "london" else f"_{args.region}"
    chunk_dir = cfg.DATA / "interim" / f"pairs_tessera_full{suffix}_chunks"
    out_path = cfg.DATA / "processed" / f"pairs_tessera_full{suffix}.parquet"
    pois_in = POIS_IN_BY_REGION[args.region]
    backbones = get_backbones(args.region)
    tile_paths = backbones["tessera"]

    print(f"=== tessera_full-Join gestartet (Region: {args.region}) ===")
    print(f"Kacheln: {len(tile_paths)}")
    print(f"Lese POIs: {pois_in}")
    gdf = gpd.read_parquet(pois_in)
    n_raw = len(gdf)
    gdf = gdf.drop_duplicates(subset=ID_COL).reset_index(drop=True)
    print(f"  {n_raw} Tag-Zeilen -> {len(gdf)} eindeutige POI-Geometrien")

    chunk_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs = join_pois_to_tiles_by_tile(gdf, tile_paths, dequantize=DEQUANTIZE["tessera"],
                                        zones=ZONES_BY_REGION[args.region], chunk_dir=chunk_dir)
    dt = time.time() - t0
    print(f"Join fertig in {dt / 60:.1f} min")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path)
    print(f"Gespeichert: {out_path}")
    print(f"  Zeilen: {len(pairs)}")
    print(f"  Eindeutige poi_id: {pairs[ID_COL].nunique()}")
    print("=== tessera_full-Join fertig ===")


if __name__ == "__main__":
    main()
