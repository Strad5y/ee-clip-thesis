"""
run_ae_join_full.py

Validierungslauf Phase 3 (CLAUDE_CODE_TASKS_umbau.md): AlphaEarth x volle POI-Menge mit dem
13-Key-Satz - analog run_tessera_join_full.py, siehe dort fuer Begruendung der separaten
Pfade (nichts am alten 4-Key-Lauf/production pairs_ae.parquet aendern, bis validiert).
"""
import argparse

import config as cfg
from chunked_join_runner import run
from spatial_join import DEQUANTIZE, POIS_IN_BY_REGION, ZONES_BY_REGION, get_backbones

CHUNK_SIZE = 20000


def main():
    parser = argparse.ArgumentParser(description="Validierungslauf AlphaEarth (13-Key-Satz)")
    parser.add_argument("--region", type=str, default="london", choices=sorted(ZONES_BY_REGION))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    # London behaelt die alten, unsuffixierten Pfade (Produktionsdateien) - andere Regionen
    # bekommen einen Suffix, damit nichts kollidiert.
    suffix = "" if args.region == "london" else f"_{args.region}"
    chunk_dir = cfg.DATA / "interim" / f"pairs_ae_full{suffix}_chunks"
    out_path = cfg.DATA / "processed" / f"pairs_ae_full{suffix}.parquet"
    pois_in = POIS_IN_BY_REGION[args.region]
    backbones = get_backbones(args.region)

    run("ae_full", backbones["ae"], DEQUANTIZE["ae"], chunk_dir, out_path,
        chunk_size=args.chunk_size, pois_in=pois_in, zones=ZONES_BY_REGION[args.region])


if __name__ == "__main__":
    main()
