"""
run_ae_join.py

Robuster Vollauf: AlphaEarth x volle POI-Menge (London, 279.988 eindeutige poi_id). Duenner
Wrapper um chunked_join_runner.run() - siehe dort fuer Chunking/Resume/Logging/Merge-Logik.
"""
import argparse

import config as cfg
from chunked_join_runner import run
from spatial_join import BACKBONES, DEQUANTIZE

CHUNK_SIZE = 20000
CHUNK_DIR = cfg.DATA / "interim" / "pairs_ae_chunks"
OUT_PATH = cfg.DATA / "processed" / "pairs_ae.parquet"


def main():
    parser = argparse.ArgumentParser(description="Robuster AlphaEarth-Vollauf (London)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    run("ae", BACKBONES["ae"], DEQUANTIZE["ae"], CHUNK_DIR, OUT_PATH, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
