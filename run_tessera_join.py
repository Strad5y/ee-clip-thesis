"""
run_tessera_join.py

Robuster Vollauf: TESSERA x volle POI-Menge (London, 279.988 eindeutige poi_id). Duenner
Wrapper um chunked_join_runner.run() - siehe dort fuer Chunking/Resume/Logging/Merge-Logik.

Abweichungen ggue. AlphaEarth (siehe run_ae_join.py): 50 Kacheln statt 2, keine
Dequantisierung (DEQUANTIZE['tessera'] ist None - TESSERA-Kacheln sind bereits float32),
keine Z-Score-Normalisierung hier (die passiert erst spaeter in train.py auf dem
Trainings-Split, nicht im Join).
"""
import argparse

import config as cfg
from chunked_join_runner import run
from spatial_join import BACKBONES, DEQUANTIZE

CHUNK_SIZE = 20000
CHUNK_DIR = cfg.DATA / "interim" / "pairs_tessera_chunks"
OUT_PATH = cfg.DATA / "processed" / "pairs_tessera.parquet"


def main():
    parser = argparse.ArgumentParser(description="Robuster TESSERA-Vollauf (London)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    run("tessera", BACKBONES["tessera"], DEQUANTIZE["tessera"], CHUNK_DIR, OUT_PATH,
        chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
