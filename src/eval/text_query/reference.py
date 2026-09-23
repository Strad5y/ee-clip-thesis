"""
reference.py

INSTRUCTIONS_maps_aether.md, Phase 3: rasterisierte Referenzkarten (Green Cover 2024,
Flughaefen) auf dem Arm-C-Grid, Regel >50% Flaechenanteil (exactextract, wie in Nachtrag 1
der Phase-0-Retrieval-Evaluation, MANIFEST.md). Ausgabe: ein uint8-GeoTIFF pro Kachel/
Referenz/Backbone (1 = positiv, 0 = negativ-aber-gueltig, nodata=255 ausserhalb der Maske),
unter outputs/text_query/reference/{backbone}/{reference}/.

Referenzen:
  - greencover: bereits reparierte Geometrien aus data/interim/greencover_2024_valid.parquet
    (509/519 Original-Polygone waren topologisch ungueltig, mit shapely.make_valid() behoben,
    siehe MANIFEST.md - HIER NICHT NEU REPARIERT, nur die fertige Datei gelesen, nichts an
    ihr geaendert).
  - airports: alle 7 aeroway=aerodrome-Flaechen aus greater_london-latest.osm.pbf (inkl.
    RAF Northolt und Kenley Airfield, Entscheidung Jakob 2026-09-22).

Grid/Masken: dieselben data/interim/text_query_masks/{backbone}/*_mask.tif wie in
similarity.py (Phase 2) - identisches Grid zu den Arm-C-Feldern, kein Resampling noetig.

Pflichtpruefung: die aufsummierten Positiv-Pixelzahlen ueber alle Kacheln MUESSEN exakt den
in Nachtrag 1 (Phase-0-Bericht, MANIFEST.md) dokumentierten Werten entsprechen - TESSERA
greencover 7.825.648 / airports 144.030, AE greencover 7.823.743 / airports 144.023. Abweichung
wird gemeldet, nicht stillschweigend hingenommen.
"""
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Repo-Root

import geopandas as gpd
import numpy as np
import rasterio
from exactextract import exact_extract
import pyrosm

import config as cfg

MASK_ROOT = cfg.DATA / "interim" / "text_query_masks"
GREENCOVER_PATH = cfg.DATA / "interim" / "greencover_2024_valid.parquet"
OSM_PBF = cfg.DATA / "raw" / "osm" / "greater_london-latest.osm.pbf"
OUT_ROOT = cfg.ROOT / "outputs" / "text_query" / "reference"

EXPECTED_COUNTS = {
    ("tessera", "greencover"): 7825648, ("tessera", "airports"): 144030,
    ("ae", "greencover"): 7823743, ("ae", "airports"): 144023,
}


def load_references() -> dict:
    greencover = gpd.read_parquet(GREENCOVER_PATH)
    assert (~greencover.geometry.is_valid).sum() == 0, "greencover_2024_valid.parquet enthaelt ungueltige Geometrien"
    if "ref_id" not in greencover.columns:
        greencover["ref_id"] = range(len(greencover))

    osm = pyrosm.OSM(str(OSM_PBF))
    airports = osm.get_data_by_custom_criteria(
        custom_filter={"aeroway": ["aerodrome"]}, filter_type="keep",
        keep_nodes=False, keep_ways=True, keep_relations=True,
    )
    airports = airports.set_crs("EPSG:4326")
    airports["ref_id"] = range(len(airports))
    assert len(airports) == 7, f"Erwartet 7 aeroway=aerodrome-Flaechen, gefunden {len(airports)}"

    print(f"Referenzen geladen: greencover={len(greencover)} Features, airports={len(airports)} Features")
    return {"greencover": greencover, "airports": airports}


def process_backbone(backbone: str, refs: dict) -> dict:
    mask_files = sorted((MASK_ROOT / backbone).glob("*_mask.tif"))
    print(f"\n=== {backbone}: {len(mask_files)} Masken-Kacheln ===")

    reproj_cache: dict = {}

    def get_reprojected(name: str, crs) -> gpd.GeoDataFrame:
        key = (name, str(crs))
        if key not in reproj_cache:
            t1 = time.time()
            reproj_cache[key] = refs[name].to_crs(crs)
            print(f"  [Cache] {name} -> {crs} reprojiziert ({time.time() - t1:.1f}s)")
        return reproj_cache[key]

    counts = {name: 0 for name in refs}
    n_valid_total = 0
    t0 = time.time()
    for i, mf in enumerate(mask_files):
        stem = mf.stem.removesuffix("_mask")
        with rasterio.open(mf) as src:
            mask = src.read(1).astype(bool)
            transform, crs, shape, width = src.transform, src.crs, src.shape, src.width
        n_valid_total += int(mask.sum())

        for name, ref_gdf in refs.items():
            ref_native = get_reprojected(name, crs)
            b = rasterio.transform.array_bounds(shape[0], shape[1], transform)
            cand = ref_native.cx[b[0]:b[2], b[1]:b[3]]

            cov = np.zeros(shape, dtype=np.float64)
            if len(cand) > 0:
                res = exact_extract(str(mf), cand[["ref_id", "geometry"]], ["cell_id", "coverage"],
                                     include_cols=["ref_id"], output="pandas")
                for _, row in res.iterrows():
                    cell_ids = np.asarray(row["cell_id"])
                    coverages = np.asarray(row["coverage"], dtype=np.float64)
                    if len(cell_ids) == 0:
                        continue
                    rr = cell_ids // width
                    cc = cell_ids % width
                    np.add.at(cov, (rr, cc), coverages)
            cov = np.minimum(cov, 1.0)
            positive = (cov > 0.5) & mask
            counts[name] += int(positive.sum())

            out = np.full(shape, 255, dtype=np.uint8)
            out[mask] = 0
            out[positive] = 1

            out_path = OUT_ROOT / backbone / name / f"{stem}.tif"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            profile = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1,
                       "dtype": "uint8", "crs": crs, "transform": transform, "nodata": 255,
                       "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256}
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out, 1)
        print(f"  Kachel {i + 1}/{len(mask_files)} ({stem}) fertig, gesamt {time.time() - t0:.0f}s")

    elapsed = time.time() - t0
    print(f"  Rechenzeit {backbone}: {elapsed:.1f}s, gueltige Pixel gesamt: {n_valid_total}")
    return {"counts": counts, "n_valid": n_valid_total, "elapsed_s": elapsed}


def main():
    refs = load_references()
    all_results = {}
    n_mismatch = 0
    for backbone in ("tessera", "ae"):
        result = process_backbone(backbone, refs)
        all_results[backbone] = result
        for name, n in result["counts"].items():
            expected = EXPECTED_COUNTS[(backbone, name)]
            ok = n == expected
            if not ok:
                n_mismatch += 1
            print(f"  [{backbone}/{name}] Positiv-Pixel: {n} (erwartet {expected}) "
                  f"{'OK' if ok else 'ABWEICHUNG!'}")

    if n_mismatch > 0:
        print(f"\n*** {n_mismatch} Abweichung(en) von Nachtrag 1 - siehe oben. ***")
    else:
        print("\nAlle Positiv-Pixelzahlen stimmen exakt mit Nachtrag 1 ueberein.")

    return all_results


if __name__ == "__main__":
    main()
