"""
similarity_untrained.py

Nutzervorgabe 2026-09-30: Similarity-Karten fuer eine Teilmenge der bestehenden Text-Queries,
aber auf den UNTRAINIERTEN Bild-Embeddings - die heruntergeladenen Roh-Embeddings (TESSERA
128d, AlphaEarth 64d) werden durch den untrainierten Projektionskopf in den 256d-Raum
abgebildet (= Arm B, bereits persistiert, run_landuse_probe.py::random_head()-Seed), ohne dass
dieser Kopf je mit Text-Embeddings trainiert wurde.

Query-Seite UNVERAENDERT gegenueber similarity.py: Qwen3-Embedding + TRAINIERTER Text-Kopf
des kanonischen Dedup-Checkpoints (encode.py) - die Query liegt also an derselben Stelle im
Raum wie bei den trainierten Karten, nur die Bildseite ist untrainiert.

Arm-B-Felder (gleicher BBox-Zuschnitt/Grid wie Arm C, daher dieselben Masken; process_tile()
prueft Shape und Transform per Assert):
  - AE:      data/interim/sdm_projected_ae_v1dedup/ae_armb/
  - TESSERA: data/interim/sdm_projected_tessera_dedupnorm/tessera_armb/ (Dedup-z-score-Statistik,
             konsistent zum v1dedup-Checkpoint, siehe build_sdm_arm_ab_tessera_dedupnorm.py)

Skalengrenzen: eigene 1./99. Perzentile ueber die Vereinigung beider untrainierter Backbones je
Query (gleiche Regel wie maps.py, aber getrennt berechnet - die Arm-B-Werte liegen in einem
anderen Bereich als die trainierten).

Ausgabe (bestehende Felder/Karten bleiben unveraendert):
  outputs/text_query/similarity_untrained/{backbone}/{query}/*.tif
  outputs/text_query/maps_untrained/similarity_untrained_{backbone}_{query}.png
  outputs/text_query/similarity_value_ranges_untrained.csv
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))  # fuer "from encode import ..." in similarity.py
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Repo-Root

import numpy as np
import pandas as pd
import rasterio
import torch

import config as cfg
import spatial_join as sj
import maps
import similarity as sim
from encode import encode_queries, load_query_encoder

ARMB_DIRS = {
    "ae": cfg.DATA / "interim" / "sdm_projected_ae_v1dedup" / "ae_armb",
    "tessera": cfg.DATA / "interim" / "sdm_projected_tessera_dedupnorm" / "tessera_armb",
}
QUERY_NAMES = ("green_space", "airport", "qual_quiet", "qual_busy", "qual_tourist", "cbd")

OUT_SIM = cfg.ROOT / "outputs" / "text_query" / "similarity_untrained"
OUT_MAPS = cfg.ROOT / "outputs" / "text_query" / "maps_untrained"
OUT_CSV = cfg.ROOT / "outputs" / "text_query" / "similarity_value_ranges_untrained.csv"


def list_armb_tile_pairs(backbone: str, region: str = "london") -> list:
    """Wie similarity.list_dedup_tile_pairs(), nur proj_path auf das Arm-B-Feld. Fehlende
    Felder/Masken werden gemeldet, nicht stillschweigend uebersprungen."""
    proj_dir = ARMB_DIRS[backbone]
    mask_dir = sim.MASK_ROOT / backbone
    raw_tiles = sj.get_backbones(region)[backbone]
    pairs, missing = [], []
    for raw_path in raw_tiles:
        proj_path = proj_dir / f"{raw_path.stem}_proj256.tif"
        mask_path = mask_dir / f"{raw_path.stem}_mask.tif"
        if not (proj_path.exists() and mask_path.exists()):
            missing.append(raw_path.name)
            continue
        with rasterio.open(raw_path) as src:
            crs = src.crs
        pairs.append(sim.DedupTilePair(backbone=backbone, tile_name=raw_path.name,
                                       raw_path=raw_path, proj_path=proj_path,
                                       mask_path=mask_path, crs=crs))
    if missing:
        print(f"  {backbone}: {len(missing)}/{len(raw_tiles)} Rohkachel(n) ohne Arm-B-Feld oder "
              f"Maske: {missing}")
    return pairs


def compute_fields(query_names: list) -> None:
    sim.OUT_ROOT = OUT_SIM  # process_tile() schreibt nach sim.OUT_ROOT
    queries = {q: sim.QUERIES[q] for q in query_names}
    print(f"Queries: {queries}")

    text_model = load_query_encoder()
    try:
        query_vecs = {b: encode_queries(list(queries.values()), b, text_model=text_model)
                      for b in ("tessera", "ae")}
    finally:
        del text_model
        torch.cuda.empty_cache()

    for backbone in ("tessera", "ae"):
        print(f"\n=== {backbone} (Arm B, untrainiert): Similarity-Felder ===")
        pairs = list_armb_tile_pairs(backbone)
        print(f"  {len(pairs)} Kachel-Paare (Roh/Arm-B/Maske)")
        stats = {q: {"n_valid": 0, "n_total": 0, "min": np.inf, "max": -np.inf, "sum": 0.0}
                 for q in query_names}
        t0 = time.time()
        for i, p in enumerate(pairs):
            t_tile = time.time()
            sim.process_tile(p, query_names, query_vecs[backbone], stats)
            print(f"  Kachel {i + 1}/{len(pairs)} ({p.tile_name}) fertig "
                  f"({time.time() - t_tile:.1f}s, gesamt {time.time() - t0:.0f}s)")


def render_maps(query_names: list) -> None:
    maps.SIM_ROOT = OUT_SIM
    transform, shape, aoi_gdf, aoi_mask = maps.build_common_grid()

    rows, limits = [], {}
    for q in query_names:
        per_backbone, p1, p99, n_union = maps.compute_query_stats_and_percentiles(q)
        limits[q] = (p1, p99)
        print(f"\n[{q}] Skalengrenzen (untrainiert, 1./99. Perzentil, n={n_union}): "
              f"vmin={p1:.4f} vmax={p99:.4f}")
        for backbone, s in per_backbone.items():
            print(f"  [{backbone}/{q}] min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} "
                  f"n_valid={s['n_valid']}/{s['n_total']} NoData={100 * s['frac_nodata']:.2f}%")
            rows.append({
                "backbone": backbone, "arm": "b_untrained", "query": q,
                "value_min": s["min"], "value_max": s["max"], "value_mean": s["mean"],
                "n_valid": s["n_valid"], "n_total": s["n_total"], "frac_nodata": s["frac_nodata"],
                "map_vmin_p1_shared": p1, "map_vmax_p99_shared": p99,
                "n_pixels_for_percentile_union": n_union,
            })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nGespeichert: {OUT_CSV}")

    for backbone in maps.BACKBONES:
        for q in query_names:
            vmin, vmax = limits[q]
            out_path = OUT_MAPS / f"similarity_untrained_{backbone}_{q}.png"
            maps.render_similarity_map(backbone, q, vmin, vmax, transform, shape,
                                       aoi_gdf, aoi_mask, out_path,
                                       title=f"{backbone} (untrained head) — {q}")
            print(f"  {out_path}")


def main():
    query_names = list(QUERY_NAMES)
    compute_fields(query_names)
    render_maps(query_names)


if __name__ == "__main__":
    main()
