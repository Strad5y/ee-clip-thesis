"""
similarity.py

INSTRUCTIONS_maps_aether.md, Phase 2: Kosinus-Similarity-Felder zwischen den drei Queries
(green/airport/control) und den kanonischen Dedup-Arm-C-256d-Feldern, kachelweise, als
float32-GeoTIFF pro Kachel.

Kachel-Zuordnung NICHT ueber src/eval/text_query/tiles.py::list_tile_pairs() (das zeigt fest
auf die ALTEN, nicht-kanonischen Arm-C-Felder unter data/interim/sdm_projected/{backbone}_armc/
- siehe Nachtrag 2/Phase-0-Bericht: diese wurden mit den alten, nicht-dedup Checkpoints
erzeugt). Diese Datei baut die Kachel-Paarung stattdessen direkt auf die in INSTRUCTIONS_
maps_aether.md genannten kanonischen Verzeichnisse (data/interim/sdm_projected_{backbone}_
v1dedup/{backbone}_armc/), mit einer zu tiles.py::TilePair analogen Struktur - tiles.py selbst
bleibt unveraendert (wird von anderen, bereits abgeschlossenen Aufgaben referenziert).

Zeilenweise gechunkt (ROW_CHUNK), NICHT die volle Kachel auf einmal gelesen: eine AlphaEarth-
Kachel hat 8192x8192x256 float32-Kanaele - ein voller src.read() waere ~64 GB RSS pro Kachel
(dasselbe Speicherproblem, das build_sdm_projected_field.py bereits einmal hatte und mit
ROW_CHUNK=200 behoben hat, siehe MANIFEST.md - hier aus Vorsicht identisch behandelt, TESSERA-
Kacheln sind mit ~1140x740 zwar unkritisch, aber einheitlicher Code ist weniger fehleranfaellig
als eine Sonderfallunterscheidung).

Beide Felder (Arm-C-Projektion UND Maske) liegen im selben Grid (siehe masks.py-Docstring:
"gleiche Shape/Transform/CRS wie das zugehoerige Arm-C-GeoTIFF") - kein Resampling noetig.
Arm-C-Vektoren sind bereits L2-normiert (verifiziert in Nachtrag 2/Phase 0), Query-Vektoren
ebenfalls (encode.py, ProjectionHead-eigene Normierung) - Kosinus reduziert sich auf ein
einfaches Skalarprodukt je Pixel, hier trotzdem ueber Normen robust berechnet (schuetzt gegen
kleine Abweichungen von exakt 1.0, siehe Cache-Verifikations-Befund in encode.py/MANIFEST.md).
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Repo-Root

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window

import config as cfg
import spatial_join as sj
from encode import encode_queries, load_query_encoder

PROJ_ROOT_TEMPLATE = "sdm_projected_{backbone}_v1dedup"  # kanonisch, siehe INSTRUCTIONS_maps_aether.md
MASK_ROOT = cfg.DATA / "interim" / "text_query_masks"
OUT_ROOT = cfg.ROOT / "outputs" / "text_query" / "similarity"
ROW_CHUNK = 200  # wie build_sdm_projected_field.py - begrenzt Speicherbedarf pro Kachel

QUERIES = {
    "green": "Green Park and Natural Reserve.",
    "airport": "Airport and Transportation Hub.",
    "control": "desert",
}


@dataclass
class DedupTilePair:
    backbone: str
    tile_name: str
    raw_path: Path
    proj_path: Path   # Arm-C, kanonischer Dedup-Checkpoint
    mask_path: Path
    crs: CRS


def list_dedup_tile_pairs(backbone: str, region: str = "london") -> list[DedupTilePair]:
    """Wie tiles.py::list_tile_pairs(), aber proj_path zeigt auf die kanonischen Dedup-Arm-C-
    Felder (data/interim/sdm_projected_{backbone}_v1dedup/{backbone}_armc/) statt auf die alten
    Felder unter data/interim/sdm_projected/. Kacheln ohne Arm-C-Feld oder ohne Maske werden
    gemeldet, nicht stillschweigend uebersprungen."""
    proj_dir = cfg.DATA / "interim" / PROJ_ROOT_TEMPLATE.format(backbone=backbone) / f"{backbone}_armc"
    mask_dir = MASK_ROOT / backbone
    raw_tiles = sj.get_backbones(region)[backbone]

    pairs = []
    missing_proj, missing_mask = [], []
    for raw_path in raw_tiles:
        proj_path = proj_dir / f"{raw_path.stem}_proj256.tif"
        mask_path = mask_dir / f"{raw_path.stem}_mask.tif"
        if not proj_path.exists():
            missing_proj.append(raw_path.name)
            continue
        if not mask_path.exists():
            missing_mask.append(raw_path.name)
            continue
        with rasterio.open(raw_path) as src:
            crs = src.crs
        pairs.append(DedupTilePair(backbone=backbone, tile_name=raw_path.name, raw_path=raw_path,
                                    proj_path=proj_path, mask_path=mask_path, crs=crs))
    if missing_proj:
        print(f"  {backbone}: {len(missing_proj)}/{len(raw_tiles)} Rohkachel(n) ohne kanonisches "
              f"Dedup-Arm-C-Feld ({proj_dir}): {missing_proj}")
    if missing_mask:
        print(f"  {backbone}: {len(missing_mask)}/{len(raw_tiles)} Rohkachel(n) ohne Maske "
              f"({mask_dir}): {missing_mask}")
    return pairs


def _out_profile(height: int, width: int, crs, transform: dict) -> dict:
    return {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan,
        "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
    }


def process_tile(p: DedupTilePair, query_names: list, query_vecs: np.ndarray,
                  stats: dict, row_chunk: int = ROW_CHUNK) -> None:
    """Liest Arm-C-Feld + Maske zeilenweise (row_chunk Zeilen je Fenster), berechnet den
    Kosinus je Query direkt im Fenster und schreibt ihn in das jeweilige Ausgabe-GeoTIFF -
    nie mehr als ein Zeilen-Fenster (alle 256 Kanaele) im Speicher, wie build_sdm_projected_
    field.py::project_tile(). stats wird IN-PLACE aktualisiert (min/max/sum/n_valid/n_total
    je Query, ueber alle Kacheln hinweg)."""
    with rasterio.open(p.proj_path) as src:
        height, width, crs, transform = src.height, src.width, src.crs, src.transform
        n_bands = src.count
        assert n_bands == 256, f"{p.proj_path.name}: erwartet 256 Kanaele, hat {n_bands}"
        with rasterio.open(p.mask_path) as msrc:
            assert msrc.shape == (height, width), (
                f"{p.proj_path.name}: Arm-C-Shape ({height},{width}) != Masken-Shape {msrc.shape}"
            )
            assert msrc.transform == transform, f"{p.proj_path.name}: Transform weicht von der Maske ab"

        q_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
        q_unit = (query_vecs / np.where(q_norms > 0, q_norms, 1.0)).astype(np.float32)

        dsts = {}
        for q in query_names:
            out_path = OUT_ROOT / p.backbone / q / f"{p.raw_path.stem}.tif"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dsts[q] = rasterio.open(out_path, "w", **_out_profile(height, width, crs, transform))
        try:
            for r0 in range(0, height, row_chunk):
                h = min(row_chunk, height - r0)
                win = Window(0, r0, width, h)
                arr = src.read(window=win)  # (256, h, width)
                with rasterio.open(p.mask_path) as msrc:
                    mask_chunk = msrc.read(1, window=win).astype(bool)

                valid = np.isfinite(arr).all(axis=0) & mask_chunk
                norms = np.linalg.norm(arr, axis=0)
                norms_safe = np.where(norms > 0, norms, 1.0)

                sims = np.einsum("qc,chw->qhw", q_unit, arr.astype(np.float32))
                sims = sims / norms_safe[None, :, :]
                sims = np.where(valid[None, :, :], sims, np.nan).astype(np.float32)

                for qi, q in enumerate(query_names):
                    band = sims[qi]
                    dsts[q].write(band, 1, window=win)
                    s = stats[q]
                    s["n_total"] += band.size
                    v = valid
                    n_valid = int(v.sum())
                    s["n_valid"] += n_valid
                    if n_valid > 0:
                        vals = band[v]
                        s["min"] = min(s["min"], float(vals.min()))
                        s["max"] = max(s["max"], float(vals.max()))
                        s["sum"] += float(vals.sum())
        finally:
            for dst in dsts.values():
                dst.close()


def run_backbone(backbone: str, query_names: list) -> dict:
    print(f"\n=== {backbone}: Similarity-Felder ===")
    pairs = list_dedup_tile_pairs(backbone)
    print(f"  {len(pairs)} Kachel-Paare (Roh/Arm-C-dedup/Maske)")

    text_model = load_query_encoder()
    try:
        query_texts = [QUERIES[q] for q in query_names]
        query_vecs = encode_queries(query_texts, backbone, text_model=text_model)  # (n_query, 256)
    finally:
        del text_model
        import torch
        torch.cuda.empty_cache()

    stats = {q: {"n_valid": 0, "n_total": 0, "min": np.inf, "max": -np.inf, "sum": 0.0} for q in query_names}
    t0 = time.time()
    for i, p in enumerate(pairs):
        t_tile = time.time()
        process_tile(p, query_names, query_vecs, stats)
        print(f"  Kachel {i + 1}/{len(pairs)} ({p.tile_name}) fertig ({time.time() - t_tile:.1f}s, "
              f"gesamt {time.time() - t0:.0f}s)")
    elapsed = time.time() - t0

    result = {"backbone": backbone, "elapsed_s": elapsed, "queries": {}}
    for q in query_names:
        s = stats[q]
        n_total, n_valid = s["n_total"], s["n_valid"]
        mean = s["sum"] / n_valid if n_valid > 0 else float("nan")
        result["queries"][q] = {
            "min": s["min"] if n_valid > 0 else float("nan"),
            "max": s["max"] if n_valid > 0 else float("nan"),
            "mean": mean,
            "n_valid": n_valid, "n_total": n_total,
            "frac_nodata": 1.0 - n_valid / n_total if n_total > 0 else float("nan"),
        }
        print(f"  [{backbone}/{q}] min={s['min']:.4f} max={s['max']:.4f} mean={mean:.4f} "
              f"gueltig={n_valid}/{n_total} ({100 * n_valid / n_total:.2f}%) "
              f"NoData={100 * (1 - n_valid / n_total):.2f}%")
    print(f"  Rechenzeit {backbone} (alle {len(query_names)} Queries, {len(pairs)} Kacheln): "
          f"{elapsed:.1f}s")
    return result


def main():
    query_names = list(QUERIES.keys())
    print(f"Queries: {QUERIES}")
    all_results = {}
    for backbone in ("tessera", "ae"):
        all_results[backbone] = run_backbone(backbone, query_names)
    return all_results


if __name__ == "__main__":
    main()
