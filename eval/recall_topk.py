"""
eval/recall_topk.py

Recall@Top-k% after AETHER (Liu et al., arXiv 2510.09894v2, Sec. 3.5 Eq. 11, Sec. 4.6, Tab. 3)
for the five AETHER categories in queries/aether_categories.yaml, both backbones. No training.

Candidate space: every valid pixel inside the Greater-London AOI polygon (whole London, no cell
filter) = text-query mask (uniqueness rule + AOI polygon, data/interim/text_query_masks/) AND a
finite canonical dedup Arm-C vector (data/interim/sdm_projected_{bb}_v1dedup/{bb}_armc/).
N = number of these pixels. Top set = the K = ceil(k*N) pixels with the highest cosine
similarity (np.argpartition, deterministic). The threshold is always taken over all of London,
recall is then reported for the GT pixels of train / test / all separately.

Ground truth (pixel level, no buffer): exactly the rule of the training pairs - a pixel is
positive if a POI of an assigned class intersects it, taken directly from the dedup training
pixel table pixels_{bb}_dedup.parquet (poi_ids column = all POIs intersecting the pixel, built
by spatial_join.py/exactextract -> aggregate_pixel_pairs.py). GT of a category = union of its
osm_classes (a POI counts if ANY of its key=value tags is in osm_classes).

Arms:
  T  trained head: query encoded exactly as in training (src/eval/text_query/encode.py:
     Qwen3-Embedding-4B, MRL 2560->1024, L2, then text_head of the dedup checkpoint, L2);
     cosine against the stored 256-d Arm-C field.
  P  prototype: mean of the RAW embeddings of all GT pixels in train cells; cosine in raw
     space. AE: dequantized, L2-normalised per pixel. TESSERA: z-score with the dedup stats
     (data/processed/norm_stats_tessera_pixels_dedup.json).
  R  random: K/N, analytic.

CIs: cell-level bootstrap (1-km cells of data/splits/splits_london.csv, cell assignment identical
to aggregate_pixel_pairs.compute_pixel_cells), 1000 resamples, seed 0, over the cells that contain
GT pixels of the category in the split. Same resample weights for T and P -> paired T-P diff.

Usage:
  python eval/recall_topk.py feasibility          # Phase 1 report (no scores)
  python eval/recall_topk.py selftest             # synthetic self-test
  python eval/recall_topk.py run --backbone ae    # Phase 3 per backbone (scores, recall, fields)
  python eval/recall_topk.py run --backbone tessera
  python eval/recall_topk.py finalize             # merged CSVs, table, maps
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "eval" / "text_query")]

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window

import config as cfg

OUT = ROOT / "outputs" / "recall_topk"
FIELDS = OUT / "fields"
CATEGORIES_YAML = ROOT / "queries" / "aether_categories.yaml"
POIS_PATH = cfg.DATA / "interim" / "pois_london_full_filtered.parquet"
PIXELS_TEMPLATE = cfg.DATA / "processed" / "pixels_{bb}_dedup.parquet"
TESSERA_STATS = cfg.DATA / "processed" / "norm_stats_tessera_pixels_dedup.json"
CKPTS = {"ae": ROOT / "checkpoints" / "ae_v1_dedup" / "best.pt",
         "tessera": ROOT / "checkpoints" / "tessera_v1_dedup" / "best.pt"}

KS = (0.01, 0.05, 0.10)
SPLITS = ("train", "test", "all")
N_BOOT = 1000
BOOT_SEED = 0
THIN_MIN_TEST_PIXELS = 50
THIN_MIN_TEST_CELLS = 3
ROW_CHUNK = 200            # like similarity.py / build_sdm_projected_field.py
MIN_FREE_GB = 50
BACKBONES = ("ae", "tessera")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def run_provenance_gate() -> None:
    """verify_provenance.py before every computation on evaluation data (hard gate)."""
    r = subprocess.run([sys.executable, str(ROOT / "src" / "verify_provenance.py")],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip())
    if r.returncode != 0:
        raise SystemExit(f"verify_provenance.py FAILED (rc={r.returncode}): {r.stderr.strip()[-500:]}")


# --------------------------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------------------------

def load_categories() -> list[dict]:
    from class_table import load_class_table
    cats = yaml.safe_load(open(CATEGORIES_YAML))["categories"]
    ct = load_class_table(cfg.CLASS_TABLE)
    known = set(ct["key"] + "=" + ct["value"])
    for c in cats:
        c.setdefault("extra_osm_tags", [])
        for tag in c["extra_osm_tags"]:
            assert tag.count("=") == 1 and tag not in known, f"{c['id']}: bad extra tag {tag}"
        missing = [k for k in c["osm_classes"] if k not in known]
        if missing:
            raise SystemExit(f"{c['id']}: classes not in class table: {missing}")
        assert c["query"] == f"A place with the type of {c['aether_name']}.", c["id"]
    return cats


# --------------------------------------------------------------------------------------------
# Tiles, candidate space, GT
# --------------------------------------------------------------------------------------------

@dataclass
class Tile:
    name: str            # raw tile file name (key in pixels_*_dedup.parquet)
    raw_path: Path
    proj_path: Path
    mask_path: Path
    height: int
    width: int
    row_off: int         # Arm-C grid (0,0) == raw tile (row_off, col_off)
    col_off: int
    transform: object
    crs: object


def list_tiles(bb: str) -> list[Tile]:
    from similarity import list_dedup_tile_pairs
    tiles = []
    for p in list_dedup_tile_pairs(bb):
        with rasterio.open(p.proj_path) as ps, rasterio.open(p.raw_path) as rs, \
                rasterio.open(p.mask_path) as ms:
            pt, rt = ps.transform, rs.transform
            assert ps.crs == rs.crs and (pt.a, pt.e) == (rt.a, rt.e), p.tile_name
            assert ms.shape == ps.shape and ms.transform == pt, p.tile_name
            col_off = (pt.c - rt.c) / rt.a
            row_off = (pt.f - rt.f) / rt.e
            assert abs(col_off - round(col_off)) < 1e-6 and abs(row_off - round(row_off)) < 1e-6
            tiles.append(Tile(p.tile_name, p.raw_path, p.proj_path, p.mask_path, ps.height,
                              ps.width, int(round(row_off)), int(round(col_off)), pt, ps.crs))
    return tiles


def candidate_masks(tiles: list[Tile]) -> list[np.ndarray]:
    """Per tile bool (h, w): text-query mask AND finite Arm-C band 1 (Arm-C is all-NaN or
    all-finite per pixel by construction, checked again chunk-wise when scoring)."""
    out = []
    for t in tiles:
        with rasterio.open(t.mask_path) as ms:
            m = ms.read(1).astype(bool)
        with rasterio.open(t.proj_path) as ps:
            m &= np.isfinite(ps.read(1))
        out.append(m)
    return out


def extract_extra_tag(tag: str):
    """POIs with an OSM tag OUTSIDE the class table (e.g. club=sport), straight from the same
    London PBF the pipeline was built from (config.PBF_LONDON). poi_id = '{osm_type}/{id}' as in
    pois_london_full_filtered.parquet. Invalid polygons are repaired with make_valid."""
    import geopandas as gpd
    import shapely
    from pyrosm import OSM
    k, v = tag.split("=")
    g = OSM(str(cfg.PBF_LONDON)).get_data_by_custom_criteria(
        custom_filter={k: [v]}, filter_type="keep", keep_nodes=True, keep_ways=True,
        keep_relations=True)
    g = g[g[k] == v].copy()
    g["poi_id"] = g["osm_type"].astype(str) + "/" + g["id"].astype(str)
    g = gpd.GeoDataFrame(g[["poi_id", "geometry"]], geometry="geometry", crs="EPSG:4326")
    bad = ~g.geometry.is_valid
    g.loc[bad, "geometry"] = shapely.make_valid(g.loc[bad, "geometry"].values)
    g = g.drop_duplicates(subset="poi_id").reset_index(drop=True)
    info = {"n_objects": len(g), "n_made_valid": int(bad.sum()),
            "geom_types": g.geometry.geom_type.value_counts().to_dict()}
    return g, info


_EXTRA_CACHE: dict = {}


def extra_tag_gdf(tag: str):
    if tag not in _EXTRA_CACHE:
        _EXTRA_CACHE[tag] = extract_extra_tag(tag)
    return _EXTRA_CACHE[tag]


def category_poi_sets(cats: list[dict]) -> tuple[dict, dict]:
    """poi_id sets per category (POI has >=1 tag in osm_classes or extra_osm_tags) + #POIs in
    London per class/tag."""
    pois = pd.read_parquet(POIS_PATH, columns=["poi_id", "key", "value"])
    pois["kv"] = pois["key"].astype(str).str.strip() + "=" + pois["value"].astype(str).str.strip()
    sets, per_class = {}, {}
    for c in cats:
        sub = pois[pois["kv"].isin(c["osm_classes"])]
        sets[c["id"]] = set(sub["poi_id"])
        for kv in c["osm_classes"]:
            per_class[kv] = int(pois.loc[pois["kv"] == kv, "poi_id"].nunique())
        for tag in c["extra_osm_tags"]:
            g, info = extra_tag_gdf(tag)
            sets[c["id"]] |= set(g["poi_id"])
            per_class[tag] = info["n_objects"]
    return sets, per_class


def build_gt(bb: str, cats: list[dict], tiles: list[Tile], cand: list[np.ndarray],
             poi_sets: dict) -> dict:
    """GT pixels per category, located in the candidate space.

    Returns {cat_id: DataFrame[tile_idx, r, c (Arm-C grid), split, cell, n_pois_hit(set)]} plus
    bookkeeping of GT pixels that fall outside the candidate space (dropped, reported)."""
    from aggregate_pixel_pairs import build_cell_to_split, compute_pixel_cells
    px = pd.read_parquet(str(PIXELS_TEMPLATE).format(bb=bb),
                         columns=["tile", "row", "col", "poi_ids", "split"])
    all_ids = set().union(*poi_sets.values()) if poi_sets else set()
    ids = px["poi_ids"].str.split(",")
    hit = ids.map(lambda l: not all_ids.isdisjoint(l))
    px = px[hit.to_numpy()].copy()
    px["poi_list"] = ids[hit.to_numpy()]
    px["source"] = "pixel_table"

    # tags outside the class table: same rasterisation rule as the pairs (spatial_join:
    # zone assignment, points -> containing pixel, polygons -> exactextract coverage > 0,
    # lines skipped), joined against exactly the raw tiles used here
    extra_info = {}
    extra_tags = sorted({t for c in cats for t in c["extra_osm_tags"]})
    if extra_tags:
        import spatial_join as sj
        for tag in extra_tags:
            g, info = extra_tag_gdf(tag)
            pairs = sj.join_pois_to_tiles(g, [t.raw_path for t in tiles])
            info = {**info, "n_pairs": len(pairs),
                    "n_pois_joined": int(pairs["poi_id"].nunique()) if len(pairs) else 0}
            extra_info[tag] = info
            if len(pairs):
                e = pairs[["tile", "row", "col", "poi_id"]].copy()
                e["poi_list"] = e["poi_id"].map(lambda x: [x])
                e["split"] = None
                e["source"] = "extra"
                px = pd.concat([px, e.drop(columns="poi_id")], ignore_index=True)

    tile_idx = {t.name: i for i, t in enumerate(tiles)}
    px["tile_idx"] = px["tile"].map(tile_idx)
    n_unknown_tile = int(px["tile_idx"].isna().sum())
    px = px[px["tile_idx"].notna()].copy()
    px["tile_idx"] = px["tile_idx"].astype(int)
    off_r = np.array([t.row_off for t in tiles])
    off_c = np.array([t.col_off for t in tiles])
    hs = np.array([t.height for t in tiles]); ws = np.array([t.width for t in tiles])
    px["r"] = px["row"].to_numpy() - off_r[px["tile_idx"].to_numpy()]
    px["c"] = px["col"].to_numpy() - off_c[px["tile_idx"].to_numpy()]
    inside = ((px["r"] >= 0) & (px["c"] >= 0) & (px["r"] < hs[px["tile_idx"]]) &
              (px["c"] < ws[px["tile_idx"]])).to_numpy()
    in_cand = np.zeros(len(px), dtype=bool)
    ii = np.where(inside)[0]
    for ti in np.unique(px["tile_idx"].to_numpy()[ii]):
        sel = ii[px["tile_idx"].to_numpy()[ii] == ti]
        in_cand[sel] = cand[ti][px["r"].to_numpy()[sel], px["c"].to_numpy()[sel]]

    # cells, identical to aggregate_pixel_pairs / build_splits_pixels
    cell_to_split, utm_crs, (minx, miny) = build_cell_to_split()
    cells = compute_pixel_cells(px[["tile", "row", "col"]], [t.raw_path for t in tiles],
                                utm_crs, minx, miny)
    px = px.merge(cells, on=["tile", "row", "col"], how="left")
    rec_split = pd.Series([cell_to_split.get((int(a), int(b))) for a, b in
                           zip(px["cell_row"], px["cell_col"])], index=px.index)
    is_tab = (px["source"] == "pixel_table").to_numpy()
    n_split_mismatch = int((rec_split[is_tab] != px.loc[is_tab, "split"]).sum())
    if n_split_mismatch:
        raise SystemExit(f"{bb}: {n_split_mismatch} GT pixels whose reconstructed cell split != "
                         f"split column of {PIXELS_TEMPLATE} - provenance mismatch, abort.")
    px.loc[~is_tab, "split"] = rec_split[~is_tab]
    n_extra_no_split = int(((~is_tab) & px["split"].isna().to_numpy() & in_cand).sum())
    px["split"] = px["split"].fillna("none")   # cell not in splits_london.csv -> only in 'all'
    px["cell"] = px["cell_row"].astype(np.int64) * 100000 + px["cell_col"].astype(np.int64)
    px["in_cand"] = in_cand

    out = {"_meta": {"n_unknown_tile": n_unknown_tile, "extra_tags": extra_info,
                     "n_extra_pixels_without_split_cell": n_extra_no_split}}
    for c in cats:
        s = poi_sets[c["id"]]
        m = px["poi_list"].map(lambda l: not s.isdisjoint(l)).to_numpy() if s else np.zeros(len(px), bool)
        sub = px[m]
        pois_hit = sub.loc[sub["in_cand"], "poi_list"].map(lambda l: s.intersection(l))
        df = sub.loc[sub["in_cand"], ["tile_idx", "r", "c", "split", "cell"]]
        df = df.drop_duplicates(subset=["tile_idx", "r", "c"])  # pixel table + extra join overlap
        out[c["id"]] = {
            "df": df.reset_index(drop=True),
            "n_dropped_outside_candidates": int((~sub["in_cand"]).sum()),
            "n_pois_with_gt": len(set().union(*pois_hit)) if len(pois_hit) else 0,
        }
    return out


def flat_index(tiles: list[Tile], cand: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Global candidate index: offsets per tile + per-tile (h,w) -> global idx lookup (int64, -1)."""
    offsets = np.zeros(len(tiles) + 1, dtype=np.int64)
    lut = []
    for i, m in enumerate(cand):
        n = int(m.sum())
        offsets[i + 1] = offsets[i] + n
        l = np.full(m.shape, -1, dtype=np.int64)
        l[m] = offsets[i] + np.arange(n)
        lut.append(l)
    return offsets, lut


def gt_global_idx(gt_df: pd.DataFrame, lut: list[np.ndarray]) -> np.ndarray:
    g = np.empty(len(gt_df), dtype=np.int64)
    for ti, sub in gt_df.groupby("tile_idx"):
        g[sub.index.to_numpy()] = lut[ti][sub["r"].to_numpy(), sub["c"].to_numpy()]
    assert (g >= 0).all()
    return g


def status_of(gtinfo: dict) -> str:
    df = gtinfo["df"]
    if len(df) == 0:
        return "no_gt"
    te = df[df["split"] == "test"]
    if len(te) < THIN_MIN_TEST_PIXELS or te["cell"].nunique() < THIN_MIN_TEST_CELLS:
        return "thin"
    return "ok"


# --------------------------------------------------------------------------------------------
# Feasibility (Phase 1)
# --------------------------------------------------------------------------------------------

def feasibility() -> None:
    run_provenance_gate()
    cats = load_categories()
    poi_sets, per_class = category_poi_sets(cats)
    rows = []
    for bb in BACKBONES:
        tiles = list_tiles(bb)
        cand = candidate_masks(tiles)
        n_cand = int(sum(m.sum() for m in cand))
        log(f"{bb}: {len(tiles)} tiles, N={n_cand} candidate pixels")
        gt = build_gt(bb, cats, tiles, cand, poi_sets)
        for c in cats:
            g = gt[c["id"]]
            df = g["df"]
            r = {"backbone": bb, "category_id": c["id"], "N": n_cand,
                 "n_pois_london": len(poi_sets[c["id"]]), "n_pois_with_gt": g["n_pois_with_gt"],
                 "gt_dropped_outside_candidates": g["n_dropped_outside_candidates"],
                 "status": status_of(g)}
            for sp in ("train", "val", "test"):
                d = df[df["split"] == sp]
                r[f"gt_pixels_{sp}"] = len(d)
                r[f"gt_cells_{sp}"] = d["cell"].nunique()
            r["gt_pixels_all"] = len(df)
            r["gt_cells_all"] = df["cell"].nunique()
            rows.append(r)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "feasibility.csv", index=False)
    write_feasibility_md(cats, per_class, res)


def write_feasibility_md(cats, per_class, res) -> None:
    L = ["# Machbarkeit Recall@Top-k% — AETHER-Kategorien", "",
         f"Erzeugt von `eval/recall_topk.py feasibility` am {time.strftime('%Y-%m-%d %H:%M')}. "
         "Zuordnung: `queries/aether_categories.yaml` (vor dieser Auszählung festgelegt).", "",
         "Status: `ok` / `thin` (< 50 Test-GT-Pixel oder < 3 Testzellen) / `no_gt` (keine GT-Pixel). "
         "GT-Pixel = Pixel der Dedup-Trainingspixeltabelle, die ein POI einer zugeordneten Klasse "
         "schneidet (kein Puffer), eingeschränkt auf den Kandidatenraum (AOI-Polygon + Eindeutigkeitsmaske "
         "+ gültiges Arm-C-Feld). Zellen = 1-km-Zellen aus `data/splits/splits_london.csv`.", ""]
    L += ["## Klassen", "", "| Kategorie | Klasse | POIs in London (Tabelle) |", "|---|---|---|"]
    for c in cats:
        if not c["osm_classes"] and not c["extra_osm_tags"]:
            L.append(f"| {c['aether_name']} | — (keine Klasse) | — |")
        for kv in c["osm_classes"]:
            L.append(f"| {c['aether_name']} | `{kv}` | {per_class[kv]} |")
        for tag in c["extra_osm_tags"]:
            L.append(f"| {c['aether_name']} | `{tag}` (außerhalb der Klassentabelle, direkt aus dem PBF) "
                     f"| {per_class[tag]} |")
    L += ["", "Unsichere Klassen (nicht in der GT, nur zur Prüfung): siehe YAML `uncertain_classes`.", ""]
    for tag, info in _EXTRA_CACHE.items():
        L.append(f"Extra-Tag `{tag}`: {info[1]['n_objects']} Objekte, Geometrietypen {info[1]['geom_types']}, "
                 f"{info[1]['n_made_valid']} per make_valid repariert. Gerastert mit spatial_join.join_pois_to_tiles "
                 "(Punkte → Pixel, Polygone → exactextract Coverage > 0, Linien übersprungen).")
    if _EXTRA_CACHE:
        L.append("")
    for bb in BACKBONES:
        sub = res[res["backbone"] == bb]
        L += [f"## {bb} (N = {int(sub['N'].iloc[0]):,} Kandidatenpixel)".replace(",", "."), "",
              "| Kategorie | POIs (Tabelle / mit GT-Pixel) | GT-Pixel train | Zellen train | GT-Pixel test | "
              "Zellen test | GT-Pixel val | GT-Pixel gesamt | außerhalb Kandidatenraum verworfen | Status |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            L.append(f"| {r['category_id']} | {r['n_pois_london']} / {r['n_pois_with_gt']} | "
                     f"{r['gt_pixels_train']} | {r['gt_cells_train']} | {r['gt_pixels_test']} | "
                     f"{r['gt_cells_test']} | {r['gt_pixels_val']} | {r['gt_pixels_all']} | "
                     f"{r['gt_dropped_outside_candidates']} | **{r['status']}** |")
        L.append("")
    (OUT / "feasibility.md").write_text("\n".join(L), encoding="utf-8")
    log(f"written {OUT / 'feasibility.md'}")


# --------------------------------------------------------------------------------------------
# Recall + bootstrap core (pure functions, self-tested)
# --------------------------------------------------------------------------------------------

def top_mask(scores: np.ndarray, k: float) -> tuple[np.ndarray, int]:
    n = len(scores)
    K = int(math.ceil(k * n))
    idx = np.argpartition(-scores, K - 1)[:K]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m, K


def boot_weights(n_cells: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, n_cells) multiplicity of each cell in each resample."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_cells, size=(n_boot, n_cells))
    w = np.zeros((n_boot, n_cells), dtype=np.int32)
    for b in range(n_boot):
        w[b] = np.bincount(draws[b], minlength=n_cells)
    return w


def recall_with_boot(hit: np.ndarray, cell: np.ndarray, w_cells: np.ndarray, cell_codes: np.ndarray):
    """hit: bool per GT pixel; cell: cell code per GT pixel; cell_codes: sorted unique cells
    matching the columns of w_cells. Returns point estimate + bootstrap samples."""
    ci = np.searchsorted(cell_codes, cell)
    n_per = np.bincount(ci, minlength=len(cell_codes)).astype(np.float64)
    h_per = np.bincount(ci, weights=hit.astype(np.float64), minlength=len(cell_codes))
    point = h_per.sum() / n_per.sum()
    samples = (w_cells @ h_per) / (w_cells @ n_per)
    return float(point), samples


def ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(lo), float(hi)


def selftest(n: int = 2_000_000, n_gt: int = 100_000, seed: int = 123) -> tuple[list[str], bool]:
    """Random scores -> recall ~ K/N (~k); oracle (1 on GT, 0 else) -> min(1, K/|GT|) exactly.
    Also checks the bootstrap path on the same data and one oracle case with |GT| > K."""
    rng = np.random.default_rng(seed)
    gt = rng.choice(n, size=n_gt, replace=False)
    cell = rng.integers(0, 500, size=n_gt)
    codes = np.unique(cell)
    w = boot_weights(len(codes), n_boot=200)
    lines, ok = [], True
    for k in KS:
        m, K = top_mask(rng.random(n).astype(np.float32), k)
        rec, samp = recall_with_boot(m[gt], cell, w, codes)
        sd = math.sqrt(k * (1 - k) / n_gt)
        good = abs(rec - K / n) < 5 * sd
        ok &= good
        lines.append(f"random k={k}: recall={rec:.5f} expected K/N={K / n:.5f} (5sd={5 * sd:.5f}) "
                     f"boot-mean={np.nanmean(samp):.5f} -> {'PASS' if good else 'FAIL'}")
        oracle = np.zeros(n, dtype=np.float32); oracle[gt] = 1.0
        m, K = top_mask(oracle, k)
        rec, _ = recall_with_boot(m[gt], cell, w, codes)
        exp = min(1.0, K / n_gt)
        good = abs(rec - exp) < 1e-12
        ok &= good
        lines.append(f"oracle k={k}: recall={rec:.6f} expected min(1,K/|GT|)={exp:.6f} -> {'PASS' if good else 'FAIL'}")
    # oracle with |GT| > K (K/|GT| < 1 branch)
    big = rng.choice(n, size=int(0.2 * n), replace=False)
    oracle = np.zeros(n, dtype=np.float32); oracle[big] = 1.0
    m, K = top_mask(oracle, 0.05)
    rec = m[big].mean()
    exp = min(1.0, K / len(big))
    good = abs(rec - exp) < 1e-12
    ok &= good
    lines.append(f"oracle |GT|>K k=0.05: recall={rec:.6f} expected={exp:.6f} -> {'PASS' if good else 'FAIL'}")
    lines.append("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return lines, ok


# --------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------

def encode_category_queries(bb: str, cats: list[dict]) -> np.ndarray:
    import torch
    from encode import encode_queries, load_query_encoder
    tm = load_query_encoder()
    try:
        q = encode_queries([c["query"] for c in cats], bb, text_model=tm)
    finally:
        del tm
        torch.cuda.empty_cache()
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float32)


def scores_trained(tiles, cand, offsets, q: np.ndarray) -> np.ndarray:
    """Cosine of every candidate pixel's Arm-C vector with each query -> (N, n_query)."""
    N = int(offsets[-1])
    out = np.empty((N, q.shape[0]), dtype=np.float32)
    for i, t in enumerate(tiles):
        t0 = time.time()
        pos = int(offsets[i])
        with rasterio.open(t.proj_path) as src:
            for r0 in range(0, t.height, ROW_CHUNK):
                h = min(ROW_CHUNK, t.height - r0)
                mk = cand[i][r0:r0 + h]
                if not mk.any():
                    continue
                arr = src.read(window=Window(0, r0, t.width, h))       # (256, h, w)
                v = arr[:, mk].T.astype(np.float32)                   # (n, 256), row-major order
                if not np.isfinite(v).all():
                    raise SystemExit(f"{t.name}: non-finite Arm-C values inside candidate mask")
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                out[pos:pos + len(v)] = v @ q.T
                pos += len(v)
        assert pos == offsets[i + 1], t.name
        log(f"  T {t.name}: {int(offsets[i+1]-offsets[i])} px ({time.time()-t0:.0f}s)")
    return out


def read_raw_normalised(bb: str, t: Tile, mk: np.ndarray, stats) -> np.ndarray:
    """Raw backbone vectors for the candidate pixels of a tile, in candidate order, normalised
    for the prototype arm (AE: dequantize + L2; TESSERA: z-score dedup stats)."""
    from dequantize import dequantize_ae
    from normalize import apply_zscore
    with rasterio.open(t.raw_path) as src:
        arr = src.read(window=Window(t.col_off, t.row_off, t.width, t.height))  # (B, h, w)
    v = arr[:, mk].T                                                            # (n, B)
    del arr
    if bb == "ae":
        v = dequantize_ae(v)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
    else:
        v = apply_zscore(v.astype(np.float32), stats)
    v = v.astype(np.float32)
    if not np.isfinite(v).all():
        raise SystemExit(f"{t.name}: non-finite raw values inside candidate mask")
    return v


def preflight_reprojection(bb: str, tiles, cand, n: int = 300, seed: int = 0) -> float:
    """Re-project random candidate pixels from raw with the dedup checkpoint's trained head
    and compare with the stored Arm-C vector (cosine >= 0.9999 required)."""
    import run_landuse_probe as rlp
    from dequantize import dequantize_ae
    from normalize import apply_zscore, load_stats
    rlp.CHECKPOINTS[bb] = CKPTS[bb]   # same override as build_sdm_arm_c_tessera_v1dedup.py
    head, ckpt_cfg = rlp.load_trained_head(bb)
    rng = np.random.default_rng(seed)
    cos_all = []
    for i in rng.choice(len(tiles), size=min(3, len(tiles)), replace=False):
        t = tiles[i]
        rr, cc = np.where(cand[i])
        if len(rr) == 0:
            continue
        pick = rng.choice(len(rr), size=min(n // 3, len(rr)), replace=False)
        with rasterio.open(t.raw_path) as rs, rasterio.open(t.proj_path) as ps:
            for j in pick:
                r, c = int(rr[j]), int(cc[j])
                raw = rs.read(window=Window(t.col_off + c, t.row_off + r, 1, 1))[:, 0, 0]
                stored = ps.read(window=Window(c, r, 1, 1))[:, 0, 0].astype(np.float32)
                x = raw[None, :]
                if bb == "ae":
                    x = dequantize_ae(x)
                else:
                    assert ckpt_cfg["data"]["norm_stats_path"].endswith("norm_stats_tessera_pixels_dedup.json")
                    x = apply_zscore(x.astype(np.float32), load_stats(ROOT / ckpt_cfg["data"]["norm_stats_path"]))
                p = rlp.project(head, x.astype(np.float32))[0]
                cos_all.append(float(p @ stored / (np.linalg.norm(p) * np.linalg.norm(stored))))
    return float(np.min(cos_all))


# --------------------------------------------------------------------------------------------
# Run per backbone (Phase 3)
# --------------------------------------------------------------------------------------------

def run_backbone(bb: str, write_fields: bool = True) -> None:
    from normalize import load_stats
    run_provenance_gate()
    cats = load_categories()
    poi_sets, _ = category_poi_sets(cats)
    t_all = time.time()
    tiles = list_tiles(bb)
    cand = candidate_masks(tiles)
    offsets, lut = flat_index(tiles, cand)
    N = int(offsets[-1])
    log(f"{bb}: {len(tiles)} tiles, N={N}")

    min_cos = preflight_reprojection(bb, tiles, cand)
    log(f"{bb}: preflight re-projection min cosine = {min_cos:.6f}")
    if min_cos < 0.9999:
        raise SystemExit(f"{bb}: preflight re-projection cosine {min_cos} < 0.9999 - Arm-C field "
                         f"does not match checkpoint {CKPTS[bb]}, abort.")

    gt = build_gt(bb, cats, tiles, cand, poi_sets)
    gidx = {c["id"]: gt_global_idx(gt[c["id"]]["df"], lut) for c in cats}
    status = {c["id"]: status_of(gt[c["id"]]) for c in cats}

    # --- T scores
    q = encode_category_queries(bb, cats)
    log(f"{bb}: queries encoded {q.shape}")
    sT = scores_trained(tiles, cand, offsets, q)

    # --- P scores (raw space), prototypes from train-cell GT pixels
    stats = load_stats(TESSERA_STATS) if bb == "tessera" else None
    protos, n_proto = {}, {}
    # collect raw vectors of train GT pixels first (tile by tile), then score all candidates
    sP = np.full((N, len(cats)), np.nan, dtype=np.float32)
    raw_sums = {c["id"]: None for c in cats}
    n_train = {c["id"]: 0 for c in cats}
    train_sets = {c["id"]: gidx[c["id"]][(gt[c["id"]]["df"]["split"] == "train").to_numpy()] for c in cats}
    tile_vecs = []
    for i, t in enumerate(tiles):
        t0 = time.time()
        v = read_raw_normalised(bb, t, cand[i], stats)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        for c in cats:
            ts = train_sets[c["id"]]
            sel = ts[(ts >= lo) & (ts < hi)] - lo
            if len(sel):
                s = v[sel].sum(axis=0, dtype=np.float64)
                raw_sums[c["id"]] = s if raw_sums[c["id"]] is None else raw_sums[c["id"]] + s
                n_train[c["id"]] += len(sel)
        tile_vecs.append(v)
        log(f"  P-read {t.name}: {hi-lo} px ({time.time()-t0:.0f}s)")
    for j, c in enumerate(cats):
        if n_train[c["id"]] == 0:
            protos[c["id"]] = None
            continue
        p = (raw_sums[c["id"]] / n_train[c["id"]]).astype(np.float32)
        protos[c["id"]] = p / np.linalg.norm(p)
        n_proto[c["id"]] = n_train[c["id"]]
    P = np.stack([protos[c["id"]] if protos[c["id"]] is not None else
                  np.zeros(tile_vecs[0].shape[1], np.float32) for c in cats])      # (n_cat, B)
    for i, v in enumerate(tile_vecs):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        sP[lo:hi] = (v / nv) @ P.T
    del tile_vecs
    for j, c in enumerate(cats):
        if protos[c["id"]] is None:
            sP[:, j] = np.nan

    # --- self-test on the real candidate space (random + oracle), per category with GT
    rng = np.random.default_rng(0)
    st_lines, st_ok = [], True
    rnd = rng.random(N).astype(np.float32)
    for c in cats:
        g = gidx[c["id"]]
        if len(g) == 0:
            continue
        orc = np.zeros(N, dtype=np.float32); orc[g] = 1.0
        for k in KS:
            m, K = top_mask(rnd, k)
            rr = m[g].mean(); sd = math.sqrt(k * (1 - k) / len(g))
            ok_r = abs(rr - K / N) < 5 * sd
            m, K = top_mask(orc, k)
            ro = m[g].mean(); ok_o = abs(ro - min(1.0, K / len(g))) < 1e-12
            st_ok &= ok_r and ok_o
            st_lines.append(f"{bb}/{c['id']} k={k}: random {rr:.5f} vs K/N {K/N:.5f} (5sd {5*sd:.5f}) "
                            f"{'PASS' if ok_r else 'FAIL'}; oracle {ro:.6f} vs {min(1.0, K/len(g)):.6f} "
                            f"{'PASS' if ok_o else 'FAIL'}")
    del rnd
    (OUT / f"selftest_real_{bb}.txt").write_text("\n".join(st_lines) + f"\nOK={st_ok}\n")
    log(f"{bb}: real-data selftest OK={st_ok}")
    if not st_ok:
        raise SystemExit(f"{bb}: real-data self-test failed, see selftest_real_{bb}.txt")

    # --- recall + bootstrap
    rows, diffs = [], []
    for j, c in enumerate(cats):
        cid = c["id"]
        df = gt[cid]["df"]
        g = gidx[cid]
        tops = {}
        for k in KS:
            mT, K = top_mask(sT[:, j], k)
            tops[("T", k)] = (mT[g] if len(g) else np.zeros(0, bool), K)
            if protos[cid] is not None:
                mP, _ = top_mask(sP[:, j], k)
                tops[("P", k)] = (mP[g], K)
        for sp in SPLITS:
            sel = np.ones(len(df), bool) if sp == "all" else (df["split"] == sp).to_numpy()
            cells = df["cell"].to_numpy()[sel]
            codes = np.unique(cells)
            base = dict(backbone=bb, category_id=cid, split=sp, n_gt_pixels=int(sel.sum()),
                        n_gt_cells=len(codes), n_pois=gt[cid]["n_pois_with_gt"], status=status[cid])
            w = boot_weights(len(codes)) if len(codes) else None
            for k in KS:
                K = int(math.ceil(k * N))
                rows.append({**base, "arm": "R", "k": k, "recall": K / N, "ci_low": np.nan, "ci_high": np.nan})
                samp = {}
                for arm in ("T", "P"):
                    if (arm, k) not in tops or sel.sum() == 0:
                        rows.append({**base, "arm": arm, "k": k, "recall": np.nan, "ci_low": np.nan, "ci_high": np.nan})
                        continue
                    hit = tops[(arm, k)][0][sel]
                    pt, s = recall_with_boot(hit, cells, w, codes)
                    samp[arm] = (pt, s)
                    lo_, hi_ = ci(s)
                    rows.append({**base, "arm": arm, "k": k, "recall": pt, "ci_low": lo_, "ci_high": hi_})
                if "T" in samp and "P" in samp:
                    d = samp["T"][1] - samp["P"][1]
                    lo_, hi_ = ci(d)
                    diffs.append({**base, "k": k, "diff_T_minus_P": samp["T"][0] - samp["P"][0],
                                  "ci_low": lo_, "ci_high": hi_,
                                  "frac_boot_T_gt_P": float(np.mean(d > 0))})
                else:
                    diffs.append({**base, "k": k, "diff_T_minus_P": np.nan, "ci_low": np.nan,
                                  "ci_high": np.nan, "frac_boot_T_gt_P": np.nan})
    cols = ["backbone", "category_id", "arm", "split", "k", "recall", "ci_low", "ci_high",
            "n_gt_pixels", "n_gt_cells", "n_pois", "status"]
    pd.DataFrame(rows)[cols].to_csv(OUT / f"recall_topk_{bb}.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / f"diffs_recall_topk_{bb}.csv", index=False)
    meta = {"backbone": bb, "N": N, "n_tiles": len(tiles), "checkpoint": str(CKPTS[bb].relative_to(ROOT)),
            "preflight_min_cos": min_cos, "n_proto_train_pixels": n_proto,
            "gt_dropped_outside_candidates": {c["id"]: gt[c["id"]]["n_dropped_outside_candidates"] for c in cats},
            "gt_unknown_tile": gt["_meta"]["n_unknown_tile"],
            "extra_tags": gt["_meta"]["extra_tags"],
            "n_extra_pixels_without_split_cell": gt["_meta"]["n_extra_pixels_without_split_cell"],
            "categories_yaml": str(CATEGORIES_YAML.relative_to(ROOT)),
            "queries": {c["id"]: c["query"] for c in cats},
            "score_T_range": {c["id"]: [float(sT[:, j].min()), float(sT[:, j].max())] for j, c in enumerate(cats)},
            "runtime_s": time.time() - t_all}
    (OUT / f"run_meta_{bb}.json").write_text(json.dumps(meta, indent=2))

    # --- fields for maps: T scores + GT per tile
    if write_fields:
        fg = free_gb()
        if fg < MIN_FREE_GB:
            log(f"{bb}: only {fg:.0f} GB free (<{MIN_FREE_GB}) - fields NOT written")
            (OUT / f"fields_skipped_{bb}.txt").write_text(f"free={fg:.1f}GB\n")
        else:
            write_fields_tifs(bb, cats, tiles, cand, offsets, sT, gidx)
    FIELDS.mkdir(parents=True, exist_ok=True)
    np.save(FIELDS / f"scoresT_{bb}.npy", sT)   # for shared map colour scale
    log(f"{bb}: done in {time.time() - t_all:.0f}s")


def write_fields_tifs(bb, cats, tiles, cand, offsets, sT, gidx) -> None:
    FIELDS.mkdir(parents=True, exist_ok=True)
    gt_flag = {}
    N = int(offsets[-1])
    for c in cats:
        f = np.zeros(N, dtype=np.uint8); f[gidx[c["id"]]] = 1
        gt_flag[c["id"]] = f
    for i, t in enumerate(tiles):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        prof = dict(driver="GTiff", height=t.height, width=t.width, count=1, crs=t.crs,
                    transform=t.transform, compress="lzw", tiled=True, blockxsize=256,
                    blockysize=256, BIGTIFF="IF_SAFER")
        stem = Path(t.name).stem
        for j, c in enumerate(cats):
            d = FIELDS / bb / c["id"]
            d.mkdir(parents=True, exist_ok=True)
            a = np.full((t.height, t.width), np.nan, dtype=np.float32)
            a[cand[i]] = sT[lo:hi, j]
            with rasterio.open(d / f"{stem}_simT.tif", "w", dtype="float32", nodata=np.nan, **prof) as dst:
                dst.write(a, 1)
            gtile = np.full((t.height, t.width), 255, dtype=np.uint8)
            gtile[cand[i]] = gt_flag[c["id"]][lo:hi]
            with rasterio.open(d / f"{stem}_gt.tif", "w", dtype="uint8", nodata=255, **prof) as dst:
                dst.write(gtile, 1)


# --------------------------------------------------------------------------------------------
# Finalize: merged CSVs, AETHER-style table, maps
# --------------------------------------------------------------------------------------------

CAT_ORDER = ["art_antiques", "religious_orgs", "medical_supplies", "sports_clubs", "import_export"]


def finalize() -> None:
    cats = load_categories()
    names = {c["id"]: c["aether_name"] for c in cats}
    parts = [pd.read_csv(OUT / f"recall_topk_{bb}.csv") for bb in BACKBONES if (OUT / f"recall_topk_{bb}.csv").exists()]
    dparts = [pd.read_csv(OUT / f"diffs_recall_topk_{bb}.csv") for bb in BACKBONES if (OUT / f"diffs_recall_topk_{bb}.csv").exists()]
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(OUT / "recall_topk.csv", index=False)
    pd.concat(dparts, ignore_index=True).to_csv(OUT / "diffs_recall_topk.csv", index=False)
    write_table(res, names)
    try:
        make_maps(cats)
    except Exception as e:  # maps must not block the numbers
        log(f"MAPS FAILED: {type(e).__name__}: {e}")
        (OUT / "maps_failed.txt").write_text(f"{type(e).__name__}: {e}\n")


def _fmt(r) -> str:
    if pd.isna(r["recall"]):
        return "—"
    s = f"{100 * r['recall']:.1f}"
    if r["arm"] in ("T", "P") and pd.notna(r["ci_low"]):
        s += f" [{100 * r['ci_low']:.1f}–{100 * r['ci_high']:.1f}]"
    return s


def write_table(res: pd.DataFrame, names: dict) -> None:
    arm_name = {"R": "Zufall", "P": "Prototyp", "T": "Trainiert"}
    L = ["# Recall@Top-k% (%) — AETHER-Kategorien, Stil Tab. 3", "",
         "Werte in Prozent; Klammer = 95-%-Zellebenen-Bootstrap-CI (1000 Resamples, Seed 0). "
         "Zufall = K/N analytisch. Schwelle/Top-Menge immer über ganz London (alle Kandidatenpixel), "
         "Recall über die GT-Pixel des jeweiligen Splits. Status: `thin` = < 50 Test-GT-Pixel oder "
         "< 3 Testzellen, `no_gt` = keine GT-Pixel (Recall undefiniert). "
         "Hinweis: Der Prototyp wird aus den GT-Pixeln der Train-Zellen gebildet; seine train/all-Werte "
         "enthalten daher die eigenen Stützpixel.", ""]
    for sp in ("test", "train", "all"):
        L.append(f"## Split: {sp}" + (" (Hauptblock)" if sp == "test" else ""))
        L.append("")
        for bb in BACKBONES:
            sub = res[(res["backbone"] == bb) & (res["split"] == sp)]
            if sub.empty:
                L += [f"### {bb}: keine Ergebnisse (failed)", ""]
                continue
            st = {cid: sub[sub["category_id"] == cid]["status"].iloc[0] for cid in CAT_ORDER}
            ngt = {cid: int(sub[sub["category_id"] == cid]["n_gt_pixels"].iloc[0]) for cid in CAT_ORDER}
            ncell = {cid: int(sub[sub["category_id"] == cid]["n_gt_cells"].iloc[0]) for cid in CAT_ORDER}
            L.append(f"### {bb}")
            L.append("")
            L.append("| k | Arm | " + " | ".join(names[c] for c in CAT_ORDER) + " |")
            L.append("|---|---|" + "---|" * len(CAT_ORDER))
            L.append("| | Status | " + " | ".join(f"**{st[c]}**" for c in CAT_ORDER) + " |")
            L.append("| | GT-Pixel / Zellen | " + " | ".join(f"{ngt[c]} / {ncell[c]}" for c in CAT_ORDER) + " |")
            for k in KS:
                for arm in ("R", "P", "T"):
                    cells = []
                    for cid in CAT_ORDER:
                        r = sub[(sub["category_id"] == cid) & (sub["arm"] == arm) & (np.isclose(sub["k"], k))]
                        cells.append(_fmt(r.iloc[0]) if len(r) else "—")
                    L.append(f"| {int(k * 100)}% | {arm_name[arm]} | " + " | ".join(cells) + " |")
            L.append("")
    (OUT / "table_aether_style.md").write_text("\n".join(L), encoding="utf-8")
    log(f"written {OUT / 'table_aether_style.md'}")


def make_maps(cats) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from rasterio.warp import Resampling
    from maps import (DPI, GRID_CRS, _extent, build_common_grid, mosaic_to_common_grid)

    transform, shape, aoi_gdf, aoi_mask = build_common_grid()
    scores = {bb: np.load(FIELDS / f"scoresT_{bb}.npy", mmap_mode="r") for bb in BACKBONES
              if (FIELDS / f"scoresT_{bb}.npy").exists()}
    limits = []
    maps_dir = OUT / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for j, c in enumerate(cats):
        union = np.concatenate([np.asarray(s[:, j]) for s in scores.values()])
        vmin, vmax = (float(x) for x in np.percentile(union, [1, 99]))
        limits.append({"category_id": c["id"], "query": c["query"], "vmin_p1": vmin, "vmax_p99": vmax,
                       "n_pixels_union": int(union.size), "backbones": ",".join(scores)})
        del union
        for bb in scores:
            d = FIELDS / bb / c["id"]
            sim = mosaic_to_common_grid(sorted(d.glob("*_simT.tif")), shape, transform, GRID_CRS,
                                        Resampling.average, fill=np.nan)
            sim = np.where(aoi_mask, sim, np.nan)
            gtg = mosaic_to_common_grid(sorted(d.glob("*_gt.tif")), shape, transform, GRID_CRS,
                                        Resampling.max, fill=255)
            gtf = gtg.astype(np.float32); gtf[gtg == 255] = np.nan; gtf[~aoi_mask] = np.nan
            n_gt_cells50 = int(np.nansum(gtf == 1))

            fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=DPI)
            cmap = plt.get_cmap("magma").copy(); cmap.set_bad((0, 0, 0, 0))
            im = axes[0].imshow(sim, cmap=cmap, vmin=vmin, vmax=vmax, extent=_extent(transform, shape))
            cb = fig.colorbar(im, ax=axes[0], shrink=0.75, pad=0.02); cb.set_label("cosine similarity")
            axes[0].set_title(f"{bb} — T: \"{c['query']}\"", fontsize=9)
            gcm = ListedColormap(["white", "green"]); gcm.set_bad((0, 0, 0, 0))
            axes[1].imshow(gtf, cmap=gcm, vmin=0, vmax=1, extent=_extent(transform, shape),
                           interpolation="nearest")
            gt_title = f"GT: {', '.join(c['osm_classes'] + c['extra_osm_tags']) or '— keine Klasse (no_gt)'}"
            axes[1].set_title(gt_title + f"\n(50-m-Kartenzellen mit ≥1 GT-Pixel: {n_gt_cells50})", fontsize=8)
            for ax in axes:
                aoi_gdf.boundary.plot(ax=ax, color="black", linewidth=0.4)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            out = maps_dir / f"recall_topk_{bb}_{c['id']}.png"
            fig.savefig(out, dpi=DPI, bbox_inches="tight", transparent=True)
            plt.close(fig)
            log(f"  map {out.name}")
    pd.DataFrame(limits).to_csv(OUT / "map_scale_limits.csv", index=False)


def main():
    global OUT, FIELDS, CATEGORIES_YAML
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["feasibility", "selftest", "run", "finalize"])
    ap.add_argument("--backbone", choices=BACKBONES)
    ap.add_argument("--no-fields", action="store_true")
    ap.add_argument("--categories", default=str(CATEGORIES_YAML),
                    help="category YAML (default: pre-registered queries/aether_categories.yaml)")
    ap.add_argument("--out", default=str(OUT), help="output dir (default: outputs/recall_topk)")
    a = ap.parse_args()
    CATEGORIES_YAML = Path(a.categories).resolve()
    OUT = Path(a.out).resolve()
    FIELDS = OUT / "fields"
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "feasibility":
        feasibility()
    elif a.cmd == "selftest":
        lines, ok = selftest()
        print("\n".join(lines))
        (OUT / "selftest_synthetic.txt").write_text("\n".join(lines) + "\n")
        sys.exit(0 if ok else 1)
    elif a.cmd == "run":
        if not a.backbone:
            ap.error("run needs --backbone")
        run_backbone(a.backbone, write_fields=not a.no_fields)
    elif a.cmd == "finalize":
        finalize()


if __name__ == "__main__":
    main()
