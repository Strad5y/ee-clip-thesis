"""
build_sdm_pooled_embeddings.py

Phase 5 (INSTRUCTIONS_sdm.md): Mean-Pooling der EE-Embeddings je LSOA, mit exactextract
(konsistent zum restlichen Vorgehen - siehe spatial_join.py::_join_polygons, gleicher
["cell_id","coverage"]-Mechanismus). Arm A poolt ueber das rohe Raster, Arm B/C ueber das
projizierte 256-Kanal-Feld aus Phase 4 (build_sdm_projected_field.py). Fuer alle drei Arme
dieselbe Pixelmenge pro LSOA - kein Filter auf POI-Pixel.

Zwei Poolungsvarianten in EINEM Durchlauf berechnet (dieselben exact_extract-Treffer, zwei
laufende Summen):
  - partitionsrein (Hauptlauf): nur Pixel aus Gitterzellen behalten, deren Split (aus
    splits_london.csv) mit dem der LSOA zugewiesenen Split (aus sdm_lsoa_splits_london.csv,
    Variante A) uebereinstimmt - schliesst den in der Diagnose gefundenen Leakage-Kanal.
  - alle Pixel im Polygon (Robustheitscheck): keine Partitionsfilterung, Phase-5-Rohdefinition.

Split-Variante B (sdm_lsoa_splits_london_min60.csv) braucht KEINEN eigenen Poolungslauf - sie
ist eine Teilmenge derselben LSOA-Menge mit identischer Mehrheits-Split-Zuordnung (nur
mehrdeutige LSOAs zusaetzlich verworfen), die gepoolten Vektoren fuer die verbleibenden LSOAs
sind bit-identisch. Variante B wird in Phase 6 durch einen Zeilenfilter auf dieselbe
Ergebnistabelle angewendet, nicht neu gepoolt.

Re-Normalisierung nach dem Pooling (offene Entscheidung, explizit getroffen): NEIN, fuer alle
drei Arme gleich (keine erneute L2-Normalisierung der gepoolten Vektoren). Begruendung siehe
MANIFEST.md - kurz: TESSERAs rohe Kanaele sind z-score-normiert, nicht L2-normiert, eine
nachtraegliche Zwangs-Normierung auf Einheitslaenge haette dort keine natuerliche Bedeutung;
der Phase-6-Vorhersagekopf (MLP+Softmax) verlangt keine Einheitsnorm (anders als der
SigLIP-Pretraining-Loss); "keine Transformation ohne Grund" ist der einfachere Default.
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from pyproj import Transformer

import config as cfg
import spatial_join as sj
from build_sdm_lsoa_splits import (
    CELL_SIZE_M, get_aoi_utm_reference, load_lsoa_geometries_for_london,
)
from build_sdm_targets import load_ts062, CODE_COL
from dequantize import dequantize_ae
from normalize import ZSCORE, apply_zscore, load_stats
from run_landuse_probe import EE_DIMS, LATENT_DIM, load_backbone_yaml_config, resolve_normalization

SPLITS_LONDON = cfg.DATA / "splits" / "splits_london.csv"
LSOA_SPLIT_A = cfg.DATA / "splits" / "sdm_lsoa_splits_london.csv"  # Variante A, Hauptlauf
PROJECTED_DIR = cfg.DATA / "interim" / "sdm_projected"
OUT_DIR = cfg.DATA / "processed"


def load_cell_lookup() -> dict:
    df = pd.read_csv(SPLITS_LONDON)
    return {(r, c): s for r, c, s in zip(df["cell_row"], df["cell_col"], df["split"])}


def raster_paths_for_arm(backbone: str, arm: str) -> list:
    if arm == "a":
        return sj.get_backbones("london")[backbone]
    return sorted((PROJECTED_DIR / f"{backbone}_arm{arm}").glob("*.tif"))


def pool_backbone_arm(backbone: str, arm: str, lsoa_utm: gpd.GeoDataFrame, lsoa_split: dict,
                       cell_lookup: dict, utm_crs, minx: float, miny: float) -> pd.DataFrame:
    tile_paths = raster_paths_for_arm(backbone, arm)
    latent_dim = LATENT_DIM if arm in ("b", "c") else EE_DIMS[backbone]
    needs_dequant = backbone == "ae" and arm == "a"

    if arm == "a":
        yaml_cfg = load_backbone_yaml_config(backbone)
        norm_mode, stats_path = resolve_normalization(yaml_cfg, f"configs/{backbone}_v1.yaml")
        norm_stats = load_stats(stats_path) if norm_mode == ZSCORE else None
    else:
        norm_mode, norm_stats = None, None  # bereits in Phase 4 normalisiert, vor der Projektion

    # Reprojektion je Zone NUR EINMAL (nicht pro Kachel) - es gibt nur 2 CRS ueberhaupt
    # (30N/31N), 50 TESSERA-Kacheln teilen sich dieselbe Zone mehrfach (Muster wie
    # spatial_join.py::join_pois_to_raster-Docstring: wiederholte pyproj-Transformation
    # derselben Polygone war dort schon einmal der dominante Kostenfaktor).
    lsoa_by_crs = {}

    acc = {}  # lsoa_code -> dict mit laufenden Summen

    def get_acc(code):
        if code not in acc:
            acc[code] = {"sum_all": np.zeros(latent_dim, dtype=np.float64), "cov_all": 0.0,
                         "sum_pure": np.zeros(latent_dim, dtype=np.float64), "cov_pure": 0.0,
                         "n_all": 0, "n_pure": 0}
        return acc[code]

    t0 = time.time()
    for ti, tp in enumerate(tile_paths):
        with rasterio.open(tp) as src:
            tile_crs = str(src.crs)
            width = src.width
            transform = src.transform
            bounds = src.bounds

        if tile_crs not in lsoa_by_crs:
            lsoa_by_crs[tile_crs] = lsoa_utm.to_crs(tile_crs) if tile_crs != str(utm_crs) else lsoa_utm
        lsoa_native = lsoa_by_crs[tile_crs]
        cand = lsoa_native.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
        if len(cand) == 0:
            continue

        res = exact_extract(str(tp), cand[["lsoa_code", "geometry"]], ["cell_id", "coverage"],
                             include_cols=["lsoa_code"], output="pandas")
        if len(res) == 0:
            continue

        with rasterio.open(tp) as src:
            arr = src.read()  # (bands, H, W)
        bands = arr.shape[0]
        if needs_dequant:
            arr = np.moveaxis(dequantize_ae(np.moveaxis(arr, 0, -1)), -1, 0)  # (bands,H,W), NaN fuer NoData

        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        reproject = tile_crs != str(utm_crs)
        transformer = Transformer.from_crs(tile_crs, str(utm_crs), always_xy=True) if reproject else None

        for _, row in res.iterrows():
            cell_ids = np.asarray(row["band_1_cell_id"])
            coverages = np.asarray(row["band_1_coverage"], dtype=np.float64)
            if len(cell_ids) == 0:
                continue
            rows_ = cell_ids // width
            cols_ = cell_ids % width

            emb = arr[:, rows_, cols_].T.astype(np.float64)  # (n, bands)
            valid = np.isfinite(emb).all(axis=1)
            if not valid.any():
                continue
            emb, coverages, rows_, cols_ = emb[valid], coverages[valid], rows_[valid], cols_[valid]

            if norm_mode == ZSCORE:
                emb = apply_zscore(emb.astype(np.float32), norm_stats).astype(np.float64)

            xs = a * (cols_ + 0.5) + b * (rows_ + 0.5) + c
            ys = d * (cols_ + 0.5) + e * (rows_ + 0.5) + f
            if reproject:
                xs, ys = transformer.transform(xs, ys)
            cell_col = np.floor((xs - minx) / CELL_SIZE_M).astype(int)
            cell_row = np.floor((ys - miny) / CELL_SIZE_M).astype(int)
            cell_splits = np.array([cell_lookup.get((r, c)) for r, c in zip(cell_row, cell_col)])

            a_ = get_acc(row["lsoa_code"])
            a_["sum_all"] += (coverages[:, None] * emb).sum(axis=0)
            a_["cov_all"] += coverages.sum()
            a_["n_all"] += len(coverages)

            pure_mask = cell_splits == lsoa_split.get(row["lsoa_code"])
            if pure_mask.any():
                a_["sum_pure"] += (coverages[pure_mask, None] * emb[pure_mask]).sum(axis=0)
                a_["cov_pure"] += coverages[pure_mask].sum()
                a_["n_pure"] += int(pure_mask.sum())
        del arr
        if (ti + 1) % 10 == 0 or ti == len(tile_paths) - 1:
            print(f"    [{backbone}/{arm}] Kachel {ti + 1}/{len(tile_paths)} verarbeitet "
                  f"({time.time() - t0:.1f}s)")

    rows = []
    for code, a_ in acc.items():
        rec = {"lsoa_code": code, "n_pixels_all": a_["n_all"], "n_pixels_pure": a_["n_pure"]}
        if a_["cov_all"] > 0:
            vec = a_["sum_all"] / a_["cov_all"]
            for i in range(latent_dim):
                rec[f"all_{i}"] = vec[i]
        if a_["cov_pure"] > 0:
            vec = a_["sum_pure"] / a_["cov_pure"]
            for i in range(latent_dim):
                rec[f"pure_{i}"] = vec[i]
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"  [{backbone}/{arm}] fertig: {len(df)} LSOAs erreicht in {time.time() - t0:.1f}s")
    return df


def main():
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    ts062 = load_ts062()
    ts062_codes = set(ts062[CODE_COL])
    lsoa_national = load_lsoa_geometries_for_london(aoi)
    lsoa_utm = lsoa_national[lsoa_national["LSOA21CD"].isin(ts062_codes)].copy()
    lsoa_utm = lsoa_utm.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]].to_crs(utm_crs)
    print(f"LSOAs (gematcht): {len(lsoa_utm)}")

    split_a = pd.read_csv(LSOA_SPLIT_A)
    lsoa_split = dict(zip(split_a["lsoa_code"], split_a["split"]))
    cell_lookup = load_cell_lookup()

    for backbone in ["ae", "tessera"]:
        for arm in ["a", "b", "c"]:
            print(f"\n=== {backbone} / Arm {arm.upper()} ===")
            df = pool_backbone_arm(backbone, arm, lsoa_utm, lsoa_split, cell_lookup,
                                    utm_crs, minx, miny)
            n_zero_all = int((df["n_pixels_all"] == 0).sum()) if len(df) else len(lsoa_utm)
            n_zero_pure = int((df["n_pixels_pure"] == 0).sum()) if len(df) else len(lsoa_utm)
            n_missing = len(lsoa_utm) - len(df)
            print(f"  LSOAs ganz ohne Treffer (kein Tile-Overlap): {n_missing}")
            print(f"  LSOAs mit 0 Pixeln (alle Pixel im Polygon): {n_zero_all}")
            print(f"  LSOAs mit 0 Pixeln (partitionsrein): {n_zero_pure}")

            out_path = OUT_DIR / f"sdm_pooled_{backbone}_arm{arm}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
