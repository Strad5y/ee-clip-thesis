"""
check_tile_overlap_side_effects.py

Separate Pruefaufgabe (2026-09-17, Jakob) - UNABHAENGIG von INSTRUCTIONS_text_query.md, nichts
an bestehenden Ergebnissen/Artefakten veraendert. Nur lesen und berichten.

Teil 1 (Land-Use TESSERA): wendet die in der Text-Query-Aufgabe gefundene Kachel-Ueberlappung
(spatial_join.py: TESSERA-Kacheln ueberlappen sich, 283 km^2, davon 257 km^2 innerhalb
derselben UTM-Zone - siehe MANIFEST.md "Text-Query-Masken, Phase 2") und die dort beschlossene
Eindeutigkeitsregel (naechster Kachelmittelpunkt gewinnt, Tie-Break ueber Dateinamen) auf
landuse_pairs_tessera.parquet und pixels_tessera.parquet an: wie viele Zeilen sind Dopplungen
eines Weltorts aus zwei Kacheln, landen Dopplungspaare im selben Split, und wie veraendern sich
die BESTEHENDEN Test-Kennzahlen der Land-Use-Probe (reproduziert deterministisch aus den
bestehenden Artefakten - LogisticRegression+lbfgs ist bei fixen Daten deterministisch, siehe
run_landuse_probe.py-Docstring -, KEIN neues Training), wenn nur die deduplizierten Testpixel
gezaehlt werden.

Teil 2 (SDM): prueft build_sdm_pooled_embeddings.py::pool_backbone_arm() - die Funktion loopt
ueber Rohkacheln UND akkumuliert pro LSOA ueber ALLE Kacheln ohne jede Kachel-Ueberlappungs-
Dedup. Prueft, wie viele LSOAs Pixel aus mindestens zwei sich ueberlappenden Kacheln bekommen
(= potenzielle Doppelzaehlung in der coverage-gewichteten Summe), und ob die dabei verwendeten
Raster (Arm A: rohe Kacheln, Arm B/C: die Arm-C-Felder aus Phase 1 der Text-Query-Aufgabe) auf
das AOI-Polygon oder nur dessen Bounding Box beschraenkt sind.

Hinweis zur Begrifflichkeit: "Fund-/Hintergrundpunkte" (klassische Species-Distribution-
Modelling-Terminologie) kommt in diesem Repo nirgends vor (grep negativ, siehe Bericht) - die
SDM-Pipeline hier ("Socioeconomic Distribution Mapping", INSTRUCTIONS_sdm.md) arbeitet
ausschliesslich mit LSOA-POLYGONEN (Census-Zielvariable), nicht mit Punkten. Dieses Skript
beantwortet Frage 2 deshalb fuer LSOA-Polygone (die naechstliegende tatsaechlich existierende
Struktur) und markiert diese Ersetzung explizit im Bericht - keine eigene Entscheidung ueber
die inhaltliche Absicht der Frage.
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import box
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
import spatial_join as sj
from dequantize import dequantize_ae
from normalize import ZSCORE, apply_zscore, load_stats
from eval_landuse import linear_probe
from run_landuse_probe import (
    EXCLUDED_CLASS, load_backbone_yaml_config, load_trained_head, project, resolve_normalization,
)

COMMON_CRS = "EPSG:32630"
ROW_KEY_MULT = 1_000_000  # col < 1e6 fuer alle vorkommenden Kacheln - sichere Kombination row*MULT+col


# ---------------------------------------------------------------------------
# Gemeinsame Kachel-Geometrie-Hilfen (leichtgewichtig - nur Bounds/Transform/CRS,
# KEIN Array-Read, anders als src/eval/text_query/masks.py::_TileData)
# ---------------------------------------------------------------------------

def _tile_meta(region: str = "london", backbone: str = "tessera") -> dict:
    meta = {}
    for p in sj.get_backbones(region)[backbone]:
        with rasterio.open(p) as src:
            meta[p.name] = {"path": p, "transform": src.transform, "crs": src.crs,
                             "bounds": src.bounds, "shape": src.shape}
    return meta


def _reprojected_bbox_polygon(bounds, src_crs, dst_crs):
    poly = box(*bounds).segmentize(500)
    return gpd.GeoDataFrame({"geometry": [poly]}, crs=src_crs).to_crs(dst_crs).geometry.iloc[0]


def _tile_center_common(bounds, crs):
    cx, cy = (bounds.left + bounds.right) / 2, (bounds.bottom + bounds.top) / 2
    tr = Transformer.from_crs(str(crs), COMMON_CRS, always_xy=True)
    return tr.transform(cx, cy)


def find_overlap_candidates(meta: dict) -> list:
    names = list(meta.keys())
    footprints = {n: _reprojected_bbox_polygon(meta[n]["bounds"], str(meta[n]["crs"]), COMMON_CRS)
                  for n in names}
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if footprints[names[i]].intersects(footprints[names[j]]):
                pairs.append((names[i], names[j]))
    return pairs


# ---------------------------------------------------------------------------
# Teil 1: Dopplungs-Analyse auf einem Pixel-Datensatz (landuse_pairs_tessera / pixels_tessera)
# ---------------------------------------------------------------------------

def find_duplicate_rows(df: pd.DataFrame, meta: dict, candidates: list) -> pd.DataFrame:
    """df braucht mindestens die Spalten tile,row,col,split. Gibt eine Zeile pro gefundenem
    Dopplungspaar zurueck (derselbe Weltort, einmal aus jeder der zwei Kacheln), inkl. welche
    der zwei Zeilen laut Eindeutigkeitsregel (naechster Kachelmittelpunkt, Tie-Break
    Dateiname) die 'verlierende' (zu entfernende) waere."""
    by_tile = {t: g[["row", "col", "split"]].reset_index(drop=True) for t, g in df.groupby("tile")}
    key_by_tile = {}
    for t, g in by_tile.items():
        keys = g["row"].to_numpy(dtype=np.int64) * ROW_KEY_MULT + g["col"].to_numpy(dtype=np.int64)
        order = np.argsort(keys)
        key_by_tile[t] = (keys[order], g["split"].to_numpy()[order])

    records = []
    for a, b in candidates:
        if a not in by_tile or b not in by_tile:
            continue
        ma, mb = meta[a], meta[b]
        ga = by_tile[a]

        bbox_b_in_a = _reprojected_bbox_polygon(mb["bounds"], str(mb["crs"]), str(ma["crs"]))
        minx, miny, maxx, maxy = bbox_b_in_a.bounds
        ha, wa = ma["shape"]
        r0, c0 = rasterio.transform.rowcol(ma["transform"], minx, maxy)
        r1, c1 = rasterio.transform.rowcol(ma["transform"], maxx, miny)
        r0, r1 = sorted((max(r0, 0), min(r1, ha)))
        c0, c1 = sorted((max(c0, 0), min(c1, wa)))
        if r0 >= r1 or c0 >= c1:
            continue

        rows_a = ga["row"].to_numpy()
        cols_a = ga["col"].to_numpy()
        splits_a = ga["split"].to_numpy()
        in_window = (rows_a >= r0) & (rows_a < r1) & (cols_a >= c0) & (cols_a < c1)
        if not in_window.any():
            continue
        rows_w, cols_w, splits_w = rows_a[in_window], cols_a[in_window], splits_a[in_window]

        xs, ys = rasterio.transform.xy(ma["transform"], rows_w, cols_w)
        xs, ys = np.asarray(xs), np.asarray(ys)
        to_b = Transformer.from_crs(str(ma["crs"]), str(mb["crs"]), always_xy=True)
        xb, yb = to_b.transform(xs, ys)
        rb, cb = rasterio.transform.rowcol(mb["transform"], xb, yb)
        rb, cb = np.asarray(rb), np.asarray(cb)
        hb, wb = mb["shape"]
        in_bounds = (rb >= 0) & (rb < hb) & (cb >= 0) & (cb < wb)
        if not in_bounds.any():
            continue

        sorted_keys_b, split_lookup_b = key_by_tile[b]
        cand_keys = rb[in_bounds].astype(np.int64) * ROW_KEY_MULT + cb[in_bounds].astype(np.int64)
        pos = np.searchsorted(sorted_keys_b, cand_keys)
        pos = np.clip(pos, 0, len(sorted_keys_b) - 1)
        member = sorted_keys_b[pos] == cand_keys
        if not member.any():
            continue

        idx_in_window = np.where(in_bounds)[0][member]
        a_rows_c = rows_w[idx_in_window]
        a_cols_c = cols_w[idx_in_window]
        a_split_c = splits_w[idx_in_window]
        b_rows_c = rb[in_bounds][member]
        b_cols_c = cb[in_bounds][member]
        b_split_c = split_lookup_b[pos[member]]
        xs_c, ys_c = xs[idx_in_window], ys[idx_in_window]

        to_common = Transformer.from_crs(str(ma["crs"]), COMMON_CRS, always_xy=True)
        xc, yc = to_common.transform(xs_c, ys_c)
        ca, cb_ = _tile_center_common(ma["bounds"], ma["crs"]), _tile_center_common(mb["bounds"], mb["crs"])
        dist_a = np.hypot(np.asarray(xc) - ca[0], np.asarray(yc) - ca[1])
        dist_b = np.hypot(np.asarray(xc) - cb_[0], np.asarray(yc) - cb_[1])
        a_loses = (dist_a > dist_b) | ((dist_a == dist_b) & (a > b))

        for k in range(len(a_rows_c)):
            records.append({
                "tile_a": a, "row_a": int(a_rows_c[k]), "col_a": int(a_cols_c[k]), "split_a": a_split_c[k],
                "tile_b": b, "row_b": int(b_rows_c[k]), "col_b": int(b_cols_c[k]), "split_b": b_split_c[k],
                "loser_tile": a if a_loses[k] else b,
                "same_split": bool(a_split_c[k] == b_split_c[k]),
            })

    return pd.DataFrame(records)


def analyze_landuse_or_pixels(path: Path, meta: dict, candidates: list, label: str) -> pd.DataFrame:
    print(f"\n=== {label}: {path} ===")
    df = pd.read_parquet(path, columns=["tile", "row", "col", "split"])
    print(f"  {len(df)} Zeilen gesamt")
    dup = find_duplicate_rows(df, meta, candidates)
    n_pairs = len(dup)
    n_rows_affected = 2 * n_pairs  # jede Dopplung betrifft 2 Zeilen (Gewinner + Verlierer)
    print(f"  Dopplungs-Weltorte (Paare): {n_pairs} ({n_rows_affected} betroffene Zeilen, "
          f"{n_pairs} wuerden bei Dedup nach der Eindeutigkeitsregel entfernt)")
    if n_pairs > 0:
        n_same_split = int(dup["same_split"].sum())
        print(f"  Davon im selben Split: {n_same_split}/{n_pairs} ({100*n_same_split/n_pairs:.1f}%)")
        print(f"  Splits der ABWEICHENDEN Paare:")
        mismatched = dup[~dup["same_split"]]
        if len(mismatched) > 0:
            print(mismatched.groupby(["split_a", "split_b"]).size().to_string())
        cross_zone = dup[dup["tile_a"].str.extract(r"grid_(-?[\d.]+)_")[0].astype(float).lt(0) !=
                          dup["tile_b"].str.extract(r"grid_(-?[\d.]+)_")[0].astype(float).lt(0)]
        print(f"  Zonenuebergreifend (30N/31N) unter den Dopplungen: {len(cross_zone)}/{n_pairs}")
    return dup


# ---------------------------------------------------------------------------
# Teil 1c: bestehende Test-Kennzahlen mit den vorhandenen (deterministisch reproduzierten)
# Vorhersagen neu auf dedupliziertem Testset auswerten - KEIN Neutraining (gleiche Daten,
# gleiches C, gleicher Fit - nur deterministische Reproduktion, siehe run_landuse_probe.py-
# Docstring: LogisticRegression+lbfgs ist bei fixen Daten deterministisch).
# ---------------------------------------------------------------------------

def recompute_tessera_test_metrics(dup_landuse: pd.DataFrame, reference_c: float) -> None:
    print("\n=== Reproduktion tessera_trainiert_projiziert (deterministisch, kein Neutraining) ===")
    path = cfg.DATA / "processed" / "landuse_pairs_tessera.parquet"
    df = pd.read_parquet(path, columns=["code_2021", "split", "embedding", "tile", "row", "col"])
    df = df[(df["code_2021"] != EXCLUDED_CLASS) & (df["split"] != "missing")].reset_index(drop=True)
    print(f"  {len(df)} Zeilen nach Filter (EXCLUDED_CLASS={EXCLUDED_CLASS!r}, split!='missing')")

    emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    y = df["code_2021"].to_numpy()
    split = df["split"].to_numpy()

    yaml_cfg = load_backbone_yaml_config("tessera")
    mode, stats_path = resolve_normalization(yaml_cfg, "configs/tessera_v1.yaml")
    trained_head, ckpt_cfg = load_trained_head("tessera")
    ckpt_mode, ckpt_stats_path = resolve_normalization(ckpt_cfg, "Checkpoint-Config (tessera)")
    assert mode == ckpt_mode and stats_path == ckpt_stats_path, "Normalisierung config vs. checkpoint weicht ab"
    stats = load_stats(stats_path)
    emb_z = apply_zscore(emb, stats)
    X = project(trained_head, emb_z)

    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    res = linear_probe(X[train_mask], y[train_mask], X[val_mask], y[val_mask],
                        X[test_mask], y[test_mask], C=reference_c, max_iter=1000, return_fit=True)
    print(f"  Reproduziert (C={reference_c}): test_oa={res['test_oa']:.6f} "
          f"test_macro_f1={res['test_macro_f1']:.6f}")
    print(f"  (Referenz aus landuse_probe_results_tessera_dedup.csv: "
          f"test_oa=0.611942 test_macro_f1=0.249801 - bestaetigt Reproduktion, falls nahe genug)")

    test_df = df[test_mask][["tile", "row", "col"]].reset_index(drop=True)
    pred_test = res["pred_test"]
    y_test = y[test_mask]
    assert len(test_df) == len(pred_test)

    loser_keys = set()
    for _, r in dup_landuse.iterrows():
        loser_tile = r["loser_tile"]
        if loser_tile == r["tile_a"]:
            loser_keys.add((r["tile_a"], r["row_a"], r["col_a"]))
        else:
            loser_keys.add((r["tile_b"], r["row_b"], r["col_b"]))

    test_keys = list(zip(test_df["tile"], test_df["row"], test_df["col"]))
    is_loser = np.array([k in loser_keys for k in test_keys])
    n_loser_in_test = int(is_loser.sum())
    print(f"  Test-Zeilen, die Dopplungs-Verlierer sind: {n_loser_in_test}/{len(test_df)} "
          f"({100*n_loser_in_test/len(test_df):.3f}%)")

    keep = ~is_loser
    oa_dedup = accuracy_score(y_test[keep], pred_test[keep])
    f1_dedup = f1_score(y_test[keep], pred_test[keep], average="macro")
    oa_full = accuracy_score(y_test, pred_test)
    f1_full = f1_score(y_test, pred_test, average="macro")
    print(f"  Test OA:       voll={oa_full:.6f}  dedupliziert={oa_dedup:.6f}  "
          f"Delta={oa_dedup - oa_full:+.6f}")
    print(f"  Test Macro-F1: voll={f1_full:.6f}  dedupliziert={f1_dedup:.6f}  "
          f"Delta={f1_dedup - f1_full:+.6f}")


# ---------------------------------------------------------------------------
# Teil 2: SDM - LSOA-Polygone in Kachel-Ueberlappungen (Ersatz fuer "Fund-/Hintergrundpunkte",
# siehe Modul-Docstring) + AOI-Bbox-vs-Polygon-Check fuer die gepoolten Raster.
# ---------------------------------------------------------------------------

def check_sdm_lsoa_tile_overlap(backbone: str = "tessera") -> None:
    print(f"\n=== SDM: LSOA-Polygone in Kachel-Ueberlappungen ({backbone}, Arm A - rohe Kacheln) ===")
    from build_sdm_lsoa_splits import get_aoi_utm_reference, load_lsoa_geometries_for_london
    from build_sdm_targets import load_ts062, CODE_COL

    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    ts062 = load_ts062()
    ts062_codes = set(ts062[CODE_COL])
    lsoa_national = load_lsoa_geometries_for_london(aoi)
    lsoa_utm = lsoa_national[lsoa_national["LSOA21CD"].isin(ts062_codes)].copy()
    lsoa_utm = lsoa_utm.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]].to_crs(utm_crs)
    print(f"  {len(lsoa_utm)} LSOAs (wie build_sdm_pooled_embeddings.py::main())")

    meta = _tile_meta("london", backbone)
    candidates = set(frozenset(p) for p in find_overlap_candidates(meta))
    print(f"  {len(candidates)} ueberlappende Kachelpaare ({backbone})")

    lsoa_by_crs = {}
    lsoa_tiles = {code: set() for code in lsoa_utm["lsoa_code"]}
    for name, m in meta.items():
        crs = str(m["crs"])
        if crs not in lsoa_by_crs:
            lsoa_by_crs[crs] = lsoa_utm.to_crs(crs) if crs != str(utm_crs) else lsoa_utm
        lsoa_native = lsoa_by_crs[crs]
        b = m["bounds"]
        cand = lsoa_native.cx[b.left:b.right, b.bottom:b.top]
        for code in cand["lsoa_code"]:
            lsoa_tiles[code].add(name)

    n_multi_tile = 0
    n_overlap_affected = 0
    for code, tiles in lsoa_tiles.items():
        if len(tiles) < 2:
            continue
        n_multi_tile += 1
        tiles_l = list(tiles)
        affected = any(frozenset((tiles_l[i], tiles_l[j])) in candidates
                        for i in range(len(tiles_l)) for j in range(i + 1, len(tiles_l)))
        if affected:
            n_overlap_affected += 1

    print(f"  LSOAs mit Kandidaten-Kacheln aus >=2 Kacheln: {n_multi_tile}/{len(lsoa_utm)}")
    print(f"  Davon mit mindestens 2 TATSAECHLICH sich ueberlappenden Kacheln "
          f"(potenzielle Doppelzaehlung in pool_backbone_arm()): "
          f"{n_overlap_affected}/{len(lsoa_utm)} ({100*n_overlap_affected/len(lsoa_utm):.1f}%)")
    print("  Mechanismus: pool_backbone_arm() akkumuliert 'sum_all'/'sum_pure' UEBER ALLE Kacheln "
          "hinweg (kein Dedup zwischen Kacheln) - jede betroffene LSOA bekommt fuer den in beiden "
          "Kacheln liegenden Teil ihrer Flaeche die Pixel/Coverage ZWEIMAL in die coverage-"
          "gewichtete Summe. Gilt identisch fuer Arm A/B/C (dieselbe Akkumulationslogik).")


def check_sdm_aoi_crop() -> None:
    print("\n=== SDM: AOI-Bbox-vs-Polygon fuer die gepoolten Raster ===")
    print("  Arm B/C: nutzen data/interim/sdm_projected/{backbone}_arm{b,c}/*.tif - bereits in "
          "der Text-Query-Aufgabe (Phase 2) verifiziert: NUR auf die Bounding Box des AOI-"
          "Polygons zugeschnitten, nicht das Polygon selbst (siehe MANIFEST.md, "
          "check_aoi_is_bbox_crop()).")
    print("  Arm A: nutzt die ROHEN, UNBESCHNITTENEN Kacheln direkt (spatial_join.get_backbones) "
          "- gar kein AOI-Zuschnitt (weder Bounding Box noch Polygon). Die effektive raeumliche "
          "Begrenzung kommt hier ausschliesslich aus der LSOA-Polygon-Geometrie selbst (echte "
          "Nomis-London-LSOAs) - kein bekannter Mechanismus, der Pixel ausserhalb einer LSOA "
          "faelschlich mitzaehlen wuerde, unabhaengig vom AOI-Bbox/Polygon-Unterschied.")


def main():
    t0 = time.time()
    print("HINWEIS: 'Fund-/Hintergrundpunkte' (Species-Distribution-Modelling-Begriff) kommt in "
          "diesem Repo nicht vor (grep negativ) - die SDM-Pipeline hier ist LSOA-Polygon-basiert, "
          "keine Punkte. Teil 2 unten prueft deshalb LSOA-Polygone statt Punkten - bitte bestaetigen, "
          "ob das die gemeinte Struktur ist.")

    meta_t = _tile_meta("london", "tessera")
    candidates_t = find_overlap_candidates(meta_t)
    print(f"\n{len(candidates_t)} ueberlappende TESSERA-Kachelpaare (identisch zu Phase 1 der "
          f"Text-Query-Aufgabe)")

    dup_landuse = analyze_landuse_or_pixels(
        cfg.DATA / "processed" / "landuse_pairs_tessera.parquet", meta_t, candidates_t,
        "Teil 1a: landuse_pairs_tessera.parquet")
    dup_pixels = analyze_landuse_or_pixels(
        cfg.DATA / "processed" / "pixels_tessera.parquet", meta_t, candidates_t,
        "Teil 1b: pixels_tessera.parquet")

    recompute_tessera_test_metrics(dup_landuse, reference_c=10.0)

    check_sdm_lsoa_tile_overlap("tessera")
    check_sdm_lsoa_tile_overlap("ae")
    check_sdm_aoi_crop()

    print(f"\nGesamtlaufzeit: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
