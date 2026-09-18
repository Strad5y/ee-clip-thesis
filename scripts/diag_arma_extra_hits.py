"""
diag_arma_extra_hits.py - Ad-hoc Diagnose (nicht Teil der SDM-Pipeline): erklaert, warum Arm A
in build_sdm_pooled_embeddings_dedup.py mehr rohe exact_extract-Treffer zaehlt als Arm B/C,
obwohl die POST-Masken-Pixelzahl pro LSOA identisch ist. Hypothese: die Arm-B/C-Raster sind auf
die AOI-Bounding-Box zugeschnitten (kleiner als die rohe Kachel); ein LSOA-Rand, der geometrisch
knapp AUSSERHALB dieses Zuschnittfensters aber noch INNERHALB der vollen Rohkachel liegt, liefert
gegen die rohe Kachel (Arm A) zusaetzliche exact_extract-Treffer, die gegen das kleinere Arm-B/C-
Raster gar nicht erst entstehen koennen (die Pixel existieren dort schlicht nicht). Diese
Zusatztreffer muessten dann aber - weil ausserhalb der AOI-Bounding-Box - auch ausserhalb des
AOI-POLYGONS liegen und von der Eindeutigkeits+AOI-Maske herausgefiltert werden, was den
identischen POST-Masken-Stand erklaeren wuerde. Wird hier direkt an den Pixelpositionen verifiziert,
nicht nur behauptet.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src" / "eval" / "text_query"))

import geopandas as gpd
import numpy as np
import rasterio
from exactextract import exact_extract

import config as cfg
import spatial_join as sj
from build_sdm_lsoa_splits import get_aoi_utm_reference, load_lsoa_geometries_for_london
from build_sdm_targets import load_ts062, CODE_COL
from build_sdm_pooled_embeddings import raster_paths_for_arm
from build_sdm_pooled_embeddings_dedup import _raw_tile_window, load_mask_for_tile

AOI_PATH = cfg.AOI_DIR / "london.gpkg"


def hits_for_raster(path, cand):
    res = exact_extract(str(path), cand[["lsoa_code", "geometry"]], ["cell_id", "coverage"],
                         include_cols=["lsoa_code"], output="pandas")
    with rasterio.open(path) as src:
        width = src.width
    out = {}
    for _, row in res.iterrows():
        cell_ids = np.asarray(row["band_1_cell_id"])
        if len(cell_ids) == 0:
            continue
        rows_ = cell_ids // width
        cols_ = cell_ids % width
        out[row["lsoa_code"]] = set(zip(rows_.tolist(), cols_.tolist()))
    return out


def diagnose(backbone: str):
    print(f"\n=== {backbone} ===")
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    aoi_gpd = gpd.read_file(AOI_PATH)
    ts062 = load_ts062()
    ts062_codes = set(ts062[CODE_COL])
    lsoa_national = load_lsoa_geometries_for_london(aoi)
    lsoa_utm = lsoa_national[lsoa_national["LSOA21CD"].isin(ts062_codes)].copy()
    lsoa_utm = lsoa_utm.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]].to_crs(utm_crs)

    raw_tiles = sj.get_backbones("london")[backbone]
    arm_b_tiles = {p.stem.replace("_proj256", ""): p for p in raster_paths_for_arm(backbone, "b")}

    lsoa_by_crs = {}
    total_extra = 0
    examples = []
    for raw_path in raw_tiles:
        with rasterio.open(raw_path) as src:
            tile_crs = str(src.crs)
            bounds = src.bounds
            raw_shape = src.shape
        if tile_crs not in lsoa_by_crs:
            lsoa_by_crs[tile_crs] = lsoa_utm.to_crs(tile_crs) if tile_crs != str(utm_crs) else lsoa_utm
        lsoa_native = lsoa_by_crs[tile_crs]
        cand = lsoa_native.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
        if len(cand) == 0:
            continue
        b_path = arm_b_tiles.get(raw_path.stem)
        if b_path is None:
            continue

        hits_a = hits_for_raster(raw_path, cand)
        hits_b = hits_for_raster(b_path, cand)

        window = _raw_tile_window(raw_path, aoi_gpd)
        r0, c0 = int(round(window.row_off)), int(round(window.col_off))
        h, w = int(round(window.height)), int(round(window.width))

        full_mask = load_mask_for_tile(backbone, raw_path, raw_path, "a", aoi_gpd, raw_shape)

        for code, pos_a in hits_a.items():
            pos_b = hits_b.get(code, set())
            # pos_b ist im ARM-B-Grid (Fenster-relativ) - auf Rohkachel-Grid zurueckschieben
            pos_b_in_raw = {(r + r0, c + c0) for r, c in pos_b}
            extra = pos_a - pos_b_in_raw
            if not extra:
                continue
            total_extra += len(extra)
            for (r, c) in extra:
                inside_window = (r0 <= r < r0 + h) and (c0 <= c < c0 + w)
                masked_out = not full_mask[r, c]
                examples.append({
                    "tile": raw_path.name, "lsoa_code": code, "row": r, "col": c,
                    "inside_arm_b_window": inside_window, "masked_out_in_full_mask": masked_out,
                })
        if len(hits_a) or len(hits_b):
            pass

    print(f"  Gesamtzahl 'nur in Arm A' gefundener (lsoa,row,col)-Treffer: {total_extra}")
    if examples:
        import pandas as pd
        df = pd.DataFrame(examples)
        print(f"  Davon ausserhalb des Arm-B-Zuschnittfensters: {int((~df['inside_arm_b_window']).sum())}/{len(df)}")
        print(f"  Davon durch die volle Maske (Eindeutigkeit+AOI) ausgeschlossen: "
              f"{int(df['masked_out_in_full_mask'].sum())}/{len(df)}")
        anomalies = df[df["inside_arm_b_window"] | ~df["masked_out_in_full_mask"]]
        if len(anomalies) > 0:
            print(f"  ANOMALIEN (widersprechen der Hypothese): {len(anomalies)}")
            print(anomalies.head(20).to_string())
        else:
            print("  Hypothese bestaetigt: ALLE 'nur Arm A'-Treffer liegen ausserhalb des "
                  "Arm-B-Fensters UND werden von der vollen Maske korrekt ausgeschlossen.")
    else:
        print("  Keine Differenz auf Positionsebene gefunden.")


if __name__ == "__main__":
    diagnose("ae")
    diagnose("tessera")
