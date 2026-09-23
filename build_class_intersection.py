"""
build_class_intersection.py

Phase-0-Retrieval-Evaluation (2026-09-23, Jakob): speichert die Schnittmenge der Klassen mit
>=50 relevanten Testpixeln in >=3 Testzellen aus BEIDEN Backbones (TESSERA-dedup 210, AE-dedup
215, Schnittmenge 209 - siehe Phase-0-Bericht) als feste Datei, damit beide Backbones
nachweislich dieselben Retrieval-Query-Klassen bekommen. Reine Wiederholung derselben
Zaehlung wie im Phase-0-Bericht, deterministisch (kein Zufall beteiligt).
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from poi_classes import extract_all_classes, KEY_FRAME
from build_sdm_lsoa_splits import get_aoi_utm_reference
from aggregate_pixel_pairs import compute_pixel_cells
from spatial_join import BACKBONES
import config as cfg

MIN_RELEVANT = 50
MIN_CELLS = 3
OUT_PATH = "queries/class_retrieval_intersection_dedup.csv"


def passing_classes(path, tile_backbone, utm_crs, minx, miny):
    df = pd.read_parquet(path, columns=["tile", "row", "col", "split", "pixel_text"])
    test = df[df["split"] == "test"].reset_index(drop=True)
    unique_texts = test["pixel_text"].drop_duplicates()
    parsed = {t: extract_all_classes(t)[0] for t in unique_texts}
    classes_test = test["pixel_text"].map(parsed)
    cells = compute_pixel_cells(test[["tile", "row", "col"]], BACKBONES[tile_backbone], utm_crs, minx, miny)
    merged = test[["tile", "row", "col"]].merge(cells, on=["tile", "row", "col"], how="left")
    cell_id = merged["cell_row"].astype(str) + "_" + merged["cell_col"].astype(str)
    class_to_rows = {}
    for i, cls_set in enumerate(classes_test):
        for c in cls_set:
            class_to_rows.setdefault(c, []).append(i)
    stats = {}
    for c, idxs in class_to_rows.items():
        idxs = np.array(idxs)
        stats[c] = (len(idxs), cell_id.iloc[idxs].nunique())
    passing = {c for c, (n, nc) in stats.items() if n >= MIN_RELEVANT and nc >= MIN_CELLS}
    return passing, stats


def main():
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    pt, stats_t = passing_classes(cfg.DATA / "processed" / "pixels_tessera_dedup.parquet", "tessera", utm_crs, minx, miny)
    pa, stats_a = passing_classes(cfg.DATA / "processed" / "pixels_ae_dedup.parquet", "ae", utm_crs, minx, miny)
    inter = sorted(pt & pa)
    print(f"TESSERA-dedup: {len(pt)}  AE-dedup: {len(pa)}  Schnittmenge: {len(inter)}")
    assert len(inter) == 209, f"Erwartet 209, erhalten {len(inter)} - Abweichung zum Phase-0-Bericht, nicht committen."

    rows = []
    for key, value in inter:
        n_t, nc_t = stats_t[(key, value)]
        n_a, nc_a = stats_a[(key, value)]
        rows.append({"key": key, "value": value, "key_frame": KEY_FRAME.get(key, ""),
                      "n_relevant_test_tessera_dedup": n_t, "n_cells_test_tessera_dedup": nc_t,
                      "n_relevant_test_ae_dedup": n_a, "n_cells_test_ae_dedup": nc_a})
    out = pd.DataFrame(rows).sort_values(["key", "value"]).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Gespeichert: {OUT_PATH} ({len(out)} Zeilen)")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
