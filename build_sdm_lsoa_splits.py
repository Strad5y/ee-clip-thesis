"""
build_sdm_lsoa_splits.py

Phase 3 (INSTRUCTIONS_sdm.md): ordnet jedes Greater-London-LSOA VOLLSTAENDIG einer Partition
(train/val/test) des bestehenden 1-km-Rasters zu, ueber den Flaechenanteil des LSOA-Polygons
in den jeweils zugeordneten Gitterzellen. Kein LSOA darf ueber Partitionen zerfallen - Grund:
das 1-km-Raster (build_splits_pixels.py) wurde fuer Retrieval/Landuse gebaut und kennt die
LSOA-Grenzen nicht, ohne diese Zuordnung wuerden Pixel desselben LSOA in Train und Test landen
(dieselbe Leakage-Klasse wie bei den Urban-Atlas-Boundary-Pixeln, siehe MANIFEST.md).

Zellgeometrie wird OHNE Raster-Rerun rekonstruiert (Phase-0-Befund): Ursprung
(minx, miny) = london_aoi.to_crs(estimate_utm_crs()).total_bounds[:2], CELL_SIZE_M = 1000.0
(identisch zu build_splits.py/build_splits_pixels.py), cell_row/cell_col->Weltkoordinate ist
die Umkehrung von deren Ganzzahl-Division (col=x-Achse/Easting, row=y-Achse/Northing).

Die Split-Zuordnung haengt nur an der Geometrie, nicht an den NS-SeC-Kategorien - wird daher
EINMAL gerechnet und deckt beide Zielvarianten (9cat/8cat) gleichermassen ab.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

import config as cfg
from build_sdm_targets import load_ts062, CODE_COL

AOI_PATH = cfg.AOI_DIR / "london.gpkg"
LSOA_GPKG = cfg.DATA / "raw" / "sdm" / (
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10_"
    "-672099234420024429.gpkg"
)
SPLITS_LONDON = cfg.DATA / "splits" / "splits_london.csv"

CELL_SIZE_M = 1000.0  # identisch zu build_splits.py/build_splits_pixels.py

OUT_MAJORITY = cfg.DATA / "splits" / "sdm_lsoa_splits_london.csv"
MIN_MAX_FRAC = 0.60  # Schwellwert Variante B - siehe Begruendung im Bericht
OUT_MIN60 = cfg.DATA / "splits" / "sdm_lsoa_splits_london_min60.csv"


def get_aoi_utm_reference() -> tuple:
    """Identisch zu build_splits_pixels.py::get_aoi_utm_reference() - eigene Kopie statt
    Import, um build_splits_pixels.py (Pixel-Pipeline) nicht an dieses SDM-Skript zu koppeln."""
    aoi = gpd.read_file(AOI_PATH)
    utm_crs = aoi.estimate_utm_crs()
    aoi_utm = aoi.to_crs(utm_crs)
    minx, miny, _, _ = aoi_utm.total_bounds
    return aoi, utm_crs, minx, miny


def load_lsoa_geometries_for_london(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Laedt die nationale EW-BFC-LSOA-Datei und filtert per raeumlichem Intersects gegen die
    AOI (OSM-Verwaltungsgrenze 'Greater London', admin_level=5, siehe data/aoi/SOURCES.txt) -
    NICHT per Namensmuster. Filter im nativen CRS der Boundaries (EPSG:27700), Reprojektion
    nach EPSG:32630 danach (Reihenfolge laut INSTRUCTIONS_sdm.md Phase 3.1: erst clippen,
    dann reprojizieren)."""
    lsoa = gpd.read_file(LSOA_GPKG)
    print(f"LSOA-Datei (national): {len(lsoa)} Features, CRS={lsoa.crs}")

    aoi_native = aoi.to_crs(lsoa.crs)
    aoi_geom = aoi_native.union_all()
    within_london = lsoa[lsoa.intersects(aoi_geom)].copy()
    print(f"Nach Intersects-Filter gegen AOI 'Greater London': {len(within_london)} LSOAs")
    return within_london


def build_cell_polygons(splits_df: pd.DataFrame, minx: float, miny: float,
                         cell_size_m: float, crs) -> gpd.GeoDataFrame:
    """Zellpolygon je (cell_row, cell_col) aus splits_london.csv - Umkehrung der
    Ganzzahl-Division aus build_splits.py: col->x (Easting), row->y (Northing)."""
    x0 = minx + splits_df["cell_col"].to_numpy() * cell_size_m
    y0 = miny + splits_df["cell_row"].to_numpy() * cell_size_m
    geoms = [box(x, y, x + cell_size_m, y + cell_size_m) for x, y in zip(x0, y0)]
    return gpd.GeoDataFrame({"split": splits_df["split"].to_numpy()}, geometry=geoms, crs=crs)


def main():
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    print(f"AOI-UTM-CRS: {utm_crs}, minx={minx:.1f}, miny={miny:.1f}, Zellgroesse={CELL_SIZE_M:.0f}m")

    # --- Geometrie: clippen, dann reprojizieren ---
    lsoa_london = load_lsoa_geometries_for_london(aoi)
    lsoa_london = lsoa_london.to_crs(utm_crs)
    print(f"Nach Reprojektion auf {utm_crs}: {len(lsoa_london)} LSOAs")

    # --- Join TS062 <-> Geometrie ueber LSOA21CD/mnemonic, NICHT ueber den Namen ---
    ts062 = load_ts062()
    ts062_codes = set(ts062[CODE_COL])
    geom_codes = set(lsoa_london["LSOA21CD"])

    only_in_ts062 = ts062_codes - geom_codes
    only_in_geom = geom_codes - ts062_codes
    print(f"\nJoin TS062 <-> Geometrie (Key: LSOA21CD/mnemonic):")
    print(f"  TS062-Codes: {len(ts062_codes)}, Geometrie-Codes (nach AOI-Filter): {len(geom_codes)}")
    print(f"  Nur in TS062, keine Geometrie gefunden: {len(only_in_ts062)}"
          + (f" -> {sorted(only_in_ts062)}" if 0 < len(only_in_ts062) <= 20 else ""))
    print(f"  Nur in Geometrie (AOI-Filter), nicht in TS062: {len(only_in_geom)}"
          + (f" -> {sorted(only_in_geom)}" if 0 < len(only_in_geom) <= 20 else ""))

    matched = lsoa_london[lsoa_london["LSOA21CD"].isin(ts062_codes)].copy()
    matched = matched.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]]
    matched["lsoa_area_m2"] = matched.geometry.area
    print(f"  Gematcht (fuer die Split-Zuordnung verwendet): {len(matched)}")

    # --- Zellpolygone aus splits_london.csv rekonstruieren ---
    splits_df = pd.read_csv(SPLITS_LONDON)
    cells = build_cell_polygons(splits_df, minx, miny, CELL_SIZE_M, utm_crs)
    print(f"\nZellpolygone rekonstruiert: {len(cells)} Zellen aus {SPLITS_LONDON}")
    print(f"  Split-Verteilung der Zellen: {cells['split'].value_counts().to_dict()}")

    # --- Ueberlagerung: LSOA-Polygon x Zellpolygon ---
    overlay = gpd.overlay(matched[["lsoa_code", "geometry"]], cells, how="intersection")
    overlay["piece_area_m2"] = overlay.geometry.area
    area_by_split = (overlay.groupby(["lsoa_code", "split"])["piece_area_m2"]
                      .sum().unstack(fill_value=0.0))
    for s in ["train", "val", "test"]:
        if s not in area_by_split.columns:
            area_by_split[s] = 0.0
    area_by_split = area_by_split[["train", "val", "test"]]

    result = matched[["lsoa_code", "lsoa_area_m2"]].merge(
        area_by_split, left_on="lsoa_code", right_index=True, how="left"
    ).fillna({"train": 0.0, "val": 0.0, "test": 0.0})

    for s in ["train", "val", "test"]:
        result[f"frac_{s}"] = result[s] / result["lsoa_area_m2"]
    frac_cols = ["frac_train", "frac_val", "frac_test"]
    result["frac_covered"] = result[frac_cols].sum(axis=1)
    result["max_frac"] = result[frac_cols].max(axis=1)
    result["split"] = result[frac_cols].idxmax(axis=1).str.replace("frac_", "", regex=False)
    result["n_partitions_touched"] = (result[frac_cols] > 0).sum(axis=1)

    # --- Berichte (Phase 3, Punkt 5) ---
    print(f"\n=== LSOAs je zugewiesener Partition (Mehrheitsregel) ===")
    print(result["split"].value_counts().to_string())

    n_multi = int((result["n_partitions_touched"] > 1).sum())
    print(f"\nLSOAs mit Flaeche in mehr als einer Partition: {n_multi} von {len(result)} "
          f"({n_multi / len(result):.1%})")

    print(f"\nVerteilung des maximalen Flaechenanteils (max_frac):")
    print(result["max_frac"].describe().to_string())

    n_below_60 = int((result["max_frac"] < 0.60).sum())
    print(f"\nLSOAs mit maximalem Flaechenanteil < 60%: {n_below_60} von {len(result)} "
          f"({n_below_60 / len(result):.1%})")

    n_incomplete_cover = int((result["frac_covered"] < 0.999).sum())
    print(f"\nLSOAs mit frac_covered < 99.9% (Restflaeche in Zellen ohne Split-Eintrag, "
          f"z.B. randlich ausserhalb des Pixel-Rasters): {n_incomplete_cover}")
    if n_incomplete_cover > 0:
        print(result.loc[result["frac_covered"] < 0.999, ["lsoa_code", "frac_covered"]]
              .sort_values("frac_covered").head(10).to_string(index=False))

    # --- Variante A: Mehrheitsregel, alle gematchten LSOAs ---
    out_cols = ["lsoa_code", "split", "max_frac", "frac_train", "frac_val", "frac_test",
                "frac_covered", "n_partitions_touched"]
    OUT_MAJORITY.parent.mkdir(parents=True, exist_ok=True)
    result[out_cols].to_csv(OUT_MAJORITY, index=False)
    print(f"\nVariante A (Mehrheitsregel, alle LSOAs) gespeichert: {OUT_MAJORITY} "
          f"({len(result)} Zeilen)")

    # --- Variante B: mehrdeutige LSOAs unterhalb MIN_MAX_FRAC verworfen ---
    variant_b = result[result["max_frac"] >= MIN_MAX_FRAC][out_cols]
    variant_b.to_csv(OUT_MIN60, index=False)
    n_dropped = len(result) - len(variant_b)
    print(f"Variante B (max_frac >= {MIN_MAX_FRAC:.0%} verworfen darunter) gespeichert: "
          f"{OUT_MIN60} ({len(variant_b)} Zeilen, {n_dropped} verworfen)")
    print(f"  Split-Verteilung Variante B: {variant_b['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
