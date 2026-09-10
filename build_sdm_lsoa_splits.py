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

Nachtrag (Diagnose Grenzfaelle + partitionsreine Poolung): zusaetzlich zur Mehrheitsregel wird
der LSOA-Zelle-Overlay selbst persistiert (data/splits/sdm_lsoa_cell_overlay_london.csv) -
genau die Information, die Phase 5 braucht, um beim Pooling NUR Pixel aus Zellen der
zugewiesenen Partition zu verwenden (partitionsreine Poolung, schliesst den Leakage-Kanal
"Regionsvektor mittelt ueber mehrere Partitionen, obwohl die Zuordnung nur Mehrheits-basiert
ist"). Pixelzahl-Schaetzung ueber Flaeche/Pixelflaeche (10 m x 10 m, beide Backbones laut
CLAUDE.md) statt echtem Pixel-Durchlauf - genau die im Auftrag verlangte Abkuerzung ("ohne das
Raster erneut zu durchlaufen"), da die Pixel auf einem regelmaessigen 10-m-Gitter liegen und
Flaechenanteil und Pixelanteil bei dieser Aufloesung praktisch identisch sind.
"""
import re
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
PIXEL_SIZE_M = 10.0   # TESSERA und AlphaEarth sind beide 10 m (CLAUDE.md, Abschnitt 0)
PIXEL_AREA_M2 = PIXEL_SIZE_M ** 2

OUT_MAJORITY = cfg.DATA / "splits" / "sdm_lsoa_splits_london.csv"
MIN_MAX_FRAC = 0.60  # Schwellwert Variante B - siehe Begruendung im Bericht
OUT_MIN60 = cfg.DATA / "splits" / "sdm_lsoa_splits_london_min60.csv"
OUT_OVERLAY = cfg.DATA / "splits" / "sdm_lsoa_cell_overlay_london.csv"


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
    Ganzzahl-Division aus build_splits.py: col->x (Easting), row->y (Northing). cell_row/
    cell_col bleiben als Spalten erhalten (nicht nur 'split') - werden fuer den persistierten
    LSOA-Zelle-Overlay gebraucht (Phase-5-Filterregel)."""
    x0 = minx + splits_df["cell_col"].to_numpy() * cell_size_m
    y0 = miny + splits_df["cell_row"].to_numpy() * cell_size_m
    geoms = [box(x, y, x + cell_size_m, y + cell_size_m) for x, y in zip(x0, y0)]
    return gpd.GeoDataFrame(
        {"cell_row": splits_df["cell_row"].to_numpy(),
         "cell_col": splits_df["cell_col"].to_numpy(),
         "split": splits_df["split"].to_numpy()},
        geometry=geoms, crs=crs,
    )


def borough_from_name(name: str) -> str:
    """LSOA21NM-Format ist '<Borough-Name> <Ward-Code>' (z.B. 'City of London 001A') - der
    Ward-Code ist immer das letzte, whitespace-getrennte Token."""
    return re.sub(r"\s+\S+$", "", name)


def compute_lsoa_cell_overlay(matched: gpd.GeoDataFrame, cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    overlay = gpd.overlay(matched[["lsoa_code", "geometry"]], cells, how="intersection")
    overlay["piece_area_m2"] = overlay.geometry.area
    return overlay


def compute_split_assignment(matched: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """Flaechenanteil je Partition, Mehrheitszuordnung, plus Diagnosespalten (margin_frac fuer
    die Grenzfall-Analyse, Pixelzahl-Schaetzungen fuer die partitionsreine Poolung)."""
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

    sorted_fracs = np.sort(result[frac_cols].to_numpy(), axis=1)  # aufsteigend
    result["max_frac"] = sorted_fracs[:, -1]
    result["second_max_frac"] = sorted_fracs[:, -2]
    result["margin_frac"] = result["max_frac"] - result["second_max_frac"]
    result["split"] = result[frac_cols].idxmax(axis=1).str.replace("frac_", "", regex=False)
    result["n_partitions_touched"] = (result[frac_cols] > 0).sum(axis=1)

    # Partitionsreine Poolung: nur die Flaeche/Pixel der ZUGEWIESENEN Partition zaehlen
    # (= max_frac * lsoa_area_m2, per Konstruktion identisch zur Flaeche der Mehrheits-
    # Partition). Pixelzahl ueber Flaeche/Pixelflaeche geschaetzt, kein Raster-Durchlauf.
    result["area_kept_m2"] = result["max_frac"] * result["lsoa_area_m2"]
    result["n_pixels_total_est"] = np.round(result["lsoa_area_m2"] / PIXEL_AREA_M2).astype(int)
    result["n_pixels_kept_est"] = np.round(result["area_kept_m2"] / PIXEL_AREA_M2).astype(int)
    result["frac_pixels_kept"] = result["max_frac"]  # identisch per Konstruktion, eigener Name fuer Klarheit in Phase 5
    return result


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
    matched["borough"] = matched["LSOA21NM"].map(borough_from_name)
    matched = matched.rename(columns={"LSOA21CD": "lsoa_code"})[
        ["lsoa_code", "borough", "geometry"]
    ]
    matched["lsoa_area_m2"] = matched.geometry.area
    print(f"  Gematcht (fuer die Split-Zuordnung verwendet): {len(matched)}")

    # --- Zellpolygone aus splits_london.csv rekonstruieren ---
    splits_df = pd.read_csv(SPLITS_LONDON)
    cells = build_cell_polygons(splits_df, minx, miny, CELL_SIZE_M, utm_crs)
    print(f"\nZellpolygone rekonstruiert: {len(cells)} Zellen aus {SPLITS_LONDON}")
    print(f"  Split-Verteilung der Zellen: {cells['split'].value_counts().to_dict()}")

    # --- Ueberlagerung: LSOA-Polygon x Zellpolygon ---
    overlay = compute_lsoa_cell_overlay(matched, cells)
    result = compute_split_assignment(matched, overlay)

    # --- Berichte (Phase 3, Punkt 5) ---
    print(f"\n=== LSOAs je zugewiesener Partition (Mehrheitsregel) ===")
    print(result["split"].value_counts().to_string())

    n_multi = int((result["n_partitions_touched"] > 1).sum())
    print(f"\nLSOAs mit Flaeche in mehr als einer Partition: {n_multi} von {len(result)} "
          f"({n_multi / len(result):.1%})")

    print(f"\nVerteilung des maximalen Flaechenanteils (max_frac):")
    print(result["max_frac"].describe().to_string())

    below_60 = result["max_frac"] < 0.60
    n_below_60 = int(below_60.sum())
    print(f"\nLSOAs mit maximalem Flaechenanteil < 60%: {n_below_60} von {len(result)} "
          f"({n_below_60 / len(result):.1%})")

    n_incomplete_cover = int((result["frac_covered"] < 0.999).sum())
    print(f"\nLSOAs mit frac_covered < 99.9% (Restflaeche in Zellen ohne Split-Eintrag, "
          f"z.B. randlich ausserhalb des Pixel-Rasters): {n_incomplete_cover}")
    if n_incomplete_cover > 0:
        print(result.loc[result["frac_covered"] < 0.999, ["lsoa_code", "frac_covered"]]
              .sort_values("frac_covered").head(10).to_string(index=False))

    # --- Diagnose Grenzfaelle (max_frac < 60%) ---
    borderline = result[below_60].merge(matched[["lsoa_code", "borough", "geometry"]], on="lsoa_code")
    borderline = gpd.GeoDataFrame(borderline, geometry="geometry", crs=utm_crs)
    n_coinflip = int((borderline["margin_frac"] < 0.10).sum())
    print(f"\n=== Diagnose: {len(borderline)} Grenzfaelle (max_frac < 60%) ===")
    print(f"Davon mit margin_frac < 0.10 (zweitgroesster Anteil < 0.10 unter dem groessten, "
          f"'Muenzwurf'): {n_coinflip} von {len(borderline)} ({n_coinflip / len(borderline):.1%})")
    print(f"margin_frac-Verteilung der Grenzfaelle:")
    print(borderline["margin_frac"].describe().to_string())

    print(f"\nBoroughs der Grenzfaelle (Anzahl LSOAs, absteigend, Top 10 von "
          f"{borderline['borough'].nunique()} betroffenen Boroughs):")
    print(borderline["borough"].value_counts().head(10).to_string())
    print(f"Boroughs insgesamt in London: {matched['borough'].nunique()}")

    centroids = borderline.geometry.centroid
    aoi_utm = aoi.to_crs(utm_crs)
    aoi_boundary = aoi_utm.geometry.iloc[0].boundary
    dist_to_aoi_edge = centroids.distance(aoi_boundary)
    print(f"\nAbstand der Grenzfall-Zentroide zum AOI-Rand (Meter):")
    print(dist_to_aoi_edge.describe().to_string())
    all_centroids = matched.geometry.centroid
    dist_all = all_centroids.distance(aoi_boundary)
    print(f"Zum Vergleich, Abstand ALLER LSOA-Zentroide zum AOI-Rand (Meter):")
    print(dist_all.describe().to_string())

    # --- Variante A: Mehrheitsregel, alle gematchten LSOAs ---
    out_cols = ["lsoa_code", "split", "max_frac", "second_max_frac", "margin_frac",
                "frac_train", "frac_val", "frac_test", "frac_covered", "n_partitions_touched",
                "lsoa_area_m2", "area_kept_m2", "n_pixels_total_est", "n_pixels_kept_est",
                "frac_pixels_kept"]
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

    # --- Partitionsreine Poolung: Pixelretentions-Statistik + persistierter Cell-Overlay ---
    print(f"\n=== Partitionsreine Poolung: verbleibende Pixel je LSOA nach Filter "
          f"(Zellen != zugewiesener Split ausgeschlossen) ===")
    print(f"frac_pixels_kept (== max_frac): Mittel={result['frac_pixels_kept'].mean():.4f} "
          f"Median={result['frac_pixels_kept'].median():.4f} "
          f"Min={result['frac_pixels_kept'].min():.4f} "
          f"5%-Quantil={result['frac_pixels_kept'].quantile(0.05):.4f}")
    n_under_100 = int((result["n_pixels_kept_est"] < 100).sum())
    print(f"LSOAs mit < 100 verbleibenden Pixeln (geschaetzt, {PIXEL_SIZE_M:.0f}m-Raster): "
          f"{n_under_100} von {len(result)} ({n_under_100 / len(result):.1%})")
    n_empty = int((result["n_pixels_kept_est"] == 0).sum())
    print(f"LSOAs komplett leer nach Filter (n_pixels_kept_est == 0): {n_empty}")
    print(f"n_pixels_total_est-Verteilung (zum Vergleich, vor dem Filter):")
    print(result["n_pixels_total_est"].describe().to_string())

    overlay_out_cols = ["lsoa_code", "cell_row", "cell_col", "split", "piece_area_m2"]
    overlay[overlay_out_cols].to_csv(OUT_OVERLAY, index=False)
    print(f"\nLSOA-Zelle-Overlay gespeichert: {OUT_OVERLAY} ({len(overlay)} Zeilen) - "
          f"Phase-5-Filterregel: Pixel behalten, wenn dessen Zelle (cell_row,cell_col) in "
          f"dieser Datei mit split == LSOA-split (aus {OUT_MAJORITY.name}) vorkommt.")


if __name__ == "__main__":
    main()
