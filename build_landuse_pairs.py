"""
build_landuse_pairs.py

Urban-Atlas-Ground-Truth fuer die Land-Use-Evaluation (Phase 1+2, Nutzervorgabe):
1. Urban-Atlas-London-FGB einlesen, auf data/aoi/london.gpkg zuschneiden (AOI wird ins
   Daten-CRS (EPSG:3035) reprojiziert, nicht umgekehrt - Copernicus liefert deutlich weniger
   Datenvolumen als eine Reprojektion der 133k-Polygon-Schicht waere).
2. Klassenausschluss: 33000 (Open spaces, little/no vegetation) und 22000 (Permanent crops) -
   Nutzerentscheidung nach Bestandsaufnahme, nur 4 bzw. 2 Polygone in London, zu wenig fuer
   einen raeumlich geblockten Split. Bleiben 25 Klassen.
3. Pixel-Label-Paare: dieselbe Kachel-Join-Logik wie beim POI-Join (spatial_join.py,
   join_pois_to_tiles/join_pois_to_raster) - Urban-Atlas-Polygone sind geometrisch nichts
   anderes als POI-Polygone, nur groesser. id_col='identifier' (in London eindeutig, siehe
   Bestandsaufnahme), carry=['code_2021','class_2021'] traegt die Klasse direkt durch den
   Join (kein Long-Format-Tag-Merge wie bei POIs noetig - eine Klasse pro Polygon).
4. Pro Polygon max. N=100 Pixel zufaellig ziehen (Nutzervorgabe) - NACH dem vollen Join,
   sonst muesste vorab bekannt sein, welche Pixel ueberhaupt in Frage kommen.
5. Split-Zuordnung PRO PIXEL (nicht pro Polygon, anders als bei POIs!): jedes Pixel bekommt
   ueber seine Weltkoordinate dieselbe Gitterzelle (cell_row, cell_col) wie build_splits.py
   (identische AOI-UTM-CRS, identischer Ursprung, identische Zellgroesse 1000 m), dann Split
   ueber die (cell_row, cell_col) -> split-Zuordnung aus data/splits/splits_london.csv
   nachschlagen. Nutzervorgabe: "Ein Urban-Atlas-Pixel gehoert zu dem Split, in dessen
   Gitterzelle es liegt" - EIN grosses Polygon kann mehrere Gitterzellen und damit mehrere
   Splits ueberspannen, das ist hier explizit erwuenscht (anders als bei den kleinen POIs,
   wo ein Polygon praktisch nie eine Zellgrenze ueberspannt).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

import config as cfg
import spatial_join as sj
from build_splits import CELL_SIZE_M

UA_FGB = (cfg.DATA / "worldatlas" / "extracted" / "london" /
          "CLMS_UA_LCU_S2021_V025ha_UK001L3_LONDON_03035_V01_R01_20250730" /
          "CLMS_UA_LCU_S2021_V025ha_UK001L3_LONDON_03035_V01_R01_20250730.fgb")
AOI_IN = cfg.AOI_DIR / "london.gpkg"
SPLITS_IN = cfg.DATA / "splits" / "splits_london.csv"

UA_OUT = cfg.DATA / "interim" / "urban_atlas_london.parquet"
OUT_DIR = cfg.DATA / "processed"

# Nutzerentscheidung (nach Bestandsaufnahme): zu wenige Polygone in London fuer einen
# raeumlich geblockten Split (4 bzw. 2 Stueck) - ausgeschlossen, nicht stillschweigend.
EXCLUDED_CLASSES = {"33000", "22000"}
EXCLUDED_CLASS_NAMES = {
    "33000": "Open spaces with little or no vegetation (beach, dune, bare rock, glacier)",
    "22000": "Permanent crops (vineyards, fruit trees, olive groves)",
}

ID_COL = "identifier"
CARRY = ["code_2021", "class_2021"]
MAX_PIXELS_PER_POLYGON = 100
SAMPLE_SEED = 0


def load_and_clip(ua_path=UA_FGB, aoi_path=AOI_IN) -> gpd.GeoDataFrame:
    ua = gpd.read_file(ua_path)
    n_before = len(ua)

    aoi = gpd.read_file(aoi_path).to_crs(ua.crs)  # AOI ins Daten-CRS, nicht umgekehrt
    clipped = gpd.clip(ua, aoi)

    n_after_clip = len(clipped)
    clipped = clipped[~clipped["code_2021"].isin(EXCLUDED_CLASSES)].reset_index(drop=True)
    n_after_excl = len(clipped)

    clipped["area_m2"] = clipped.geometry.area
    total_area = clipped["area_m2"].sum()

    print(f"Polygone vor Zuschnitt: {n_before}")
    print(f"Polygone nach Zuschnitt auf AOI: {n_after_clip}")
    print(f"Polygone nach Klassenausschluss ({sorted(EXCLUDED_CLASSES)}): {n_after_excl} "
          f"({n_after_clip - n_after_excl} entfernt)")
    print(f"Gesamtflaeche: {total_area / 1e6:.2f} km2")
    print(f"Anzahl Klassen: {clipped['code_2021'].nunique()}")

    dist = (clipped.groupby(["code_2021", "class_2021"])
            .agg(n_polygons=("area_m2", "size"), area_m2=("area_m2", "sum")))
    dist["area_share"] = dist["area_m2"] / total_area
    dist = dist.sort_values("n_polygons", ascending=False)
    print("\nKlassenverteilung (Polygone, Flaechenanteil):")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(dist)

    return clipped


def assign_pixel_splits(pairs: pd.DataFrame, tile_paths: dict, aoi_path=AOI_IN,
                         splits_path=SPLITS_IN, cell_size_m=CELL_SIZE_M) -> pd.Series:
    """Weltkoordinate je Pixel aus (tile, row, col) rekonstruieren (rasterio.transform.xy),
    dann dieselbe Gitterzellen-Formel wie build_splits.py::assign_cells() anwenden (identische
    AOI-UTM-CRS, identischer Ursprung aoi_utm.total_bounds, identische Zellgroesse) und gegen
    die (cell_row, cell_col) -> split-Zuordnung aus splits_london.csv nachschlagen. Zellen ohne
    POI (und damit ohne Split-Zuordnung, siehe splits.py) werden NICHT stillschweigend
    verworfen, sondern als eigene Kategorie 'missing' zurueckgegeben und muessen vom Aufrufer
    explizit behandelt/berichtet werden."""
    aoi = gpd.read_file(aoi_path)
    utm_crs = aoi.estimate_utm_crs()
    aoi_utm = aoi.to_crs(utm_crs)
    minx, miny, _, _ = aoi_utm.total_bounds

    # Weltkoordinate (in der jeweils nativen Tile-CRS) je Zeile rekonstruieren
    xs = np.empty(len(pairs))
    ys = np.empty(len(pairs))
    crs_per_tile = {}
    for tile_name, sub in pairs.groupby("tile"):
        tile_path = tile_paths[tile_name]
        with rasterio.open(tile_path) as src:
            transform = src.transform
            crs_per_tile[tile_name] = src.crs
        x, y = rasterio.transform.xy(transform, sub["row"].to_numpy(), sub["col"].to_numpy())
        xs[sub.index.to_numpy()] = x
        ys[sub.index.to_numpy()] = y

    # Punkte je nativer Tile-CRS gruppiert reprojizieren (kann sich zwischen Kacheln
    # unterscheiden, siehe utm_zone-Spalte aus dem Join)
    cell_row = np.full(len(pairs), -1, dtype=np.int64)
    cell_col = np.full(len(pairs), -1, dtype=np.int64)
    for tile_name, sub in pairs.groupby("tile"):
        idx = sub.index.to_numpy()
        pts = gpd.GeoSeries(gpd.points_from_xy(xs[idx], ys[idx]), crs=crs_per_tile[tile_name])
        pts_utm = pts.to_crs(utm_crs)
        cell_col[idx] = np.floor((pts_utm.x.to_numpy() - minx) / cell_size_m).astype(np.int64)
        cell_row[idx] = np.floor((pts_utm.y.to_numpy() - miny) / cell_size_m).astype(np.int64)

    splits_df = pd.read_csv(splits_path)
    cell_to_split = (splits_df[["cell_row", "cell_col", "split"]]
                      .drop_duplicates()
                      .set_index(["cell_row", "cell_col"])["split"])
    # Konsistenz-Check: eine Zelle darf nicht zwei verschiedene Splits im Artefakt haben
    n_conflicting = cell_to_split.index.duplicated().sum()
    if n_conflicting > 0:
        raise ValueError(f"{n_conflicting} Gitterzellen haben widerspruechliche Split-Zuordnungen "
                          f"in {splits_path} - Split-Artefakt inkonsistent.")

    keys = pd.MultiIndex.from_arrays([cell_row, cell_col])
    split = cell_to_split.reindex(keys).to_numpy()
    split = pd.Series(split, index=pairs.index).fillna("missing")
    return split


def build_landuse_pairs(name: str, ua: gpd.GeoDataFrame, tile_paths_list) -> pd.DataFrame:
    print(f"\n=== Backbone: {name} ===")
    pairs = sj.join_pois_to_tiles(ua, tile_paths_list, id_col=ID_COL, carry=CARRY,
                                   dequantize=sj.DEQUANTIZE.get(name))
    if len(pairs) == 0:
        print("  Keine Paare erzeugt.")
        return pairs
    print(f"  Pixel-Zeilen vor Sampling: {len(pairs)} (aus {pairs[ID_COL].nunique()} Polygonen)")

    rng = np.random.default_rng(SAMPLE_SEED)
    sampled = (pairs.groupby(ID_COL, group_keys=False)
               .apply(lambda g: g.sample(n=min(len(g), MAX_PIXELS_PER_POLYGON), random_state=rng)))
    sampled = sampled.reset_index(drop=True)
    print(f"  Pixel-Zeilen nach Sampling (max {MAX_PIXELS_PER_POLYGON}/Polygon): {len(sampled)}")

    tile_paths = {p.name: p for p in tile_paths_list}
    split = assign_pixel_splits(sampled, tile_paths)
    sampled["split"] = split.to_numpy()

    n_missing = int((sampled["split"] == "missing").sum())
    if n_missing > 0:
        print(f"  Warnung: {n_missing} Pixel ohne Split-Zuordnung (Gitterzelle ohne POI im "
              f"Split-Artefakt) - im Output markiert als split='missing', NICHT verworfen.")

    return sampled


def main():
    print(f"Lese Urban Atlas: {UA_FGB}")
    print(f"Zuschnitt auf: {AOI_IN}\n")
    ua = load_and_clip()

    UA_OUT.parent.mkdir(parents=True, exist_ok=True)
    ua.to_parquet(UA_OUT)
    print(f"\nGespeichert: {UA_OUT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tile_paths in sj.BACKBONES.items():
        existing = [p for p in tile_paths if p.exists()]
        if not existing:
            print(f"\n=== Backbone: {name} === Keine Kacheln vorhanden - uebersprungen")
            continue

        pairs = build_landuse_pairs(name, ua, existing)
        if len(pairs) == 0:
            continue

        out_path = OUT_DIR / f"landuse_pairs_{name}.parquet"
        pairs.to_parquet(out_path)
        print(f"  Gespeichert: {out_path}")

        print(f"\n  Split-Report ({name}):")
        print(f"  {pairs.groupby('split').size()}")
        for split_name, g in pairs.groupby("split"):
            if split_name == "missing":
                continue
            n_classes = g["code_2021"].nunique()
            print(f"    {split_name}: {len(g)} Zeilen, {n_classes} Klassen")
        all_classes = set(pairs["code_2021"].unique())
        for split_name in ["train", "val", "test"]:
            present = set(pairs.loc[pairs["split"] == split_name, "code_2021"].unique())
            missing_classes = all_classes - present
            if missing_classes:
                print(f"    {split_name}: FEHLENDE Klassen: {sorted(missing_classes)}")


if __name__ == "__main__":
    main()
