import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))   # damit config gefunden wird

import geopandas as gpd
import pandas as pd
from pyrosm import OSM
import config as cfg


def extract(bbox_or_geom, pbf):
    # pyrosm.OSM(bounding_box=...) akzeptiert entweder [minx,miny,maxx,maxy] ODER direkt ein
    # Shapely Polygon/MultiPolygon - list(...) wuerde ein Polygon falsch/gar nicht iterieren
    bounding_box = list(bbox_or_geom) if isinstance(bbox_or_geom, (list, tuple)) else bbox_or_geom
    osm = OSM(str(pbf), bounding_box=bounding_box)
    frames = []

    for key in cfg.OSM_KEYS:
        gdf = osm.get_data_by_custom_criteria(
            custom_filter={key: True},
            filter_type="keep",
            keep_nodes=True,
            keep_ways=True,
            keep_relations=True,
        )
        if gdf is None or len(gdf) == 0:
            print(f"  {key}: nichts gefunden")
            continue

        gdf = gdf[gdf[key].notna()].copy()
        gdf["key"] = key
        gdf["value"] = gdf[key].astype(str)
        gdf["poi_id"] = gdf["osm_type"].astype(str) + "/" + gdf["id"].astype(str)
        # "name" ist pro poi_id konstant (id_var), keine eigene key/value-Zeile -
        # manche Keys (z.B. landuse) liefern gar keine "name"-Spalte von pyrosm
        gdf["name"] = gdf["name"] if "name" in gdf.columns else pd.NA

        frames.append(gdf[["poi_id", "key", "value", "name", "geometry"]])
        print(f"  {key}: {len(gdf)} POIs")

    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")

    # disused / abandoned raus
    out = out[~out["value"].str.startswith(("disused", "abandoned"))]
    out = out.drop_duplicates(subset=["poi_id", "key"])
    # Linien raus - nur Punkte und Flächen sind rasterfähig
    ok = ["Point", "Polygon", "MultiPolygon"]
    out = out[out.geometry.geom_type.isin(ok)]
    return out


if __name__ == "__main__":
    aoi = gpd.read_file(cfg.AOI_DIR / "london.gpkg").to_crs("EPSG:4326")
    aoi_geom = aoi.geometry.iloc[0]  # pyrosm.OSM(bounding_box=...) akzeptiert Polygon/MultiPolygon direkt
    print(f"Lese OSM-Daten gegen AOI-Polygon (nicht BBOX_TEST): {cfg.AOI_DIR / 'london.gpkg'}")
    pois = extract(aoi_geom, cfg.PBF_LONDON)

    pois["gtype"] = pois.geometry.geom_type
    print()
    print(pois.groupby(["key", "gtype"]).size())
    print()
    print("Gesamt Zeilen:", len(pois))
    print("Eindeutige poi_id:", pois["poi_id"].nunique())

    out = cfg.DATA / "interim" / "pois_london_test.parquet"
    pois.drop(columns=["gtype"]).to_parquet(out)
    print("Gespeichert:", out)
