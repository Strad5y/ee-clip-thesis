import argparse
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))   # damit config gefunden wird

import geopandas as gpd
import pandas as pd
from pyrosm import OSM
import config as cfg
from class_table import load_class_table

PBF_BY_REGION = {"london": cfg.PBF_LONDON, "niedersachsen": cfg.PBF_NIEDERSACHSEN}

# Objekteigenes wikidata-Tag: Spalte heisst object_qid, NICHT wikidata - "wikidata" ist
# bereits die Klassen-QID aus class_table.py belegt, eine Kollision waere ein stiller
# Datenfehler (zwei semantisch verschiedene Dinge unter demselben Namen).
QID_FULLMATCH = re.compile(r"^Q[0-9]+$")


def _normalize_object_qid(raw: pd.Series) -> tuple[pd.Series, dict]:
    """Trimmt, schreibt das Q-Praefix gross, verwirft Mehrfachwerte (z.B. 'Q123;Q456' -
    NICHT aufspalten) und setzt Werte, die nicht exakt Q[0-9]+ entsprechen, auf null.
    Gibt (normalisierte Serie, {n_multi_dropped, n_invalid_dropped}) zurueck."""
    trimmed = raw.astype("string").str.strip()
    trimmed = trimmed.where(trimmed.ne(""), pd.NA)

    is_multi = trimmed.str.contains(";", na=False)
    n_multi = int(is_multi.sum())
    single = trimmed.mask(is_multi, pd.NA)

    upper = single.str.upper()
    is_valid = upper.str.fullmatch(QID_FULLMATCH.pattern).fillna(False)
    n_invalid = int((upper.notna() & ~is_valid).sum())
    cleaned = upper.where(is_valid, pd.NA)

    return cleaned, {"n_multi_dropped": n_multi, "n_invalid_dropped": n_invalid}


def active_keys_from_class_table(class_table_path=cfg.CLASS_TABLE) -> list[str]:
    """Aktiver OSM-Key-Satz = distinkte Werte der key-Spalte der Klassentabelle (Nutzervorgabe:
    NICHT in der Config pflegen, sonst koennen Config und Tabelle auseinanderlaufen)."""
    cls = load_class_table(class_table_path)
    return sorted(cls["key"].unique().tolist())


def extract(bbox_or_geom, pbf, active_keys):
    # pyrosm.OSM(bounding_box=...) akzeptiert entweder [minx,miny,maxx,maxy] ODER direkt ein
    # Shapely Polygon/MultiPolygon - list(...) wuerde ein Polygon falsch/gar nicht iterieren
    bounding_box = list(bbox_or_geom) if isinstance(bbox_or_geom, (list, tuple)) else bbox_or_geom
    osm = OSM(str(pbf), bounding_box=bounding_box)
    frames = []
    qid_counts = {"n_multi_dropped": 0, "n_invalid_dropped": 0}

    for key in active_keys:
        # extra_attributes=["wikidata"] ist der verifizierte Mechanismus (siehe Phase-1-Bericht):
        # ergaenzt NUR die zusaetzliche Spalte, ohne pyrosms Default-Spaltenset (u.a. "name") zu
        # ersetzen - tags_as_columns wuerde genau das tun und "name" stillschweigend verlieren.
        gdf = osm.get_data_by_custom_criteria(
            custom_filter={key: True},
            filter_type="keep",
            keep_nodes=True,
            keep_ways=True,
            keep_relations=True,
            extra_attributes=["wikidata"],
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

        raw_wikidata = gdf["wikidata"] if "wikidata" in gdf.columns else pd.Series(pd.NA, index=gdf.index)
        object_qid, stats = _normalize_object_qid(raw_wikidata)
        gdf["object_qid"] = object_qid
        qid_counts["n_multi_dropped"] += stats["n_multi_dropped"]
        qid_counts["n_invalid_dropped"] += stats["n_invalid_dropped"]

        frames.append(gdf[["poi_id", "key", "value", "name", "object_qid", "geometry"]])
        print(f"  {key}: {len(gdf)} POIs, {int(object_qid.notna().sum())} mit object_qid")

    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    print(f"  object_qid: {qid_counts['n_multi_dropped']} Mehrfachwerte verworfen (nicht "
          f"aufgespalten), {qid_counts['n_invalid_dropped']} ungueltige Werte (kein Q[0-9]+) "
          f"auf null gesetzt")

    # disused / abandoned raus
    out = out[~out["value"].str.startswith(("disused", "abandoned"))]
    out = out.drop_duplicates(subset=["poi_id", "key"])
    # Linien raus - nur Punkte und Flächen sind rasterfähig
    ok = ["Point", "Polygon", "MultiPolygon"]
    out = out[out.geometry.geom_type.isin(ok)]
    return out


def main():
    parser = argparse.ArgumentParser(description="POI-Extraktion je Region")
    parser.add_argument("--region", type=str, default="london", choices=sorted(PBF_BY_REGION),
                         help="AOI aus data/aoi/{region}.gpkg, PBF aus PBF_BY_REGION.")
    args = parser.parse_args()

    active_keys = active_keys_from_class_table()
    print(f"Aktive Keys aus Klassentabelle ({cfg.CLASS_TABLE}): {active_keys}")

    aoi_path = cfg.AOI_DIR / f"{args.region}.gpkg"
    pbf = PBF_BY_REGION[args.region]
    aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    aoi_geom = aoi.geometry.iloc[0]  # pyrosm.OSM(bounding_box=...) akzeptiert Polygon/MultiPolygon direkt
    print(f"Region: {args.region}")
    print(f"Lese OSM-Daten gegen AOI-Polygon (nicht BBOX_TEST): {aoi_path}")
    print(f"PBF: {pbf}")
    pois = extract(aoi_geom, pbf, active_keys)

    pois["gtype"] = pois.geometry.geom_type
    print()
    print(pois.groupby(["key", "gtype"]).size())
    print()
    print("Gesamt Zeilen:", len(pois))
    print("Eindeutige poi_id:", pois["poi_id"].nunique())

    out = cfg.DATA / "interim" / f"pois_{args.region}_full.parquet"
    pois.drop(columns=["gtype"]).to_parquet(out)
    print("Gespeichert:", out)


if __name__ == "__main__":
    main()
