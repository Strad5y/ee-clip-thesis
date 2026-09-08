import geopandas as gpd

ALLOWED_TYPES = {"Point", "Polygon", "MultiPolygon"}


def filter_rasterizable_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Behält nur Point, Polygon, MultiPolygon.

    Entfernt LineString/MultiLineString (sowie MultiPoint/GeometryCollection),
    die weder per exactextract (Coverage) noch per rasterio-Punktlookup
    verarbeitbar sind, und filtert zusätzlich leere/ungültige Geometrien.

    MultiPolygon-Relationen bleiben INTAKT (kein explode): ein Pixel, das zwei
    Teile derselben Relation schneidet, würde sonst doppelt gezählt.
    """
    n_in = len(gdf)
    geom_types = gdf.geometry.geom_type

    # 1) Typfilter: nur punkt-/flächenhafte Features
    type_mask = geom_types.isin(ALLOWED_TYPES)
    n_removed_type = int((~type_mask).sum())
    if n_removed_type > 0:
        print(f"Entferne {n_removed_type} nicht-rasterisierbare Geometrien "
              f"({100 * n_removed_type / max(n_in, 1):.2f} %):")
        for gtype, n in geom_types[~type_mask].value_counts().items():
            print(f"  {gtype:<18} {n:>7}  ({100 * n / max(n_in, 1):.2f} %)")

    gdf = gdf[type_mask].copy()
    n_after_type = len(gdf)  # <-- entspricht deiner Erwartung (10.533)

    # 2) Validitätsfilter: leere/None/ungültige Geometrien absichern.
    #    (~is_empty & is_valid) verwirft leere UND fehlende Geometrien warnungsfrei
    #    – exactextract/rasterio stolpern sonst darüber.
    valid_mask = ~gdf.geometry.is_empty & gdf.geometry.is_valid
    n_removed_valid = int((~valid_mask).sum())
    if n_removed_valid > 0:
        print(f"Entferne {n_removed_valid} leere/ungültige Geometrien.")
    gdf = gdf[valid_mask].reset_index(drop=True)

    print(f"Behalten: {len(gdf)} von {n_in}  "
          f"(nach Typfilter: {n_after_type}, nach Validität: {len(gdf)})")
    return gdf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None,
                         help="Ohne Angabe: altes London-Testsubset (pois_london_test.parquet, "
                              "unveraendertes Default-Verhalten). Mit Angabe: volle Region "
                              "(pois_{region}_full.parquet -> pois_{region}_full_filtered.parquet).")
    args = parser.parse_args()

    if args.region is None:
        input_path = "data/interim/pois_london_test.parquet"
        output_path = "data/interim/pois_london_test_filtered.parquet"
    else:
        input_path = f"data/interim/pois_{args.region}_full.parquet"
        output_path = f"data/interim/pois_{args.region}_full_filtered.parquet"

    gdf = gpd.read_parquet(input_path)
    print(f"POIs vor Filterung: {len(gdf)}")

    gdf_filtered = filter_rasterizable_geometries(gdf)
    print(f"POIs nach Filterung: {len(gdf_filtered)}")  # Erwartung: 10.533 (nach Typfilter)

    gdf_filtered.to_parquet(output_path)
    print(f"Gespeichert: {output_path}")