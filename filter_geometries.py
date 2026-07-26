import geopandas as gpd

def filter_rasterizable_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Behält nur Point, Polygon, MultiPolygon.
    Entfernt LineString/MultiLineString, die weder per exactextract (Coverage)
    noch per rasterio-Punktlookup verarbeitbar sind.
    """
    allowed_types = {"Point", "Polygon", "MultiPolygon"}
    geom_types = gdf.geometry.geom_type
    mask = geom_types.isin(allowed_types)

    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"Entferne {n_removed} nicht-rasterisierbare Geometrien:")
        print(geom_types[~mask].value_counts().to_string())

    return gdf[mask].copy()


if __name__ == "__main__":
    input_path = "data/interim/pois_london_test.parquet"
    output_path = "data/interim/pois_london_test_filtered.parquet"

    gdf = gpd.read_parquet(input_path)
    print(f"POIs vor Filterung: {len(gdf)}")

    gdf_filtered = filter_rasterizable_geometries(gdf)
    print(f"POIs nach Filterung: {len(gdf_filtered)}")  # Erwartung: 10.533

    gdf_filtered.to_parquet(output_path)
    print(f"Gespeichert: {output_path}")