"""
validate_pairs.py

Absicherung der von spatial_join.py erzeugten Paar-Parquets, bevor sie ins Training gehen.
Bricht bei einem fehlgeschlagenen Check ab (sys.exit(1)) statt stillschweigend weiterzumachen.
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from spatial_join import BACKBONES, ID_COL, OUT_DIR, POIS_IN

# Pixelzentrum darf max. so viele Pixelbreiten von der POI-Geometrie entfernt liegen -
# fängt CRS-/Transform-Fehler ab, ohne legitime Randpixel (geringe coverage) zu verwerfen
MAX_DIST_IN_PIXELS = 3


def validate(pairs_path, raster_path, pois_path=POIS_IN, id_col=ID_COL):
    errors = []

    pairs = pd.read_parquet(pairs_path)
    pois = gpd.read_parquet(pois_path)

    with rasterio.open(raster_path) as src:
        n_bands = src.count
        transform = src.transform
        raster_crs = src.crs

    # 1) Embedding-Dimension konsistent und korrekt
    dims = pairs["embedding"].apply(len)
    if dims.nunique() != 1:
        errors.append(f"Uneinheitliche Embedding-Dim: {dims.unique()}")
    elif dims.iloc[0] != n_bands:
        errors.append(f"Embedding-Dim {dims.iloc[0]} != Bandanzahl Raster {n_bands}")
    else:
        print(f"  OK Embedding-Dim = {n_bands}")

    # 2) Keine NaN- oder Null-Vektoren
    emb_stack = np.stack(pairs["embedding"].to_numpy())
    n_nan = int(np.isnan(emb_stack).any(axis=1).sum())
    n_null = int((~emb_stack.any(axis=1)).sum())
    if n_nan > 0:
        errors.append(f"{n_nan} Embeddings enthalten NaN")
    if n_null > 0:
        errors.append(f"{n_null} Embeddings sind komplett Null (vermutlich NoData-Pixel)")
    if n_nan == 0 and n_null == 0:
        print("  OK keine NaN-/Null-Vektoren")

    # 3) coverage in [0,1]
    cov = pairs["coverage"]
    if cov.min() < 0 or cov.max() > 1:
        errors.append(f"coverage außerhalb [0,1]: min={cov.min()}, max={cov.max()}")
    else:
        print(f"  OK coverage in [0,1] (min={cov.min():.6g}, max={cov.max():.6g})")

    # 4) alle poi_id referenzierbar
    unknown = set(pairs[id_col]) - set(pois[id_col])
    if unknown:
        errors.append(f"{len(unknown)} poi_id in pairs ohne Entsprechung in POIs, z.B. {list(unknown)[:5]}")
    else:
        print(f"  OK alle {pairs[id_col].nunique()} poi_id in pairs sind in POIs vorhanden")

    # 5) räumlicher Sanity-Check: Pixelzentrum nahe an der POI-Geometrie (fängt CRS-Fehler)
    # pois ist Long-Format (eine Zeile pro poi_id UND Tag) -> vor der Indizierung auf
    # eindeutige poi_id reduzieren, sonst verschiebt .loc[] mit doppeltem Index die Geometrien
    pois_unique = pois.drop_duplicates(subset=id_col)
    pois_proj = pois_unique.set_index(id_col).to_crs(raster_crs)
    px_w = abs(transform.a)
    max_dist = MAX_DIST_IN_PIXELS * px_w

    xs, ys = rasterio.transform.xy(transform, pairs["row"].to_numpy(), pairs["col"].to_numpy())
    centers = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=raster_crs)

    geoms = pois_proj.loc[pairs[id_col].to_numpy(), "geometry"].reset_index(drop=True)
    dists = centers.reset_index(drop=True).distance(geoms)

    n_far = int((dists > max_dist).sum())
    if n_far > 0:
        worst = dists.sort_values(ascending=False).head(5)
        errors.append(
            f"{n_far} Paare mit Pixelzentrum > {max_dist:.1f} m von der POI-Geometrie entfernt "
            f"(deutet auf CRS-/Transform-Fehler hin). Größte Abstände: {list(worst.round(1))}"
        )
    else:
        print(f"  OK alle Pixelzentren <= {max_dist:.1f} m ({MAX_DIST_IN_PIXELS} Pixel) von ihrer POI-Geometrie entfernt "
              f"(max beobachtet: {dists.max():.2f} m)")

    return errors


def main():
    any_errors = False
    for name, raster_path in BACKBONES.items():
        pairs_path = OUT_DIR / f"pairs_{name}.parquet"
        if not pairs_path.exists():
            print(f"\n=== {name}: übersprungen (kein {pairs_path.name}) ===")
            continue

        print(f"\n=== Validiere {pairs_path.name} gegen {raster_path.name} ===")
        errors = validate(pairs_path, raster_path)

        if errors:
            any_errors = True
            print(f"\n  FEHLGESCHLAGEN ({len(errors)} Problem(e)):")
            for e in errors:
                print(f"    - {e}")
        else:
            print("  Alle Checks bestanden.")

    if any_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
