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

    # 6) Keine Duplikate: jedes (poi_id, row, col) darf nur einmal vorkommen - deckt sowohl
    # eine versehentliche Paar-Vervielfachung durchs Long-Format als auch eine Doppelzaehlung
    # durch MultiPolygon-Teile ab (beide Faelle wuerden sonst still durchrutschen, weil Check 4
    # nur prueft, OB eine poi_id bekannt ist, nicht WIE OFT sie vorkommt)
    dup_mask = pairs.duplicated(subset=[id_col, "row", "col"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        n_dup_cells = pairs.loc[dup_mask, [id_col, "row", "col"]].drop_duplicates().shape[0]
        errors.append(
            f"{n_dup_rows} Zeilen sind Duplikate von ({id_col}, row, col) ({n_dup_cells} betroffene "
            f"Pixel) - deutet auf Paar-Vervielfachung durchs Long-Format oder Doppelzaehlung durch "
            f"MultiPolygon-Teile hin"
        )
    else:
        print(f"  OK keine Duplikate in ({id_col}, row, col)")

    # 7) sum(coverage / n_pixels_poi) je poi_id (= das Sampling-Gewicht, siehe dataset.py) muss
    # in [0,1] liegen - das ist algebraisch der gewichtete Mittelwert der Pixel-Coverage dieses
    # POI und darf NICHT mit der Polygongroesse (n_pixels_poi) skalieren, genau das ist der Zweck
    # der Normierung (A1/F1). Ein Wert ausserhalb [0,1] zeigt inkonsistentes n_pixels_poi an,
    # typischerweise durch Duplikate (siehe Check 6).
    if "n_pixels_poi" not in pairs.columns:
        errors.append("Spalte 'n_pixels_poi' fehlt - Sampling-Gewicht coverage/n_pixels_poi nicht pruefbar")
    else:
        weight_sum = (pairs["coverage"] / pairs["n_pixels_poi"]).groupby(pairs[id_col]).sum()
        out_of_range = weight_sum[(weight_sum < 0) | (weight_sum > 1)]
        if len(out_of_range) > 0:
            errors.append(
                f"{len(out_of_range)} poi_id mit sum(coverage/n_pixels_poi) außerhalb [0,1] "
                f"(z.B. {out_of_range.head(3).to_dict()}) - deutet auf inkonsistentes n_pixels_poi hin "
                f"(z.B. durch Duplikate, siehe Check 6)"
            )
        else:
            n_pix_per_poi = pairs.groupby(id_col)["n_pixels_poi"].first().reindex(weight_sum.index)
            corr = weight_sum.corr(n_pix_per_poi)
            print(f"  OK sum(coverage/n_pixels_poi) je poi_id in [0,1] "
                  f"(mean={weight_sum.mean():.4f}, entspricht der mittleren Pixel-Coverage je POI); "
                  f"Korrelation mit n_pixels_poi={corr:.3f} (nahe 0 = unabhaengig von der "
                  f"Polygongroesse, wie beabsichtigt)")

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


def _smoke_test():
    """Synthetischer Mini-Raster + Pairs/POIs statt echter Daten (pairs_*.parquet existieren
    noch nicht - Paarbau ist diese Session bewusst nicht gestartet). Prueft Check 6/7 (F5):
    ein sauberer Fall muss fehlerfrei durchlaufen, ein absichtlich duplizierter Fall muss
    Check 6 (und in Folge Check 7) auslösen."""
    import tempfile
    from pathlib import Path

    from shapely.geometry import Point

    crs = "EPSG:32630"
    transform = rasterio.transform.from_origin(west=0, north=100, xsize=10, ysize=10)
    n_bands = 2

    with tempfile.TemporaryDirectory() as tmp:
        raster_path = Path(tmp) / "mini.tif"
        with rasterio.open(raster_path, "w", driver="GTiff", height=10, width=10, count=n_bands,
                            dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(np.random.default_rng(0).standard_normal((n_bands, 10, 10)).astype(np.float32))

        # p0: 1 Pixel (Punkt), p1: 2 Pixel (Polygon-Fall) - beide mit korrektem n_pixels_poi
        rows = [0, 3, 3]
        cols = [0, 5, 6]
        poi_ids = ["p0", "p1", "p1"]
        coverage = [1.0, 0.6, 0.4]
        n_pixels_poi = [1, 2, 2]

        def make_pairs(rows, cols, poi_ids, coverage, n_pixels_poi):
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            return pd.DataFrame({
                ID_COL: poi_ids,
                "row": rows,
                "col": cols,
                "coverage": coverage,
                "n_pixels_poi": n_pixels_poi,
                "embedding": [np.random.default_rng(i).standard_normal(n_bands).astype(np.float32)
                              for i in range(len(rows))],
            }), xs, ys

        pairs_clean, xs, ys = make_pairs(rows, cols, poi_ids, coverage, n_pixels_poi)

        pois = gpd.GeoDataFrame(
            {ID_COL: ["p0", "p1"]},
            geometry=[Point(xs[0], ys[0]), Point(xs[1], ys[1])],
            crs=crs,
        )

        pairs_path = Path(tmp) / "pairs_clean.parquet"
        pois_path = Path(tmp) / "pois.parquet"
        pairs_clean.to_parquet(pairs_path)
        pois.to_parquet(pois_path)

        errors_clean = validate(pairs_path, raster_path, pois_path=pois_path)
        assert not errors_clean, f"Sauberer Fall sollte fehlerfrei sein, bekam: {errors_clean}"

        # Duplikat einschleusen: dieselbe (poi_id, row, col)-Zeile wie p1@(3,5) nochmal anhaengen
        pairs_dup = pd.concat([pairs_clean, pairs_clean.iloc[[1]]], ignore_index=True)
        pairs_dup_path = Path(tmp) / "pairs_dup.parquet"
        pairs_dup.to_parquet(pairs_dup_path)

        errors_dup = validate(pairs_dup_path, raster_path, pois_path=pois_path)
        assert any("Duplikate" in e for e in errors_dup), \
            f"Check 6 haette das Duplikat erkennen muessen, bekam: {errors_dup}"

    print("Smoke-Test OK: sauberer Fall fehlerfrei, duplizierter (poi_id, row, col)-Fall "
          "von Check 6 korrekt erkannt.")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        main()
