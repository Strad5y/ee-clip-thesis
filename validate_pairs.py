"""
validate_pairs.py

Absicherung der von spatial_join.py erzeugten Paar-Parquets, bevor sie ins Training gehen.
Bricht bei einem fehlgeschlagenen Check ab (sys.exit(1)) statt stillschweigend weiterzumachen.

spatial_join.py joint jetzt kachelweise (siehe dortiger Docstring) statt gegen ein einzelnes
Mosaik - raster_path ist hier deshalb eine KACHELLISTE, und row/col sind nur INNERHALB einer
Kachel eindeutig (Spalte 'tile' identifiziert die Quellkachel jeder Zeile). Checks 5-7 sind
entsprechend kachelweise (Check 5: pro Kachel eigene Transform/CRS) bzw. tile-aware (Check 6:
Duplikat-Schluessel ist (poi_id, tile, row, col)) umgebaut.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from spatial_join import BACKBONES, ID_COL, OUT_DIR, POIS_IN

# Pixelzentrum darf max. so viele Pixelbreiten von der POI-Geometrie entfernt liegen -
# fängt CRS-/Transform-Fehler ab, ohne legitime Randpixel (geringe coverage) zu verwerfen
MAX_DIST_IN_PIXELS = 3


def validate(pairs_path, raster_paths, pois_path=POIS_IN, id_col=ID_COL):
    errors = []

    pairs = pd.read_parquet(pairs_path)
    pois = gpd.read_parquet(pois_path)

    raster_paths = [Path(p) for p in raster_paths if Path(p).exists()]
    tile_info = {}
    for rp in raster_paths:
        with rasterio.open(rp) as src:
            tile_info[rp.name] = {"n_bands": src.count, "transform": src.transform, "crs": src.crs}

    # 1) Embedding-Dimension konsistent und korrekt (alle Kacheln muessen dieselbe Bandanzahl haben)
    dims = pairs["embedding"].apply(len)
    n_bands_set = {info["n_bands"] for info in tile_info.values()}
    if dims.nunique() != 1:
        errors.append(f"Uneinheitliche Embedding-Dim: {dims.unique()}")
    elif len(n_bands_set) != 1:
        errors.append(f"Kacheln haben uneinheitliche Bandanzahl: {n_bands_set}")
    elif dims.iloc[0] != next(iter(n_bands_set)):
        errors.append(f"Embedding-Dim {dims.iloc[0]} != Bandanzahl Kacheln {n_bands_set}")
    else:
        print(f"  OK Embedding-Dim = {next(iter(n_bands_set))} (konsistent ueber {len(tile_info)} Kacheln)")

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

    # 5) räumlicher Sanity-Check: Pixelzentrum nahe an der POI-Geometrie (fängt CRS-Fehler) -
    # PRO KACHEL, da jede Kachel ihre eigene Transform/CRS hat (nicht mehr ein gemeinsames
    # Mosaik). POIs werden pro EINDEUTIGER CRS reprojiziert (gecacht), nicht pro Kachel -
    # mehrere Kacheln teilen sich oft dieselbe CRS.
    pois_unique = pois.drop_duplicates(subset=id_col).set_index(id_col)
    crs_cache = {}

    n_far_total = 0
    max_dist_seen = 0.0
    missing_tiles = set()
    if "tile" not in pairs.columns:
        errors.append("Spalte 'tile' fehlt - kachelweiser Sanity-Check (Check 5) nicht möglich")
    else:
        for tile_name, group in pairs.groupby("tile"):
            info = tile_info.get(tile_name)
            if info is None:
                missing_tiles.add(tile_name)
                continue
            transform, crs = info["transform"], info["crs"]
            crs_key = str(crs)
            if crs_key not in crs_cache:
                crs_cache[crs_key] = pois_unique.to_crs(crs)
            pois_proj = crs_cache[crs_key]

            px_w = abs(transform.a)
            max_dist = MAX_DIST_IN_PIXELS * px_w

            xs, ys = rasterio.transform.xy(transform, group["row"].to_numpy(), group["col"].to_numpy())
            centers = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=crs)
            geoms = pois_proj.loc[group[id_col].to_numpy(), "geometry"].reset_index(drop=True)
            dists = centers.reset_index(drop=True).distance(geoms)

            n_far_total += int((dists > max_dist).sum())
            max_dist_seen = max(max_dist_seen, float(dists.max()))

        if missing_tiles:
            errors.append(
                f"{len(missing_tiles)} Kachel(n) aus pairs['tile'] nicht unter raster_paths "
                f"gefunden, z.B. {list(missing_tiles)[:5]}"
            )
        if n_far_total > 0:
            errors.append(
                f"{n_far_total} Paare (über alle Kacheln) mit Pixelzentrum > "
                f"{MAX_DIST_IN_PIXELS} Pixelbreiten von der POI-Geometrie entfernt "
                f"(deutet auf CRS-/Transform-Fehler hin)"
            )
        elif not missing_tiles:
            print(f"  OK alle Pixelzentren <= {MAX_DIST_IN_PIXELS} Pixel von ihrer POI-Geometrie "
                  f"entfernt (kachelweise geprüft über {len(tile_info)} Kacheln, "
                  f"max beobachtet: {max_dist_seen:.2f} m)")

    # 6) Keine Duplikate: jedes (poi_id, tile, row, col) darf nur einmal vorkommen - deckt sowohl
    # eine versehentliche Paar-Vervielfachung durchs Long-Format als auch eine Doppelzaehlung
    # durch MultiPolygon-Teile ab (beide Faelle wuerden sonst still durchrutschen, weil Check 4
    # nur prueft, OB eine poi_id bekannt ist, nicht WIE OFT sie vorkommt). 'tile' ist Teil des
    # Schluessels, weil row/col nur innerhalb einer Kachel eindeutig sind.
    dup_key = [id_col, "tile", "row", "col"] if "tile" in pairs.columns else [id_col, "row", "col"]
    dup_mask = pairs.duplicated(subset=dup_key, keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        n_dup_cells = pairs.loc[dup_mask, dup_key].drop_duplicates().shape[0]
        errors.append(
            f"{n_dup_rows} Zeilen sind Duplikate von ({', '.join(dup_key)}) ({n_dup_cells} betroffene "
            f"Pixel) - deutet auf Paar-Vervielfachung durchs Long-Format oder Doppelzaehlung durch "
            f"MultiPolygon-Teile hin"
        )
    else:
        print(f"  OK keine Duplikate in ({', '.join(dup_key)})")

    # 7) sum(coverage / n_pixels_poi) je poi_id (= das Sampling-Gewicht, siehe dataset.py) muss
    # in [0,1] liegen - das ist algebraisch der gewichtete Mittelwert der Pixel-Coverage dieses
    # POI (jetzt ueber ALLE Kacheln hinweg summiert, siehe spatial_join.py::join_pois_to_tiles)
    # und darf NICHT mit der Polygongroesse (n_pixels_poi) skalieren, genau das ist der Zweck
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
    for name, raster_paths in BACKBONES.items():
        pairs_path = OUT_DIR / f"pairs_{name}.parquet"
        if not pairs_path.exists():
            print(f"\n=== {name}: übersprungen (kein {pairs_path.name}) ===")
            continue

        existing = [Path(p) for p in raster_paths if Path(p).exists()]
        print(f"\n=== Validiere {pairs_path.name} gegen {len(existing)} Kachel(n) ===")
        errors = validate(pairs_path, existing)

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
    """Zwei synthetische Mini-Kacheln (unterschiedliche CRS, wie TESSERA an der Zonengrenze)
    statt echter Daten (pairs_*.parquet existieren noch nicht - Paarbau ist diese Session
    bewusst nicht gestartet). Prueft: sauberer Fall (2 Kacheln) fehlerfrei, ein absichtlich
    injiziertes (poi_id, tile, row, col)-Duplikat wird von Check 6 erkannt, UND zwei Kacheln
    mit IDENTISCHEN row/col aber verschiedenem 'tile' werden NICHT faelschlich als Duplikat
    markiert (das waere der Bug, den 'tile' im Schluessel verhindern soll)."""
    import tempfile

    from shapely.geometry import Point

    n_bands = 2

    def make_tile(path, crs, origin_x, origin_y):
        transform = rasterio.transform.from_origin(west=origin_x, north=origin_y, xsize=10, ysize=10)
        with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=n_bands,
                            dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(np.random.default_rng(0).standard_normal((n_bands, 10, 10)).astype(np.float32))
        return transform

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tile_a = tmp / "tile_a.tif"
        tile_b = tmp / "tile_b.tif"
        # Zwei Kacheln in verschiedenen CRS (wie 30N/31N) mit identischem row/col=(3,5) -
        # muss NICHT als Duplikat erkannt werden, weil 'tile' unterscheidet
        transform_a = make_tile(tile_a, "EPSG:32630", 0, 100)
        transform_b = make_tile(tile_b, "EPSG:32631", 500000, 5700000)

        xs_a, ys_a = rasterio.transform.xy(transform_a, [0, 3], [0, 5])
        xs_b, ys_b = rasterio.transform.xy(transform_b, [3], [5])

        pairs_clean = pd.DataFrame({
            ID_COL: ["p0", "p1", "p2"],
            "tile": ["tile_a.tif", "tile_a.tif", "tile_b.tif"],
            "row": [0, 3, 3],
            "col": [0, 5, 5],
            "coverage": [1.0, 0.6, 1.0],
            "n_pixels_poi": [1, 1, 1],
            "embedding": [np.random.default_rng(i).standard_normal(n_bands).astype(np.float32)
                          for i in range(3)],
        })

        pois = gpd.GeoDataFrame(
            {ID_COL: ["p0", "p1", "p2"]},
            geometry=[
                Point(xs_a[0], ys_a[0]),
                Point(xs_a[1], ys_a[1]),
                gpd.GeoSeries([Point(xs_b[0], ys_b[0])], crs="EPSG:32631").to_crs("EPSG:32630").iloc[0],
            ],
            crs="EPSG:32630",
        )

        pairs_path = tmp / "pairs_clean.parquet"
        pois_path = tmp / "pois.parquet"
        pairs_clean.to_parquet(pairs_path)
        pois.to_parquet(pois_path)

        errors_clean = validate(pairs_path, [tile_a, tile_b], pois_path=pois_path)
        assert not errors_clean, f"Sauberer Fall (2 Kacheln, gleiches row/col) sollte fehlerfrei sein: {errors_clean}"

        # Echtes Duplikat einschleusen: dieselbe (poi_id, tile, row, col)-Zeile nochmal anhaengen
        pairs_dup = pd.concat([pairs_clean, pairs_clean.iloc[[1]]], ignore_index=True)
        pairs_dup_path = tmp / "pairs_dup.parquet"
        pairs_dup.to_parquet(pairs_dup_path)

        errors_dup = validate(pairs_dup_path, [tile_a, tile_b], pois_path=pois_path)
        assert any("Duplikate" in e for e in errors_dup), \
            f"Check 6 haette das echte Duplikat erkennen muessen, bekam: {errors_dup}"

    print("Smoke-Test OK: 2 Kacheln mit identischem row/col aber verschiedenem 'tile' korrekt "
          "NICHT als Duplikat markiert; echtes (poi_id, tile, row, col)-Duplikat korrekt erkannt.")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        main()
