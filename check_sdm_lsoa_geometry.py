"""
check_sdm_lsoa_geometry.py

Schritt 2 (2026-09-17, Jakob): prueft die LSOA-Geometrien selbst - eine von der Kachel-
Ueberlappung UNABHAENGIGE moegliche Fehlerquelle fuer die SDM-Poolung:
  1. Ueberschneiden sich LSOA-Polygone untereinander (Topologiefehler)?
  2. Liegen LSOAs teilweise ausserhalb des AOI-Polygons (Greater London, data/aoi/london.gpkg),
     und wie viele Pixel fehlen dadurch?
  3. Verteilung der Pixelzahl pro LSOA (aus den deduplizierten Poolungen, Schritt 1) und Minimum,
     fuer beide Zaehlungen (n_pixels_all, n_pixels_pure).

Nur lesen und berichten, nichts veraendert.
"""
import geopandas as gpd
import pandas as pd

import config as cfg
from build_sdm_lsoa_splits import get_aoi_utm_reference, load_lsoa_geometries_for_london
from build_sdm_targets import load_ts062, CODE_COL

AOI_PATH = cfg.AOI_DIR / "london.gpkg"


def main():
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    ts062 = load_ts062()
    ts062_codes = set(ts062[CODE_COL])
    lsoa_national = load_lsoa_geometries_for_london(aoi)
    lsoa_utm = lsoa_national[lsoa_national["LSOA21CD"].isin(ts062_codes)].copy()
    lsoa_utm = lsoa_utm.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]].to_crs(utm_crs)
    print(f"{len(lsoa_utm)} LSOAs (wie build_sdm_pooled_embeddings.py::main())")

    # --- 1. Ueberschneidung LSOA-Polygone untereinander ---
    print("\n=== 1. LSOA-Selbstueberschneidung ===")
    total_area = lsoa_utm.geometry.area.sum()
    union_area = lsoa_utm.geometry.union_all().area
    overlap_area = total_area - union_area
    print(f"  Summe Einzelflaechen: {total_area / 1e6:.4f} km^2")
    print(f"  Flaeche der Vereinigung: {union_area / 1e6:.4f} km^2")
    print(f"  Ueberschneidung: {overlap_area / 1e6:.6f} km^2 ({100 * overlap_area / total_area:.4f}%)")

    if overlap_area > 1.0:  # > 1 m^2 - alles darunter ist reines Gleitkomma-/Topologie-Rauschen
        sidx = lsoa_utm.sindex
        geoms = lsoa_utm.geometry.to_numpy()
        codes = lsoa_utm["lsoa_code"].to_numpy()
        pairs = []
        for i, geom in enumerate(geoms):
            for j in sidx.intersection(geom.bounds):
                if j <= i:
                    continue
                inter_area = geom.intersection(geoms[j]).area
                if inter_area > 1.0:
                    pairs.append((codes[i], codes[j], inter_area))
        print(f"  Betroffene Polygonpaare (>1 m^2 Overlap): {len(pairs)}")
        for a, b, area in sorted(pairs, key=lambda x: -x[2])[:20]:
            print(f"    {a} <-> {b}: {area:.1f} m^2")
    else:
        print("  Keine nennenswerte Ueberschneidung - reines Gleitkomma-/Topologie-Rauschen.")

    # --- 2. LSOAs teilweise ausserhalb des AOI-Polygons ---
    print("\n=== 2. LSOAs teilweise ausserhalb des AOI-Polygons ===")
    aoi_utm = aoi.to_crs(utm_crs) if str(aoi.crs) != str(utm_crs) else aoi
    aoi_geom = aoi_utm.geometry.iloc[0]
    lsoa_utm["area_m2"] = lsoa_utm.geometry.area
    lsoa_utm["area_in_aoi_m2"] = lsoa_utm.geometry.intersection(aoi_geom).area
    lsoa_utm["area_outside_m2"] = lsoa_utm["area_m2"] - lsoa_utm["area_in_aoi_m2"]
    lsoa_utm["frac_outside"] = lsoa_utm["area_outside_m2"] / lsoa_utm["area_m2"]

    n_partial = int((lsoa_utm["area_outside_m2"] > 1.0).sum())
    total_outside_m2 = lsoa_utm["area_outside_m2"].sum()
    print(f"  LSOAs mit >1 m^2 Flaeche ausserhalb des AOI-Polygons: {n_partial}/{len(lsoa_utm)}")
    print(f"  Gesamtflaeche ausserhalb: {total_outside_m2 / 1e6:.6f} km^2 "
          f"(~{total_outside_m2 / 100:.0f} Pixel bei 100 m^2/Pixel, theoretische Obergrenze - "
          f"tatsaechlich fehlende Pixel koennen weniger sein, falls dort ohnehin kein Kachel-Overlap "
          f"vorlag)")
    top = lsoa_utm.sort_values("area_outside_m2", ascending=False).head(10)
    print(top[["lsoa_code", "area_m2", "area_outside_m2", "frac_outside"]].to_string(index=False))

    # --- 3. Verteilung der Pixelzahl pro LSOA (aus den dedup-Poolungen, Schritt 1) ---
    print("\n=== 3. Verteilung Pixelzahl pro LSOA (dedupliziert, Schritt 1) ===")
    for backbone in ["ae", "tessera"]:
        df = pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_arma_dedup.parquet",
                              columns=["lsoa_code", "n_pixels_all", "n_pixels_pure"])
        for col in ["n_pixels_all", "n_pixels_pure"]:
            s = df[col]
            print(f"  {backbone}/{col}: min={s.min()} p1={s.quantile(0.01):.1f} "
                  f"p5={s.quantile(0.05):.1f} median={s.median():.1f} mean={s.mean():.1f} max={s.max()}")
        n_low_10 = int((df["n_pixels_pure"] < 10).sum())
        n_low_100 = int((df["n_pixels_pure"] < 100).sum())
        print(f"  {backbone}: LSOAs mit <10 / <100 Pixeln (partitionsrein): {n_low_10} / {n_low_100}")


if __name__ == "__main__":
    main()
