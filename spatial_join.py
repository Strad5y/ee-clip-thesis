"""
spatial_join.py

Kern-Join: POI-Geometrie x EE-Pixel -> Trainingspaare.
Settled Decision (CLAUDE.md): jedes EE-Pixel, das eine POI-Geometrie schneidet, wird ein
Trainingspaar; coverage in [0,1] ist der Flaechenanteil des Pixels, den die Geometrie abdeckt
(Punkte zaehlen immer als coverage=1.0).

Backbones sind KACHELLISTEN, kein einzelnes Mosaik: pro Kachel wird separat gejoint - die
POIs werden dabei in die jeweils EIGENE native CRS der Kachel reprojiziert (billig, ein paar
hunderttausend Punkte/Polygone), statt das ganze Raster zu reprojizieren (teuer, siehe
Sitzungsvorfall: ein bandweiser TESSERA-Rasterreproject fuer 128 Baender x 20 Kacheln hing
fest). Die Ergebnisse werden konkateniert. AlphaEarth joint jetzt ebenfalls gegen die nativen
Rohkacheln statt gegen das fertige Mosaik (build_mosaic.py) - sonst waere AlphaEarth einmal
resampled und TESSERA nicht, eine unnoetige Asymmetrie fuer einen Backbone-Vergleich. Das
Mosaik bleibt als Artefakt liegen, wird hier aber nicht mehr referenziert. Beide Backbones
durchlaufen denselben Codepfad.

Die Spalte 'tile' identifiziert die Quellkachel jeder Zeile - row/col sind NUR INNERHALB
einer Kachel eindeutig, nicht global (siehe validate_pairs.py Check 6/7, die deshalb ueber
(poi_id, tile, row, col) statt (poi_id, row, col) pruefen).

⚠ Bekannte, noch NICHT behobene Leakage-Quelle: TESSERA-Kacheln aus den Zonen EPSG:32630
und EPSG:32631 ueberlappen sich geografisch in einem ~2.7 km breiten Streifen an der
Zonennaht (verifiziert: 6.483 von 279.988 POIs werden von Kacheln BEIDER Zonen getroffen,
davon 1.633 Punkt-POIs mit eindeutiger Doppel-Zuordnung). Der (poi_id, tile, row, col)-
Duplikat-Check faengt das NICHT, weil tile/row/col je Zone unterschiedlich sind, obwohl es
dieselbe reale Position ist. Deduplizierung ist noch nicht implementiert (Entscheidung steht
aus), siehe Sitzungsbericht.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from exactextract import exact_extract

import config as cfg
from dequantize import dequantize_ae

ID_COL = "poi_id"
CARRY = []  # zusaetzliche POI-Spalten fuer den Output; bewusst leer -
            # Tag-Verknuepfung (key/value) laeuft spaeter separat ueber poi_id,
            # sonst wuerde das Long-Format (mehrere Tags je poi_id) die Pixel-Paare vervielfachen

POIS_IN = cfg.DATA / "interim" / "pois_london_test_filtered.parquet"
POIS_IN_BY_REGION = {
    "london": POIS_IN,
    "niedersachsen": cfg.DATA / "interim" / "pois_niedersachsen_full_filtered.parquet",
}


def _ensure_north_up(raster_path: Path, cache_dir: Path) -> Path:
    """exact_extract (siehe _join_polygons) kann Raster mit bottom-up-Geotransform (positive
    y-Aufloesung, bounds.bottom > bounds.top) nicht verarbeiten - bricht mit 'Incompatible
    extents' ab. Die rohen AlphaEarth-GCS-Kacheln sind genau so gespeichert (verifiziert:
    transform.e > 0). Punkte sind NICHT betroffen (rasterio.transform.xy/rowcol handhaben
    beliebige Vorzeichen korrekt) - nur der exact_extract-Polygonpfad.

    Fix: NUR die Zeilenreihenfolge spiegeln + das Transform-Vorzeichen korrigieren (keine
    Interpolation, kein Resampling - dieselben Werte, nur umsortiert) und das Ergebnis cachen.
    Raster, die bereits north-up sind (z.B. TESSERA), werden unveraendert zurueckgegeben."""
    with rasterio.open(raster_path) as src:
        if src.transform.e < 0:
            return raster_path  # bereits north-up, nichts zu tun

        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / raster_path.name
        if out_path.exists():
            return out_path

        arr = src.read()  # (bands, H, W)
        flipped = arr[:, ::-1, :]
        new_transform = src.transform * Affine.translation(0, src.height) * Affine.scale(1, -1)

        profile = src.profile.copy()
        profile.update(transform=new_transform)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(flipped)
    print(f"  {raster_path.name}: bottom-up -> north-up gespiegelt (fuer exact_extract), "
          f"gecacht unter {out_path}")
    return out_path


# Beide Backbones joinen gegen die nativen Rohkacheln, KEIN Rasterreproject fuer irgendeinen
# der beiden - das fertige AlphaEarth-Mosaik (build_mosaic.py) waere einmal resampled gewesen,
# TESSERA nicht; fuer einen Backbone-Vergleich ist das eine unnoetige Asymmetrie. Das Mosaik
# bleibt als Artefakt liegen, wird aber hier nicht mehr referenziert.
#
# Region-Parametrisierung (Niedersachsen-Erweiterung): TESSERA-Kacheln UND AlphaEarth-Kacheln
# liegen je Region in eigenen Unterverzeichnissen (data/raw/{tessera,alphaearth/tiles}/{region}/)
# - siehe make_tessera_tif.py bzw. download_ae_tiles.py. Ein flaches AE_TILES_DIR ohne
# Region-Suffix wuerde sonst POIs einer Region gegen Kacheln einer anderen Region joinen
# (geografisch meist folgenlos, da 0 Treffer, aber langsam und nicht reproduzierbar).
ZONES_BY_REGION = {
    "london": ["30N", "31N"],          # 0-Meridian, siehe CLAUDE.md
    "niedersachsen": ["32N"],          # eine einzelne UTM-Zone, siehe CLAUDE.md
}
AE_NORTHUP_DIR = cfg.DATA / "interim" / "ae_northup"


def tessera_dir(region: str) -> Path:
    return cfg.DATA / "raw" / "tessera" / region / "global_0.1_degree_representation" / "2025"


def ae_tiles_dir(region: str) -> Path:
    return cfg.DATA / "raw" / "alphaearth" / "tiles" / region


def get_backbones(region: str = "london") -> dict:
    """Baut das {backbone: [Kachelpfade]}-Dict fuer eine Region neu (statt es einmalig beim
    Modul-Import fest zu verdrahten) - noetig, weil mehrere Regionen mit unterschiedlichen
    Kachelverzeichnissen im selben Prozess/Testlauf existieren koennen."""
    tdir = tessera_dir(region)
    adir = ae_tiles_dir(region)
    return {
        "tessera": sorted(tdir.glob("*/*.tiff")),
        "ae": [_ensure_north_up(p, AE_NORTHUP_DIR) for p in sorted(adir.glob("*.tiff"))],
    }


# Backward-kompatibler Default (London) - bestehende Importe (`from spatial_join import
# BACKBONES`) funktionieren unveraendert weiter.
TESSERA_DIR = tessera_dir("london")
AE_TILES_DIR = ae_tiles_dir("london")
BACKBONES = get_backbones("london")

# UTM-Zonen-Zuordnung (Nutzervorgabe): London spreizt 30N/31N am 0-Meridian, deren Kachel-
# Gitter ueberlappen sich geografisch (~2.7 km breiter Streifen, verifiziert: 6.483 von
# 279.988 POIs von Kacheln BEIDER Zonen getroffen -> systematisch verdoppeltes n_pixels_poi,
# halbiertes Sampling-Gewicht fuer genau diesen raeumlichen Streifen). Fix: jeder POI wird
# GENAU EINER Zone zugeordnet (Laenge des Zentroids ggue. der nominellen Grenze bei 0 Grad),
# beim Join werden fuer diesen POI nur Kacheln der zugewiesenen Zone beruecksichtigt - gilt
# fuer TESSERA und AlphaEarth gleichermassen (beide haben 30N/31N-Kacheln).
#
# Niedersachsen hat nur EINE UTM-Zone (32N, siehe CLAUDE.md) - keine Zonennaht, keine
# Ueberlappung, also keine Zentroid-Grenzlogik noetig (assign_utm_zone nimmt dafuer einen
# Fast-Path: bei genau einer moeglichen Zone wird sie ungeprueft an alle POIs vergeben).
UTM_ZONE_BOUNDARY_LON = 0.0
ZONE_CRS = {"30N": "EPSG:32630", "31N": "EPSG:32631", "32N": "EPSG:32632"}
ZONE_CRS_INV = {v: k for k, v in ZONE_CRS.items()}
DEFAULT_ZONES = ("30N", "31N")


def assign_utm_zone(gdf: gpd.GeoDataFrame, zones=DEFAULT_ZONES) -> pd.Series:
    """Ordnet jedem POI (Index-aligned) genau eine UTM-Zone aus `zones` zu.

    Ein-Zonen-Regionen (z.B. Niedersachsen/32N): jeder POI bekommt trivial die eine Zone,
    keine Grenzberechnung noetig.

    Zwei-Zonen-Fall (London/30N+31N, die einzige bisher unterstuetzte Mehrzonen-Kombination):
    Zentroid-Laenge < 0 Grad -> erste Zone, >= 0 Grad -> zweite Zone. Zentroid wird in einer
    projizierten CRS (EPSG:3857) berechnet statt direkt in Grad - geopandas warnt sonst zu
    Recht vor verzerrten Zentroiden in geografischer CRS; fuer die hier benoetigte Praezision
    (nur das Vorzeichen ggue. 0 Grad, POI-Polygone sind klein) ist die Web-Mercator-Verzerrung
    irrelevant."""
    if len(zones) == 1:
        return pd.Series(zones[0], index=gdf.index)
    if len(zones) != 2:
        raise NotImplementedError(
            f"assign_utm_zone unterstuetzt nur 1 Zone oder das London-Paar {DEFAULT_ZONES}, "
            f"bekam: {zones}")
    gdf_3857 = gdf.to_crs("EPSG:3857")
    centroid_lon = gdf_3857.geometry.centroid.to_crs("EPSG:4326").x
    return pd.Series(np.where(centroid_lon < UTM_ZONE_BOUNDARY_LON, zones[0], zones[1]), index=gdf.index)


# Backbone-spezifische Dequantisierung (siehe dequantize.py) - None = keine noetig/bekannt.
# Greift weiterhin, auch gegen die rohen int8-Kacheln direkt (nicht mehr nur gegen das Mosaik).
DEQUANTIZE = {
    "tessera": None,  # bereits float32 beim Download (siehe Sitzungsbericht)
    "ae": dequantize_ae,
}
OUT_DIR = cfg.DATA / "processed"


def _join_points(points: gpd.GeoDataFrame, raster_path, id_col, carry) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        xs = points.geometry.x.to_numpy()
        ys = points.geometry.y.to_numpy()
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        in_bounds = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        n_out = int((~in_bounds).sum())
        if n_out > 0:
            # Kein Datenverlust-Hinweis: bei kachelweisem Join (siehe join_pois_to_tiles) ist
            # ein Punkt fuer JEDE Kachel seiner Zone ausser der einen, die ihn tatsaechlich
            # enthaelt, "ausserhalb" - das ist der Normalfall, nicht ein Fehlschlag. Ein echter
            # Fehlschlag (POI in KEINER Kachel) zeigt sich stattdessen in "POIs ohne Pixel"
            # auf Gesamtergebnis-Ebene (siehe validate_pairs.py Check 4).
            print(f"  Points: {n_out} nicht in dieser Kachel (normal bei kachelweisem Join)")

        out = pd.DataFrame({
            id_col: points[id_col].to_numpy()[in_bounds],
            "row": rows[in_bounds],
            "col": cols[in_bounds],
            "coverage": 1.0,
        })
        for c in carry:
            out[c] = points[c].to_numpy()[in_bounds]
        return out


def _join_polygons(polys: gpd.GeoDataFrame, raster_path, id_col, carry) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        width = src.width

    cols_to_include = [id_col] + list(carry)
    res = exact_extract(
        str(raster_path), polys, ["cell_id", "coverage"],
        include_cols=cols_to_include, output="pandas",
    )

    # cell_id/coverage sind pro Band identisch (gleiches Grid) -> Band 1 genuegt
    records = []
    for _, row in res.iterrows():
        cell_ids = np.asarray(row["band_1_cell_id"])
        coverages = np.asarray(row["band_1_coverage"])
        if len(cell_ids) == 0:
            continue
        rows_ = cell_ids // width
        cols_ = cell_ids % width
        n = len(cell_ids)
        rec = {
            id_col: np.full(n, row[id_col]),
            "row": rows_,
            "col": cols_,
            "coverage": coverages,
        }
        for c in carry:
            rec[c] = np.full(n, row[c])
        records.append(pd.DataFrame(rec))

    if not records:
        # Explizite dtypes noetig: ein leeres pd.DataFrame(columns=[...]) hat sonst object-dtype
        # fuer row/col. Beim spaeteren pd.concat() mit einem nicht-leeren Punkte-Ergebnis
        # (int64) castet pandas die Spalte insgesamt auf object hoch, auch wenn alle echten
        # Werte Integer sind - numpy verweigert dann object-Arrays als Fancy-Index
        # ("arrays used as indices must be of integer (or boolean) type"). Trifft in der Praxis
        # genau die kleinen TESSERA-Kacheln, die fuer eine Teilmenge Punkte aber keine
        # Polygone (oder umgekehrt) enthalten.
        empty = pd.DataFrame(columns=[id_col, "row", "col", "coverage", *carry])
        return empty.astype({"row": "int64", "col": "int64", "coverage": "float64"})
    return pd.concat(records, ignore_index=True)


def join_pois_to_raster(gdf: gpd.GeoDataFrame, raster_path, id_col=ID_COL, carry=CARRY,
                         dequantize=None) -> pd.DataFrame:
    """POIs x EIN Raster (eine Kachel oder ein Mosaik) -> DataFrame mit einer Zeile pro (poi,
    geschnittenem Pixel), inkl. Spalte 'tile' = Dateiname dieses Rasters.

    gdf wird HIER in die CRS DIESES Rasters reprojiziert (uebersprungen, wenn gdf schon in
    dieser CRS ist - siehe join_pois_to_tiles, das bei kachelweisem Aufruf VORHER einmal pro
    Zone reprojiziert, statt es hier pro Kachel neu zu tun: bei vielen kleinen Kacheln
    derselben Zone, z.B. 1474 TESSERA-Kacheln/Niedersachsen (alle EPSG:32632), war die
    wiederholte pyproj-Transformation derselben 20.000 POIs 1474x pro Block der dominante
    Kostenfaktor, nicht der Pixel-Lookup selbst).

    dequantize: optionale Funktion (n, bands) int8 -> (n, bands) float, angewendet NACH dem
    Nachschlagen der rohen Bandwerte und VOR dem Speichern als 'embedding' (siehe dequantize.py -
    z.B. AlphaEarth ist quantisiert verteilt und braucht eine nichtlineare Dequantisierung,
    sonst sind die Rohwerte weder float noch einheitsnormiert). n_pixels_poi wird HIER NICHT
    gesetzt - das passiert in join_pois_to_tiles() nach dem Konkatenieren aller Kacheln, falls
    ein POI mehrere Kacheln schneidet."""
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    points = gdf[gdf.geometry.geom_type == "Point"]
    polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    parts = []
    if len(points) > 0:
        parts.append(_join_points(points, raster_path, id_col, carry))
    if len(polys) > 0:
        parts.append(_join_polygons(polys, raster_path, id_col, carry))

    pairs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(pairs) == 0:
        return pairs

    # Embedding-Werte je Pixel nachschlagen: adaptive Strategie je nach Trefferdichte.
    # - WENIGE Treffer ggue. Kachelgroesse (typisch: kachelweiser Join eines kleinen
    #   POI-Blocks gegen VIELE kleine Kacheln, z.B. TESSERA/Niedersachsen 1474 Kacheln):
    #   rasterio.sample() liest nur die GDAL-Bloecke der angefragten Punkte - ein
    #   begrenzendes Fenster hilft hier NICHT zuverlaessig, weil POIs innerhalb einer Kachel
    #   raeumlich verstreut sind (gemessen: Fenster deckte bei 5-20 Treffern oft 30-95% der
    #   Kachel ab). Gemessen: sample() ~135x schneller als windowed read bei dieser Dichte.
    # - VIELE Treffer (z.B. AE: wenige riesige Kacheln, oft ein Grossteil eines POI-Blocks
    #   pro Kachel): ein einziger Vollread ist guenstiger als viele einzelne
    #   1x1-Fenster-Reads (sample() hat ~20ms Fixkosten PRO Punkt - siehe rasterio.sample.
    #   sample_gen, das pro Punkt einen eigenen dataset.read()-Aufruf macht, unabhaengig von
    #   gemeinsam genutzten internen Bloecken).
    with rasterio.open(raster_path) as src:
        n_tile_px = src.width * src.height
        sparse = len(pairs) < 2000 and len(pairs) < 0.02 * n_tile_px
        if sparse:
            xs, ys = rasterio.transform.xy(src.transform, pairs["row"].to_numpy(), pairs["col"].to_numpy())
            band_vals = np.array(list(src.sample(zip(xs, ys))))  # (n_pairs, bands)
        else:
            arr = src.read()  # (bands, H, W)
            band_vals = arr[:, pairs["row"].to_numpy(), pairs["col"].to_numpy()].T  # (n_pairs, bands)
    if dequantize is not None:
        band_vals = dequantize(band_vals)
    else:
        band_vals = band_vals.astype(np.float32)
    pairs["embedding"] = list(band_vals)
    pairs["tile"] = raster_path.name if hasattr(raster_path, "name") else str(raster_path)
    return pairs


def join_pois_to_tiles(gdf: gpd.GeoDataFrame, tile_paths, id_col=ID_COL, carry=CARRY,
                        dequantize=None, zones=DEFAULT_ZONES) -> pd.DataFrame:
    """Iteriert ueber eine Kachelliste, joint POIs gegen JEDE Kachel einzeln (POIs werden pro
    Aufruf guenstig in die jeweils native CRS der Kachel reprojiziert - kein Rasterreproject),
    konkateniert die Ergebnisse. n_pixels_poi wird ERST NACH dem Konkatenieren berechnet
    (ueber ALLE Kacheln hinweg), falls ein POI mehrere Kacheln schneidet.

    UTM-Zonen-Dedup: jeder POI wird VORHER genau einer Zone aus `zones` zugeordnet
    (assign_utm_zone) - eine Kachel, deren CRS eine bekannte Zone ist (siehe ZONE_CRS), bekommt
    beim Join NUR die POIs dieser Zone. Verhindert, dass ein POI im geografischen
    Ueberlappungsstreifen zweier Zonen doppelte (redundante) Pixel aus beiden Kacheln bekommt.
    Fuer Ein-Zonen-Regionen (zones mit Laenge 1, z.B. Niedersachsen/32N) ist das ein No-Op -
    jede Kachel bekommt einfach alle POIs. Spalte 'utm_zone' im Output macht die Zuordnung
    nachvollziehbar.

    Reprojektions-Cache: alle Kacheln derselben CRS (typischerweise alle Kacheln einer Zone)
    teilen sich EINE reprojizierte Kopie des zonengefilterten gdf, statt bei jedem Kachel-Aufruf
    (in join_pois_to_raster) neu zu reprojizieren - bei vielen kleinen Kacheln derselben Zone
    (z.B. 1474 TESSERA-Kacheln/Niedersachsen) war die wiederholte pyproj-Transformation
    derselben POIs sonst der dominante Kostenfaktor (gemessen: >>1h Hochrechnung fuer wenige
    Bloecke)."""
    zone_per_poi = assign_utm_zone(gdf, zones=zones)
    reproj_cache: dict = {}  # tile_crs (str) -> bereits reprojizierte, zonengefilterte Kopie

    parts = []
    for i, tile_path in enumerate(tile_paths):
        with rasterio.open(tile_path) as src:
            tile_crs = str(src.crs)
        tile_zone = ZONE_CRS_INV.get(tile_crs)

        if tile_crs not in reproj_cache:
            base_subset = gdf[zone_per_poi == tile_zone] if tile_zone is not None else gdf
            reproj_cache[tile_crs] = base_subset.to_crs(tile_crs) if len(base_subset) > 0 else base_subset
        subset = reproj_cache[tile_crs]

        zone_note = f" (Zone {tile_zone}, {len(subset)}/{len(gdf)} POIs zugeordnet)" if tile_zone else ""
        print(f"    Kachel {i + 1}/{len(tile_paths)}: {tile_path.name}{zone_note}")
        if len(subset) == 0:
            continue

        part = join_pois_to_raster(subset, tile_path, id_col, carry, dequantize)
        if len(part) > 0:
            part["utm_zone"] = tile_zone if tile_zone is not None else "n/a"
            parts.append(part)

    if not parts:
        return pd.DataFrame()

    pairs = pd.concat(parts, ignore_index=True)
    # Pixelzahl je POI ueber ALLE Kacheln (fuer das Sampling-Gewicht coverage / n_pixels_poi,
    # siehe dataset.py): macht das Gewicht naeherungsweise unabhaengig von der Polygongroesse.
    pairs["n_pixels_poi"] = pairs.groupby(id_col)[id_col].transform("size")
    return pairs


def join_pois_to_tiles_by_tile(gdf: gpd.GeoDataFrame, tile_paths, id_col=ID_COL, carry=CARRY,
                                dequantize=None, zones=DEFAULT_ZONES, chunk_dir: Path = None) -> pd.DataFrame:
    """Wie join_pois_to_tiles(), aber mit VERTAUSCHTER Schleifenreihenfolge: die Kachel ist
    die AEUSSERE Schleife, nicht der POI-Block.

    Hintergrund: chunked_join_runner.run() zerlegt den POI-Datensatz in Bloecke (fuer Resume/
    Fortschritt) und ruft join_pois_to_tiles() EINMAL PRO BLOCK auf - jede Kachel wird dabei
    also einmal PRO BLOCK geoeffnet/verarbeitet. Fuer AlphaEarth (wenige riesige Kacheln, z.B.
    18 fuer Niedersachsen) ist das unproblematisch (18 x 257 Bloecke = 4.626 Kachel-Touches).
    Fuer TESSERA (viele kleine Kacheln, z.B. 1474 fuer Niedersachsen) ist es katastrophal:
    1474 x 257 = 378.818 Kachel-Touches, gemessene Hochrechnung >16h. Hier wird stattdessen
    jede Kachel GENAU EINMAL gegen den gesamten (zonengefilterten) POI-Datensatz verarbeitet -
    mathematisch dieselbe Kreuzmenge aus (poi, pixel)-Paaren wie join_pois_to_tiles() (nur die
    Reihenfolge der Schleifen ist vertauscht), aber ohne die 257-fache Redundanz. Reprojektion
    wird wie dort einmal pro Zone gecacht.

    Resumable ueber chunk_dir: eine Parquet-Datei pro Kachel (auch wenn 0 Paare - der leere
    Frame markiert die Kachel als "bereits verarbeitet", nicht "noch zu tun")."""
    zone_per_poi = assign_utm_zone(gdf, zones=zones)
    reproj_cache: dict = {}

    parts = []
    n_tiles = len(tile_paths)
    for i, tile_path in enumerate(tile_paths):
        chunk_path = chunk_dir / f"tile_{i:04d}.parquet" if chunk_dir is not None else None
        if chunk_path is not None and chunk_path.exists():
            existing = pd.read_parquet(chunk_path)
            if len(existing) > 0:
                parts.append(existing)
            if (i + 1) % 50 == 0 or i + 1 == n_tiles:
                print(f"  Kachel {i + 1}/{n_tiles}: {tile_path.name} - bereits vorhanden, uebersprungen", flush=True)
            continue

        with rasterio.open(tile_path) as src:
            tile_crs = str(src.crs)
        tile_zone = ZONE_CRS_INV.get(tile_crs)

        if tile_crs not in reproj_cache:
            base_subset = gdf[zone_per_poi == tile_zone] if tile_zone is not None else gdf
            reproj_cache[tile_crs] = base_subset.to_crs(tile_crs) if len(base_subset) > 0 else base_subset
        subset = reproj_cache[tile_crs]

        part = join_pois_to_raster(subset, tile_path, id_col, carry, dequantize)
        if len(part) > 0:
            part["utm_zone"] = tile_zone if tile_zone is not None else "n/a"

        if chunk_path is not None:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            part.to_parquet(chunk_path)

        if len(part) > 0:
            parts.append(part)
        print(f"  Kachel {i + 1}/{n_tiles}: {tile_path.name} -> {len(part)} Paare", flush=True)

    pairs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(pairs) > 0:
        pairs["n_pixels_poi"] = pairs.groupby(id_col)[id_col].transform("size")
    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dev-Join (kleiner Testlauf, siehe run_*_join_full.py fuer Produktion)")
    parser.add_argument("--region", type=str, default="london", choices=sorted(ZONES_BY_REGION))
    args = parser.parse_args()

    pois_in = POIS_IN_BY_REGION[args.region]
    zones = ZONES_BY_REGION[args.region]
    backbones = get_backbones(args.region)

    print(f"Region: {args.region}  Lese POIs: {pois_in}")
    gdf = gpd.read_parquet(pois_in)

    # Geometrien pro poi_id deduplizieren: das Long-Format enthaelt mehrere Tag-Zeilen
    # je poi_id (gleiche Geometrie). Fuer den Join zaehlt nur die eindeutige Geometrie;
    # die Tags (key/value) werden erst NACH dem Join separat ueber poi_id gemergt.
    n_raw = len(gdf)
    gdf = gdf.drop_duplicates(subset=ID_COL).reset_index(drop=True)
    print(f"  {n_raw} Tag-Zeilen -> {len(gdf)} eindeutige POI-Geometrien")
    print(f"  Typen: {dict(gdf.geometry.geom_type.value_counts())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, tile_paths in backbones.items():
        print(f"\n=== Backbone: {name} ===")
        existing = [p for p in tile_paths if p.exists()]
        if not existing:
            print(f"  Keine Kacheln vorhanden ({len(tile_paths)} erwartet) - übersprungen")
            continue
        if len(existing) < len(tile_paths):
            print(f"  Warnung: {len(tile_paths) - len(existing)} von {len(tile_paths)} "
                  f"Kacheln fehlen auf der Platte")

        pairs = join_pois_to_tiles(gdf, existing, dequantize=DEQUANTIZE.get(name), zones=zones)
        if len(pairs) == 0:
            print("  Keine Paare erzeugt.")
            continue

        emb_dim = len(pairs["embedding"].iloc[0])
        n_unique_pois = pairs[ID_COL].nunique()
        cov = pairs["coverage"]
        print(f"  Paare: {len(pairs)}  (aus {n_unique_pois} von {len(gdf)} POIs, "
              f"{len(existing)} Kacheln)")
        print(f"  Embedding-Dim: {emb_dim}")
        print(f"  Coverage: min={cov.min():.4f} max={cov.max():.4f} mean={cov.mean():.4f}")
        print(f"  NaN in embeddings: {sum(np.isnan(e).any() for e in pairs['embedding'])}")

        suffix = "" if args.region == "london" else f"_{args.region}"
        out_path = OUT_DIR / f"pairs_{name}{suffix}.parquet"
        pairs.to_parquet(out_path)
        print(f"  Gespeichert: {out_path}")


def _smoke_test():
    """Zwei benachbarte synthetische Mini-Kacheln (gleiche CRS, wie zwei TESSERA-0.1-Grad-
    Kacheln) statt echter Rasterdaten - pairs_*.parquet/echte Kacheln existieren noch nicht
    (Paarbau ist diese Session bewusst nicht gestartet). Prueft die eigentliche Neuerung
    dieses Umbaus: ein POI, dessen Polygon ueber die Kachelgrenze reicht, muss Zeilen aus
    BEIDEN Kacheln bekommen (Spalte 'tile' unterscheidet sie), und n_pixels_poi muss die
    Pixel BEIDER Kacheln zusammenzaehlen (nicht nur der zuletzt verarbeiteten).

    CRS bewusst NICHT EPSG:32630/32631 (siehe ZONE_CRS) - dieser Test prueft die
    Kachel-Konkatenierung isoliert von der UTM-Zonen-Zuordnung (siehe _smoke_test_utm_zone
    dafuer); mit einer echten Zonen-CRS wuerden die (hier voellig unrealistischen, kleinen)
    Testkoordinaten eine zufaellige Zone zugewiesen bekommen und den Test verfaelschen."""
    import tempfile

    from shapely.geometry import Point, box

    crs = "EPSG:3857"
    n_bands = 3

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tile_a = tmp / "tile_a.tif"
        tile_b = tmp / "tile_b.tif"

        # Kachel A: x in [0,100), Kachel B: x in [100,200), beide y in [0,100), Pixelgroesse 10
        transform_a = rasterio.transform.from_origin(west=0, north=100, xsize=10, ysize=10)
        transform_b = rasterio.transform.from_origin(west=100, north=100, xsize=10, ysize=10)
        rng = np.random.default_rng(0)
        for path, transform in [(tile_a, transform_a), (tile_b, transform_b)]:
            with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=n_bands,
                                dtype="float32", crs=crs, transform=transform) as dst:
                dst.write(rng.standard_normal((n_bands, 10, 10)).astype(np.float32))

        gdf = gpd.GeoDataFrame(
            {ID_COL: ["point_poi", "boundary_poi"]},
            geometry=[
                Point(50, 50),                    # liegt vollstaendig in Kachel A
                box(80, 30, 120, 70),              # reicht ueber die Grenze bei x=100 in beide Kacheln
            ],
            crs=crs,
        )

        pairs = join_pois_to_tiles(gdf, [tile_a, tile_b])

        assert "tile" in pairs.columns, "Spalte 'tile' fehlt"
        assert set(pairs.loc[pairs[ID_COL] == "point_poi", "tile"]) == {"tile_a.tif"}
        boundary_tiles = set(pairs.loc[pairs[ID_COL] == "boundary_poi", "tile"])
        assert boundary_tiles == {"tile_a.tif", "tile_b.tif"}, \
            f"boundary_poi sollte Pixel aus BEIDEN Kacheln haben, bekam: {boundary_tiles}"

        n_pix_boundary = pairs.loc[pairs[ID_COL] == "boundary_poi", "n_pixels_poi"].iloc[0]
        n_rows_boundary = (pairs[ID_COL] == "boundary_poi").sum()
        assert n_pix_boundary == n_rows_boundary, \
            f"n_pixels_poi ({n_pix_boundary}) sollte ALLE Kacheln zusammenzaehlen ({n_rows_boundary} Zeilen)"

        emb_dim = len(pairs["embedding"].iloc[0])
        assert emb_dim == n_bands

    print(f"Smoke-Test OK: point_poi nur in Kachel A, boundary_poi in beiden Kacheln "
          f"({n_rows_boundary} Zeilen, n_pixels_poi={n_pix_boundary} korrekt kachelübergreifend "
          f"summiert), Embedding-Dim={emb_dim}.")


def _smoke_test_utm_zone():
    """Regressionstest fuer die UTM-Zonen-Deduplizierung (Nutzervorgabe nach dem 2.32%-
    Ueberlappungsbefund): zwei synthetische Kacheln, die BEIDE denselben realen Ausschnitt
    abdecken (eine in EPSG:32630, eine in EPSG:32631 - wie die echte Zonennaht-Ueberlappung),
    plus zwei POIs knapp westlich/oestlich von 0 Grad Laenge. Ohne Zonen-Filter wuerden BEIDE
    POIs Pixel aus BEIDEN Kacheln bekommen (der urspruengliche Bug). Mit Filter darf jeder POI
    nur Pixel aus SEINER zugewiesenen Zone bekommen."""
    import tempfile

    import geopandas as gpd
    from shapely.geometry import Point, box

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Realer Ausschnitt, der 0 Grad Laenge einschliesst - in BEIDE Zonen-CRS reprojiziert,
        # um zwei Kacheln zu bauen, die denselben Boden abdecken (wie die echte Ueberlappung)
        real_bbox = gpd.GeoDataFrame(geometry=[box(-0.1, 51.45, 0.1, 51.55)], crs="EPSG:4326")
        bounds_30630 = real_bbox.to_crs("EPSG:32630").total_bounds
        bounds_31631 = real_bbox.to_crs("EPSG:32631").total_bounds

        n_bands = 2
        rng = np.random.default_rng(0)

        def make_tile(path, crs, bounds, size=20):
            minx, miny, maxx, maxy = bounds
            transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, size, size)
            with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=n_bands,
                                dtype="float32", crs=crs, transform=transform) as dst:
                dst.write(rng.standard_normal((n_bands, size, size)).astype(np.float32))

        tile_30630 = tmp / "tile_30630.tif"
        tile_31631 = tmp / "tile_31631.tif"
        make_tile(tile_30630, "EPSG:32630", bounds_30630)
        make_tile(tile_31631, "EPSG:32631", bounds_31631)

        pois = gpd.GeoDataFrame(
            {ID_COL: ["poi_west", "poi_east"]},
            geometry=[Point(-0.05, 51.5), Point(0.05, 51.5)],  # knapp westlich/oestlich 0 Grad
            crs="EPSG:4326",
        )

        # Kontrolle, dass die Zonen-Zuordnung selbst stimmt, bevor der Join geprueft wird
        zones = assign_utm_zone(pois)
        assert zones.iloc[0] == "30N" and zones.iloc[1] == "31N", \
            f"Zonen-Zuordnung falsch: {zones.tolist()}"

        pairs = join_pois_to_tiles(pois, [tile_30630, tile_31631])

        assert "utm_zone" in pairs.columns, "Spalte 'utm_zone' fehlt"

        west_tiles = set(pairs.loc[pairs[ID_COL] == "poi_west", "tile"])
        east_tiles = set(pairs.loc[pairs[ID_COL] == "poi_east", "tile"])
        assert west_tiles == {"tile_30630.tif"}, \
            f"poi_west sollte NUR Pixel aus der 30N-Kachel bekommen, bekam: {west_tiles}"
        assert east_tiles == {"tile_31631.tif"}, \
            f"poi_east sollte NUR Pixel aus der 31N-Kachel bekommen, bekam: {east_tiles}"

        west_zones = set(pairs.loc[pairs[ID_COL] == "poi_west", "utm_zone"])
        east_zones = set(pairs.loc[pairs[ID_COL] == "poi_east", "utm_zone"])
        assert west_zones == {"30N"} and east_zones == {"31N"}, \
            f"utm_zone-Spalte falsch: west={west_zones}, east={east_zones}"

        # Kernkriterium (Nutzervorgabe): kein POI darf nach dem Fix noch in Kacheln beider
        # Zonen auftauchen
        both_zones = pairs.groupby(ID_COL)["utm_zone"].nunique()
        n_both = int((both_zones > 1).sum())
        assert n_both == 0, f"{n_both} POIs immer noch in mehr als einer Zone (sollte 0 sein)"

    print("Smoke-Test OK: poi_west nur aus 30N-Kachel, poi_east nur aus 31N-Kachel (obwohl "
          "beide Kacheln denselben realen Ausschnitt abdecken) - 0 POIs in mehr als einer Zone.")


def _smoke_test_north_up():
    """Regressionstest fuer den bottom-up-Bug (reale AlphaEarth-Rohkacheln haben transform.e > 0,
    exact_extract bricht darauf mit 'Incompatible extents' ab, siehe _ensure_north_up-Docstring).
    Baut eine synthetische bottom-up-Kachel, prueft dass _ensure_north_up sie korrekt spiegelt
    (Werte an derselben Weltkoordinate bleiben identisch, nur Zeilenreihenfolge/Vorzeichen
    aendern sich) UND dass exact_extract danach tatsaechlich funktioniert (vorher: Absturz)."""
    import tempfile

    from shapely.geometry import box

    crs = "EPSG:32630"
    n_bands = 2
    h = w = 20

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bottom_up_path = tmp / "bottom_up.tif"

        # positive y-Aufloesung = bottom-up, wie die echten AlphaEarth-GCS-Kacheln
        transform = Affine(10.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((n_bands, h, w)).astype("float32")
        with rasterio.open(bottom_up_path, "w", driver="GTiff", height=h, width=w, count=n_bands,
                            dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(data)

        with rasterio.open(bottom_up_path) as src:
            assert src.transform.e > 0, "Testkachel sollte bottom-up sein (Vorbedingung)"
            assert src.bounds.bottom > src.bounds.top, "bounds sollten invertiert sein (bottom-up)"

        cache_dir = tmp / "northup_cache"
        fixed_path = _ensure_north_up(bottom_up_path, cache_dir)
        assert fixed_path != bottom_up_path, "sollte eine neue, gespiegelte Datei zurueckgeben"

        with rasterio.open(fixed_path) as src:
            assert src.transform.e < 0, "Ergebnis sollte north-up sein"
            assert src.bounds.bottom < src.bounds.top, "bounds sollten jetzt normal sein"
            fixed_transform = src.transform
            fixed_data = src.read()

        # Werte an dem WELTKOORDINATEN-Punkt (5,5) muessen in beiden Versionen identisch sein -
        # das Spiegeln darf keine Werte veraendern, nur die interne Zeilen-/Vorzeichenordnung
        with rasterio.open(bottom_up_path) as src:
            row_orig, col_orig = rasterio.transform.rowcol(src.transform, 5, 5)
            val_orig = src.read()[:, row_orig, col_orig]
        row_fixed, col_fixed = rasterio.transform.rowcol(fixed_transform, 5, 5)
        val_fixed = fixed_data[:, row_fixed, col_fixed]
        assert np.allclose(val_orig, val_fixed), \
            f"Wert an Weltkoordinate (5,5) hat sich durchs Spiegeln veraendert: {val_orig} vs {val_fixed}"

        # exact_extract muss gegen die gespiegelte Kachel funktionieren (vorher: Absturz)
        poly = gpd.GeoDataFrame({"id": [1]}, geometry=[box(50, 50, 80, 80)], crs=crs)
        res = exact_extract(str(fixed_path), poly, ["cell_id", "coverage"],
                             include_cols=["id"], output="pandas")
        assert len(res) == 1

        # Bereits north-up (wie TESSERA) -> unveraendert zurueckgeben, kein Cache-Schreiben
        already_ok_path = tmp / "already_northup.tif"
        with rasterio.open(already_ok_path, "w", driver="GTiff", height=h, width=w, count=n_bands,
                            dtype="float32", crs=crs,
                            transform=rasterio.transform.from_origin(0, 200, 10, 10)) as dst:
            dst.write(data)
        result = _ensure_north_up(already_ok_path, cache_dir)
        assert result == already_ok_path, "bereits north-up sollte unveraendert zurueckgegeben werden"

    print("Smoke-Test OK: bottom-up-Kachel korrekt gespiegelt (Werte unveraendert, "
          "exact_extract funktioniert danach), bereits-north-up-Kachel unveraendert durchgereicht.")


def _smoke_test_empty_geom_type():
    """Regressionstest fuer den dtype-Bug (Kachel 46/50 im TESSERA-Benchmark): eine Kachel mit
    NUR Punkt- oder NUR Polygon-Treffern (die jeweils andere Geometrieart liefert 0 Treffer)
    darf nicht mit 'arrays used as indices must be of integer (or boolean) type' abstuerzen -
    das leere pd.DataFrame(columns=[...]) aus _join_polygons() hatte sonst object-dtype fuer
    row/col, was pd.concat() mit dem nicht-leeren Punkte-Ergebnis auf object hochcastete."""
    import tempfile

    from shapely.geometry import Point, box

    crs = "EPSG:3857"
    n_bands = 2

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        transform = rasterio.transform.from_origin(west=0, north=100, xsize=10, ysize=10)
        tile_path = tmp / "tile.tif"
        rng = np.random.default_rng(0)
        with rasterio.open(tile_path, "w", driver="GTiff", height=10, width=10, count=n_bands,
                            dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(rng.standard_normal((n_bands, 10, 10)).astype(np.float32))

        # Fall 1: nur ein Punkt trifft, keine Polygone (der urspruengliche Absturzfall)
        points_only = gpd.GeoDataFrame({ID_COL: ["p0"]}, geometry=[Point(50, 50)], crs=crs)
        pairs = join_pois_to_raster(points_only, tile_path)
        assert len(pairs) == 1 and pairs["row"].dtype.kind in "iu"
        assert len(pairs["embedding"].iloc[0]) == n_bands

        # Fall 2: nur ein Polygon trifft, keine Punkte (umgekehrter Fall)
        polys_only = gpd.GeoDataFrame({ID_COL: ["p1"]}, geometry=[box(20, 20, 40, 40)], crs=crs)
        pairs = join_pois_to_raster(polys_only, tile_path)
        assert len(pairs) > 0 and pairs["row"].dtype.kind in "iu"

        # Fall 3: gemischt in einem Aufruf (Punkt UND Polygon gleichzeitig, wie im echten Lauf)
        mixed = gpd.GeoDataFrame(
            {ID_COL: ["p0", "p1"]},
            geometry=[Point(50, 50), box(20, 20, 40, 40)],
            crs=crs,
        )
        pairs = join_pois_to_raster(mixed, tile_path)
        assert pairs["row"].dtype.kind in "iu" and pairs["col"].dtype.kind in "iu"
        # das war exakt der urspruengliche Crash: fancy-indexing mit object-dtype row/col
        arr_test = np.zeros((n_bands, 10, 10))
        _ = arr_test[:, pairs["row"].to_numpy(), pairs["col"].to_numpy()]

    print("Smoke-Test OK: Punkt-only, Polygon-only und gemischte Treffer liefern durchgehend "
          "int-dtype row/col - kein object-dtype-Absturz mehr bei leeren Teil-Ergebnissen.")


def _smoke_test_by_tile_equivalence():
    """join_pois_to_tiles_by_tile() (Kachel-Schleife aussen, fuer TESSERA/viele-Kacheln
    gebaut) muss EXAKT dieselbe Paarmenge liefern wie join_pois_to_tiles() (POI-Schleife
    aussen, Referenzimplementierung) - nur die Durchlaufreihenfolge unterscheidet sich. Nutzt
    dasselbe Zwei-Kacheln-Setup wie _smoke_test() (Grenz-POI ueber beide Kacheln, Punkt-POI
    nur in einer)."""
    import tempfile

    from shapely.geometry import Point, box

    crs = "EPSG:3857"
    n_bands = 3

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tile_a = tmp / "tile_a.tif"
        tile_b = tmp / "tile_b.tif"

        transform_a = rasterio.transform.from_origin(west=0, north=100, xsize=10, ysize=10)
        transform_b = rasterio.transform.from_origin(west=100, north=100, xsize=10, ysize=10)
        rng = np.random.default_rng(0)
        for path, transform in [(tile_a, transform_a), (tile_b, transform_b)]:
            with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=n_bands,
                                dtype="float32", crs=crs, transform=transform) as dst:
                dst.write(rng.standard_normal((n_bands, 10, 10)).astype(np.float32))

        gdf = gpd.GeoDataFrame(
            {ID_COL: ["point_poi", "boundary_poi", "point_poi_2"]},
            geometry=[
                Point(50, 50),
                box(80, 30, 120, 70),
                Point(150, 60),
            ],
            crs=crs,
        )

        chunk_dir = tmp / "chunks"
        pairs_ref = join_pois_to_tiles(gdf, [tile_a, tile_b])
        pairs_new = join_pois_to_tiles_by_tile(gdf, [tile_a, tile_b], chunk_dir=chunk_dir)

        cols = [ID_COL, "row", "col", "coverage", "tile"]
        ref_sorted = pairs_ref[cols].sort_values(cols).reset_index(drop=True)
        new_sorted = pairs_new[cols].sort_values(cols).reset_index(drop=True)
        assert ref_sorted.equals(new_sorted), (
            f"join_pois_to_tiles_by_tile() weicht von join_pois_to_tiles() ab:\n"
            f"ref:\n{ref_sorted}\nnew:\n{new_sorted}")

        ref_emb = np.stack(pairs_ref.sort_values(cols)["embedding"].to_numpy())
        new_emb = np.stack(pairs_new.sort_values(cols)["embedding"].to_numpy())
        assert np.allclose(ref_emb, new_emb), "Embedding-Werte weichen zwischen beiden Implementierungen ab"

        # Resume-Pfad: zweiter Aufruf mit denselben (jetzt existierenden) Chunk-Dateien muss
        # identisches Ergebnis liefern, ohne die Kacheln neu zu lesen
        pairs_resumed = join_pois_to_tiles_by_tile(gdf, [tile_a, tile_b], chunk_dir=chunk_dir)
        resumed_sorted = pairs_resumed[cols].sort_values(cols).reset_index(drop=True)
        assert ref_sorted.equals(resumed_sorted), "Resume-Lauf weicht vom Erstlauf ab"

    print("Smoke-Test OK: join_pois_to_tiles_by_tile() liefert identische Paare/Embeddings "
          "wie join_pois_to_tiles() (inkl. Resume ueber bestehende Chunk-Dateien).")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
        _smoke_test_utm_zone()
        _smoke_test_north_up()
        _smoke_test_empty_geom_type()
        _smoke_test_by_tile_equivalence()
    else:
        main()
