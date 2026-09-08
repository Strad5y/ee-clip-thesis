"""
aggregate_pixel_pairs.py

Phase 5 (CLAUDE_CODE_TASKS_phase4_7.md): aus den POI-Pixel-Paaren (pairs_{backbone}_full.
parquet, mehrere POIs koennen dasselbe Pixel schneiden) wird GENAU EIN Paar pro Pixel.

Pixel-Schluessel (tile, row, col) - utm_zone ist redundant (Phase 0, empirisch bestaetigt:
jede tile mappt auf genau eine Zone).

Sortierung innerhalb eines Pixels: coverage absteigend, bei Gleichstand poi_id aufsteigend -
deterministisch. coverage ist AUSSCHLIESSLICH Sortierkriterium (kein Gewicht, kein Ausschluss).

Textkomposition:
  n=1: der POI-Text (poi_text_full.parquet::text) unveraendert.
  n>1: "A location containing {t1}; {t2}; {t3}." - t1 = voller Text des Top-Coverage-POI
       (inkl. Name), t2..tn = 'body' (OHNE Namensprefix, siehe build_poi_text.py) der
       uebrigen POIs. Kein Cap - alle POIs des Pixels gehen ein.

Split: Pixel folgen ihrer raeumlichen Gitterzelle (data/splits/splits_london.csv wurde von
build_splits.py auf 1km-UTM-Zellen gebaut, POIs erben den Split ihrer Zelle - hier wird
dieselbe Zellzuordnung fuer die PIXEL-Position nachvollzogen, nicht die POI-Zeile aus der
CSV direkt uebernommen, da splits_london.csv nur die alten 279.988 POIs abdeckt, nicht die
1,65 Mio der 13-Key-Extraktion). cell_row/cell_col -> split wird aus der CSV rekonstruiert
(deterministisch, 1 Split pro Zelle, verifiziert). Pixel, deren Zelle in der CSV nicht
vorkommt (keine der alten POIs lag dort), bekommen split=NaN - gezaehlt und gemeldet, nicht
erfunden.

Effizienz: die weit ueberwiegende Mehrheit der Pixel hat genau 1 POI (siehe Phase-0/3-
Dichtemessung) - fuer diese ist die Textkomposition ein reiner Merge (kein Python-Loop).
Nur die Minderheit mit >1 POI durchlaeuft eine gruppierte Aggregation.
"""
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))   # damit config gefunden wird

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import config as cfg
from spatial_join import BACKBONES

PAIRS_DIR = cfg.DATA / "processed"
POI_TEXT_PATH = cfg.DATA / "interim" / "poi_text_full.parquet"
SPLITS_PATH = cfg.DATA / "splits" / "splits_london.csv"
AOI_PATH = cfg.AOI_DIR / "london.gpkg"
OUT_DIR = cfg.DATA / "processed"

CELL_SIZE_M = 1000.0  # muss zu build_splits.py passen


def build_cell_to_split() -> tuple[dict, object, tuple]:
    """Rekonstruiert (cell_row,cell_col)->split aus splits_london.csv (1 Split je Zelle,
    verifiziert) UND liefert dieselbe AOI-UTM-CRS + minx/miny, die build_splits.py fuer die
    Zellzuordnung benutzt hat (fuer die Pixel-Position muss dieselbe Referenz gelten)."""
    sp = pd.read_csv(SPLITS_PATH)
    n_multi = int((sp.groupby(["cell_row", "cell_col"])["split"].nunique() > 1).sum())
    if n_multi > 0:
        raise RuntimeError(f"{n_multi} Zellen mit widerspruechlichem Split in {SPLITS_PATH} - "
                            f"Annahme (1 Split je Zelle) verletzt.")
    cell_to_split = sp.drop_duplicates(["cell_row", "cell_col"]).set_index(
        ["cell_row", "cell_col"])["split"].to_dict()

    aoi = gpd.read_file(AOI_PATH)
    utm_crs = aoi.estimate_utm_crs()
    aoi_utm = aoi.to_crs(utm_crs)
    minx, miny, _, _ = aoi_utm.total_bounds
    return cell_to_split, utm_crs, (minx, miny)


def compute_pixel_cells(unique_px: pd.DataFrame, tile_paths: list,
                         utm_crs, minx: float, miny: float) -> pd.DataFrame:
    """Fuer jede distinkte (tile,row,col)-Kombination: Pixelmittelpunkt via Affine-Transform
    der Kachel (kein rasterio-Aufruf pro Zeile - vektorisiert ueber die Affine-Koeffizienten),
    reprojiziert auf die AOI-UTM-CRS (pyproj.Transformer, vektorisiert), cell_row/cell_col per
    Ganzzahl-Division WIE build_splits.py. Reine Geometrie, KEIN Split-Lookup - wird sowohl
    hier (Split fuer bestehende Pixel nachschlagen) als auch in build_splits_pixels.py
    (Split-Zellen aus dem Pixelbestand neu bauen) verwendet, damit beide Stellen garantiert
    dieselbe Zellzuordnung berechnen."""
    tile_by_name = {p.name: p for p in tile_paths}
    unique_px = unique_px[["tile", "row", "col"]].drop_duplicates().reset_index(drop=True)
    cell_row_out = pd.Series(index=unique_px.index, dtype="Int64")
    cell_col_out = pd.Series(index=unique_px.index, dtype="Int64")

    for tile_name, idx in unique_px.groupby("tile").groups.items():
        sub = unique_px.loc[idx]
        tile_path = tile_by_name.get(tile_name)
        if tile_path is None:
            continue
        with rasterio.open(tile_path) as src:
            transform = src.transform
            tile_crs = src.crs

        rows = sub["row"].to_numpy()
        cols = sub["col"].to_numpy()
        # Pixelmittelpunkt: +0.5 (Zeilen-/Spaltenmitte), Affine-Formel manuell statt
        # rasterio.transform.xy() pro Zeile - identisches Ergebnis, vektorisiert.
        a, b, c, d, e, f = (transform.a, transform.b, transform.c,
                            transform.d, transform.e, transform.f)
        xs = a * (cols + 0.5) + b * (rows + 0.5) + c
        ys = d * (cols + 0.5) + e * (rows + 0.5) + f

        if str(tile_crs) != str(utm_crs):
            transformer = Transformer.from_crs(tile_crs, utm_crs, always_xy=True)
            xs, ys = transformer.transform(xs, ys)

        cell_col_out.loc[idx] = np.floor((xs - minx) / CELL_SIZE_M).astype(int)
        cell_row_out.loc[idx] = np.floor((ys - miny) / CELL_SIZE_M).astype(int)

    unique_px["cell_row"] = cell_row_out
    unique_px["cell_col"] = cell_col_out
    return unique_px


def assign_splits_to_pixels(df: pd.DataFrame, tile_paths: list, cell_to_split: dict,
                             utm_crs, minx: float, miny: float) -> pd.Series:
    """Split-Lookup fuer bestehende Pixel: Zelle berechnen (compute_pixel_cells), dann gegen
    cell_to_split nachschlagen. Pixel ohne Treffer (Zelle nicht im Mapping) -> NaN."""
    unique_px = compute_pixel_cells(df, tile_paths, utm_crs, minx, miny)
    unique_px["split"] = [cell_to_split.get((r, c)) for r, c in
                           zip(unique_px["cell_row"], unique_px["cell_col"])]
    return df.merge(unique_px[["tile", "row", "col", "split"]],
                     on=["tile", "row", "col"], how="left")["split"]


def _strip_trailing_period(s: str) -> str:
    return s[:-1] if s.endswith(".") else s


def compose_pixel_texts_multi(group: pd.DataFrame) -> pd.Series:
    """group ist BEREITS sortiert (coverage desc, poi_id asc) und nach (tile,row,col)
    gruppiert. Erstes POI je Gruppe: voller Text (inkl. Name). Weitere: nur body (Frame+
    Value+Beschreibung, ohne Name). Kein Trailing-Semikolon, genau ein Punkt am Ende."""
    def _compose(sub: pd.DataFrame) -> str:
        pieces = [_strip_trailing_period(sub["text"].iloc[0])]
        pieces += [_strip_trailing_period(b) for b in sub["body"].iloc[1:]]
        return "A location containing " + "; ".join(pieces) + "."

    return group.groupby(["tile", "row", "col"], sort=False).apply(_compose, include_groups=False)


def aggregate(label: str, tile_paths: list, pairs_path: Path, poi_text: pd.DataFrame,
              cell_to_split: dict, utm_crs, minx: float, miny: float) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    pairs = pd.read_parquet(pairs_path)
    n_pairs_raw = len(pairs)
    print(f"  {pairs_path.name}: {n_pairs_raw} Zeilen gelesen ({time.time()-t0:.0f}s)")

    merged = pairs.merge(poi_text[["poi_id", "text", "body"]], on="poi_id", how="inner")
    n_pairs_with_text = len(merged)
    print(f"  Nach Inner-Join mit poi_text_full (Fall D faellt raus): {n_pairs_with_text} Zeilen "
          f"({time.time()-t0:.0f}s)")

    # EO-Vektor-Konsistenz: JEDES Pixel muss ueber alle beteiligten POIs denselben Vektor
    # tragen (reiner Funktions eines Rasterzellen-Lookups, unabhaengig vom POI). Vektorisiert:
    # Embeddings als 2D-Array stapeln, gegen den ERSTEN Wert je Gruppe vergleichen.
    merged = merged.sort_values(["tile", "row", "col"]).reset_index(drop=True)
    emb = np.stack(merged["embedding"].to_numpy())
    same_key = ((merged["tile"] == merged["tile"].shift()) &
                (merged["row"] == merged["row"].shift()) &
                (merged["col"] == merged["col"].shift())).to_numpy()
    mismatch = same_key & ~(emb == np.roll(emb, 1, axis=0)).all(axis=1)
    n_mismatch = int(mismatch.sum())
    if n_mismatch > 0:
        example_idx = np.where(mismatch)[0][0]
        raise RuntimeError(
            f"{n_mismatch} Pixel mit UNTERSCHIEDLICHEM EO-Vektor ueber ihre POIs - das ist ein "
            f"Bug (Rasterzellen-Lookup sollte poi-unabhaengig sein), kein Fall zum Mitteln. "
            f"Beispiel-Index: {example_idx} ({merged.iloc[example_idx][['tile','row','col']].to_dict()})")
    print(f"  EO-Vektor-Konsistenz verifiziert: 0 Abweichungen ueber "
          f"{merged[['tile','row','col']].drop_duplicates().shape[0]} Pixel ({time.time()-t0:.0f}s)")

    # Sortierung: coverage absteigend, bei Gleichstand poi_id aufsteigend - EIN globaler Sort
    # statt pro Gruppe (viel schneller), Gruppen-Reihenfolge bleibt danach stabil erhalten.
    merged = merged.sort_values(["tile", "row", "col", "coverage", "poi_id"],
                                 ascending=[True, True, True, False, True]).reset_index(drop=True)

    n_pois_per_pixel = merged.groupby(["tile", "row", "col"], sort=False)["poi_id"].transform("size")
    merged["n_pois"] = n_pois_per_pixel

    single = merged[merged["n_pois"] == 1]
    multi = merged[merged["n_pois"] > 1]
    print(f"  Pixel mit 1 POI: {len(single)}, mit >1 POI: "
          f"{multi[['tile','row','col']].drop_duplicates().shape[0]} "
          f"({multi.shape[0]} Zeilen) ({time.time()-t0:.0f}s)")

    # n=1: reiner Merge, kein Loop.
    single_out = single[["tile", "row", "col", "embedding", "coverage", "n_pixels_poi", "poi_id"]].copy()
    single_out["pixel_text"] = single["text"].to_numpy()
    single_out["poi_ids"] = single["poi_id"].astype(str)

    # n>1: gruppierte Aggregation, nur ueber die Minderheit.
    if len(multi) > 0:
        pixel_texts_multi = compose_pixel_texts_multi(multi)
        grp = multi.groupby(["tile", "row", "col"], sort=False)
        multi_out = grp.agg(
            embedding=("embedding", "first"),
            coverage=("coverage", "first"),      # Top-Coverage-POI (nach dem Sort oben)
            n_pixels_poi=("n_pixels_poi", "first"),
            n_pois=("n_pois", "first"),           # bereits per transform("size") gesetzt, konstant je Gruppe
            poi_ids=("poi_id", lambda s: ",".join(s.astype(str))),
        ).reset_index()
        multi_out = multi_out.merge(
            pixel_texts_multi.rename("pixel_text").reset_index(), on=["tile", "row", "col"])
        print(f"  Multi-POI-Aggregation fertig ({time.time()-t0:.0f}s)")
    else:
        multi_out = pd.DataFrame(columns=single_out.columns.tolist() + ["pixel_text", "n_pois"])

    single_out["n_pois"] = 1

    out = pd.concat([
        single_out[["tile", "row", "col", "embedding", "pixel_text", "coverage", "n_pixels_poi", "n_pois", "poi_ids"]],
        multi_out[["tile", "row", "col", "embedding", "pixel_text", "coverage", "n_pixels_poi", "n_pois", "poi_ids"]],
    ], ignore_index=True)
    out["weight"] = 1.0

    # Split zuordnen
    out["split"] = assign_splits_to_pixels(out, tile_paths, cell_to_split, utm_crs, minx, miny)
    print(f"  Split zugeordnet ({time.time()-t0:.0f}s)")

    stats = {
        "n_pairs_raw": n_pairs_raw,
        "n_pairs_with_text": n_pairs_with_text,
        "n_pixels": len(out),
    }
    return out, stats


def validate(out: pd.DataFrame, label: str) -> dict:
    dup = out.duplicated(subset=["tile", "row", "col"]).sum()
    assert dup == 0, f"{label}: {dup} doppelte Pixel-Schluessel (tile,row,col) - Bug."

    n_empty = int((out["pixel_text"].isna() | (out["pixel_text"].str.len() == 0)).sum())
    assert n_empty == 0, f"{label}: {n_empty} leere pixel_text - Bug."

    n_no_split = int(out["split"].isna().sum())
    print(f"  Validierung OK: Pixel-Schluessel eindeutig, kein leerer pixel_text.")
    print(f"  Pixel ohne Split-Zelle (Ausnahme, GEMESSEN): {n_no_split} "
          f"({100*n_no_split/len(out):.2f}% von {len(out)})")
    return {"n_no_split": n_no_split}


def report(out: pd.DataFrame, stats: dict, label: str):
    n_pixels = len(out)
    print(f"\n=== {label}: Bericht ===")
    print(f"Zeilen vorher (POI-Pixel-Paare, GEMESSEN): {stats['n_pairs_raw']}")
    print(f"  davon mit Text (Fall A/B/C, GEMESSEN): {stats['n_pairs_with_text']}")
    print(f"Zeilen nachher (Pixel, GEMESSEN): {n_pixels}")

    print(f"\nVerteilung n_pois (GEMESSEN): median={out['n_pois'].median()}, "
          f"mean={out['n_pois'].mean():.4f}, p90={out['n_pois'].quantile(0.9)}, "
          f"p99={out['n_pois'].quantile(0.99)}, max={out['n_pois'].max()}")
    frac_single = (out["n_pois"] == 1).mean()
    print(f"Anteil Pixel mit genau 1 POI (GEMESSEN): {100*frac_single:.2f}%")

    n_unique = out["pixel_text"].nunique()
    print(f"\n*** Anteil eindeutiger pixel_text an allen Pixeln (GEMESSEN, DIE Zahl): "
          f"{n_unique} von {n_pixels} = {100*n_unique/n_pixels:.2f}% ***")

    top10 = out["pixel_text"].value_counts().head(10)
    print("\nTop-10 pixel_text (GEMESSEN):")
    for text, n in top10.items():
        print(f"  {100*n/n_pixels:5.2f}%  n={n:>8}  {text[:100]}")

    lens = out["pixel_text"].str.len()
    print(f"\nZeichenlaenge pixel_text (GEMESSEN): median={lens.median()}, "
          f"p99={lens.quantile(0.99)}, max={lens.max()}")
    longest = out.loc[lens.idxmax(), "pixel_text"]
    print(f"\nLaengster Text (GEMESSEN, {len(longest)} Zeichen):\n  {longest}")

    print("\nZeilen je Split (GEMESSEN, inkl. NaN = keine Split-Zelle):")
    print(out["split"].value_counts(dropna=False))

    print("\nBeispieltexte:")
    one = out[out["n_pois"] == 1]
    three = out[out["n_pois"] == 3]
    densest = out.loc[out["n_pois"].idxmax()]
    if len(one):
        print(f"  [1 POI] {one['pixel_text'].iloc[0]}")
    if len(three):
        print(f"  [3 POIs] {three['pixel_text'].iloc[0]}")
    print(f"  [dichtestes Pixel, n_pois={densest['n_pois']}] {densest['pixel_text']}")


def main():
    print("Lade Split-Zellen-Referenz...")
    cell_to_split, utm_crs, (minx, miny) = build_cell_to_split()
    print(f"  {len(cell_to_split)} Zellen mit bekanntem Split, UTM-CRS={utm_crs}, "
          f"minx={minx:.1f} miny={miny:.1f}")

    poi_text = pd.read_parquet(POI_TEXT_PATH, columns=["poi_id", "text", "body"])
    print(f"poi_text_full.parquet: {len(poi_text)} POIs geladen")

    for label, tile_paths in BACKBONES.items():
        print(f"\n=== Backbone: {label} ===")
        pairs_path = PAIRS_DIR / f"pairs_{label}_full.parquet"
        if not pairs_path.exists():
            print(f"  {pairs_path} fehlt - uebersprungen")
            continue

        out, stats = aggregate(label, tile_paths, pairs_path, poi_text, cell_to_split, utm_crs, minx, miny)
        validate(out, label)
        report(out, stats, label)

        out_path = OUT_DIR / f"pixels_{label}.parquet"
        out[["tile", "row", "col", "embedding", "pixel_text", "weight", "n_pois",
             "coverage", "n_pixels_poi", "poi_ids", "split"]].to_parquet(out_path)
        print(f"\nGespeichert: {out_path}")


if __name__ == "__main__":
    main()
