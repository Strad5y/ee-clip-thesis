"""
build_splits_pixels.py

Nachtrag zu Phase 5: splits_london.csv wurde von build_splits.py auf den ALTEN 279.988 POIs
(4-Key-Extraktion) gebaut - Zellen ohne eine dieser POIs fehlen im Mapping, 3,56%/3,43% der
Pixel aus der 13-Key-Extraktion bekamen deshalb split=NaN (siehe aggregate_pixel_pairs.py-
Bericht). Dieses Skript baut die Split-Zellen stattdessen direkt aus dem PIXELBESTAND beider
Backbones (pixels_tessera.parquet, pixels_ae.parquet) - Vereinigung der von PIXELN (nicht
POIs) beruehrten Zellen, damit jedes tatsaechlich existierende Pixel eine Zelle hat.

Beide Pixelmengen werden GEMEINSAM als Grundlage genommen (nicht getrennt), damit exakt
dieselbe Zelle in TESSERA und AlphaEarth denselben Split bekommt - sonst waeren die beiden
Backbones nicht direkt vergleichbar (unterschiedliche Kachelgeometrie/Aufloesung fuehrt sonst
zu leicht unterschiedlichen Pixelrastern, aber die Zellzuordnung (1 km, POI-/Pixel-unabhaengig)
ist backbone-agnostisch).

Verfahren unveraendert ggue. build_splits.py: 1-km-Gitter (CELL_SIZE_M), Zellen (nicht
einzelne Pixel/POIs) werden zufaellig auf train/val/test verteilt (80/10/10, SEED), Pixel
erben den Split ihrer Zelle. assign_splits() wird von dort direkt importiert (gleiche Logik,
keine Duplizierung).

Ergebnis ersetzt data/splits/splits_london.csv (neues Schema: cell_row,cell_col,split - kein
poi_id mehr, die Zelle ist jetzt aus Pixeln abgeleitet, nicht aus POI-Geometrien). Die alte
Datei wird VORHER nach splits_london_4key.csv gesichert, nicht ueberschrieben.
"""
import shutil
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd

import config as cfg
from build_splits import CELL_SIZE_M, SEED, SPLIT_RATIOS, assign_splits
from spatial_join import BACKBONES
from aggregate_pixel_pairs import compute_pixel_cells

AOI_PATH = cfg.AOI_DIR / "london.gpkg"
SPLITS_OUT = cfg.DATA / "splits" / "splits_london.csv"
SPLITS_BACKUP = cfg.DATA / "splits" / "splits_london_4key.csv"
PIXELS_PATHS = {name: cfg.DATA / "processed" / f"pixels_{name}.parquet" for name in BACKBONES}


def get_aoi_utm_reference():
    aoi = gpd.read_file(AOI_PATH)
    utm_crs = aoi.estimate_utm_crs()
    aoi_utm = aoi.to_crs(utm_crs)
    minx, miny, _, _ = aoi_utm.total_bounds
    return utm_crs, minx, miny


def main():
    utm_crs, minx, miny = get_aoi_utm_reference()
    print(f"AOI-UTM-CRS: {utm_crs}, minx={minx:.1f}, miny={miny:.1f}, Zellgroesse={CELL_SIZE_M:.0f}m")

    # Zellen aus BEIDEN Pixelbestaenden vereinigen
    cells_per_backbone = {}
    all_cells = []
    for name, path in PIXELS_PATHS.items():
        px = pd.read_parquet(path, columns=["tile", "row", "col"])
        cells = compute_pixel_cells(px, BACKBONES[name], utm_crs, minx, miny)
        cells_per_backbone[name] = cells
        all_cells.append(cells[["cell_row", "cell_col"]])
        print(f"  {name}: {len(px)} Pixel -> {cells[['cell_row','cell_col']].drop_duplicates().shape[0]} distinkte Zellen")

    union_cells = pd.concat(all_cells, ignore_index=True).drop_duplicates().reset_index(drop=True)
    union_cells["cell_id"] = union_cells["cell_row"].astype(str) + "_" + union_cells["cell_col"].astype(str)
    n_cells = len(union_cells)
    print(f"\nVereinigung beider Backbones: {n_cells} distinkte Zellen")

    # Split je Zelle (identische Logik wie build_splits.py, importiert - nicht dupliziert)
    cell_to_split = assign_splits(union_cells["cell_id"].to_numpy(), SPLIT_RATIOS, SEED)
    union_cells["split"] = union_cells["cell_id"].map(cell_to_split)

    print(f"\nSeed: {SEED}")
    print("Zellen je Split:")
    print(union_cells["split"].value_counts())

    # Leakage-Check: jede Zelle darf nur EINEN Split haben (hier trivial per Konstruktion,
    # aber explizit verifiziert statt nur angenommen - gleiche Disziplin wie build_splits.py).
    n_conflict = int((union_cells.groupby(["cell_row", "cell_col"])["split"].nunique() > 1).sum())
    print(f"\nLeakage-Verifikation (Zellen mit >1 Split, sollte 0 sein): {n_conflict}")
    assert n_conflict == 0, f"{n_conflict} Zellen mit widerspruechlichem Split - Bug."

    # Alte Datei sichern, NICHT ueberschreiben
    if SPLITS_OUT.exists() and not SPLITS_BACKUP.exists():
        shutil.copy2(SPLITS_OUT, SPLITS_BACKUP)
        print(f"\nAlte Datei gesichert: {SPLITS_BACKUP}")
    elif SPLITS_BACKUP.exists():
        print(f"\nBackup existiert bereits, nicht erneut ueberschrieben: {SPLITS_BACKUP}")

    union_cells[["cell_row", "cell_col", "split"]].to_csv(SPLITS_OUT, index=False)
    print(f"Neue Datei gespeichert: {SPLITS_OUT}")

    # Pixel-Dateien mit dem NEUEN Split versehen (ersetzt die alte split-Spalte, die zu 3,5%
    # NaN war) - in-place aktualisiert, kein neuer Dateiname (direkte Korrektur von Phase 5).
    cell_to_split_tuple = dict(zip(zip(union_cells["cell_row"], union_cells["cell_col"]),
                                    union_cells["split"]))

    print("\n=== Pixel je Split (nach Neuzuordnung) ===")
    all_pixel_ids_by_split = {}
    for name, path in PIXELS_PATHS.items():
        df = pd.read_parquet(path)
        cells = cells_per_backbone[name]
        px_with_cell = df[["tile", "row", "col"]].merge(cells, on=["tile", "row", "col"], how="left")
        new_split = [cell_to_split_tuple.get((r, c)) for r, c in
                     zip(px_with_cell["cell_row"], px_with_cell["cell_col"])]
        df["split"] = new_split

        n_nan = int(pd.isna(df["split"]).sum())
        print(f"\n{name}:")
        print(df["split"].value_counts(dropna=False))
        print(f"  Pixel mit split=NaN (muss 0 sein): {n_nan}")
        assert n_nan == 0, f"{name}: {n_nan} Pixel ohne Split nach Neuzuordnung - Bug."

        df.to_parquet(path)
        print(f"  Aktualisiert (in-place): {path}")

        # Pixel-Leakage-Check: (tile,row,col) darf nur in genau einem Split-Wert vorkommen
        # (strukturell garantiert, da split ausschliesslich von cell_row/cell_col abhaengt -
        # trotzdem explizit gegen die tatsaechlich gespeicherten Daten geprueft).
        dup_split = df.groupby(["tile", "row", "col"])["split"].nunique()
        n_px_conflict = int((dup_split > 1).sum())
        assert n_px_conflict == 0, f"{name}: {n_px_conflict} Pixel mit >1 Split - Bug."

    print(f"\nAnzahl Zellen gesamt: {n_cells}")
    print(f"Verwendeter Seed: {SEED}")


if __name__ == "__main__":
    main()
