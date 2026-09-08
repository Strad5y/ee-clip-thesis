"""
verify_provenance.py

Konsistenzpruefung (Nutzervorgabe, 3.9.): vor jedem Trainings- oder Probe-Lauf aufrufbar, prueft
ob die split-Spalte ALLER Dateien, die einen Split tragen, ueber die GEMEINSAMEN Pixelschluessel
(tile,row,col) exakt uebereinstimmt. Das ist der Mechanismus, der die am 3.9. gefundene
Split-Divergenz (landuse_pairs_ae.parquet vs. pixels_ae.parquet, 80,81% Leakage in den
Test-Pixeln durch zwei unabhaengig randomisierte splits_london.csv-Staende) automatisch
gefunden haette - reine Zeitstempel-Betrachtung findet das NICHT, nur der direkte
Kreuztabellen-Vergleich ueber die Schluessel selbst.

Bricht bei einer Abweichung MIT FEHLER ab (kein Warnung-und-weiter, Nutzervorgabe). Fehlt eine
der beiden Dateien eines Paars, wird nur dieses Paar uebersprungen (gemeldet) - nicht jede
registrierte Datei muss zu jedem Zeitpunkt existieren.

Nur Paare INNERHALB desselben Backbones sind hier sinnvoll direkt pruefbar: 'tile' ist der
COG-Dateiname und daher backbone-spezifisch (AlphaEarth- und TESSERA-Kacheln teilen sich keine
Tile-Namen) - AE- und TESSERA-Dateien haben deshalb strukturell KEINE gemeinsamen
(tile,row,col)-Schluessel. Die Cross-Backbone-Konsistenz (dieselbe Zelle -> derselbe Split in
beiden Backbones) ist stattdessen durch die Konstruktion von build_splits_pixels.py garantiert
(EIN gemeinsamer Zell-Split ueber die Vereinigung beider Pixelmengen), nicht Gegenstand dieser
Pruefung.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))   # damit config gefunden wird

import pandas as pd

import config as cfg

KEY_COLS = ["tile", "row", "col"]

# Registrierte Paare (Name A, Pfad A, Name B, Pfad B) - beide Dateien muessen 'split' UND
# (tile,row,col) tragen. Neue Split-tragende Dateien hier ergaenzen, sobald sie entstehen -
# nicht in einem weiteren, separaten Skript.
PAIRS = [
    ("pixels_ae", cfg.DATA / "processed" / "pixels_ae.parquet",
     "landuse_pairs_ae", cfg.DATA / "processed" / "landuse_pairs_ae.parquet"),
    ("pixels_tessera", cfg.DATA / "processed" / "pixels_tessera.parquet",
     "landuse_pairs_tessera", cfg.DATA / "processed" / "landuse_pairs_tessera.parquet"),
]


def _load_keys(path: Path) -> pd.DataFrame:
    """Laedt nur (tile,row,col,split). Prueft VORHER, ob die Datei intern schon widerspruechlich
    ist (derselbe Pixelschluessel mit zwei verschiedenen split-Werten in EINER Datei) - das waere
    ein eigenstaendiger Bug, unabhaengig vom Cross-Datei-Vergleich, und muss genauso hart
    abbrechen."""
    df = pd.read_parquet(path, columns=KEY_COLS + ["split"])
    n_conflict = int(df.groupby(KEY_COLS)["split"].nunique().gt(1).sum())
    if n_conflict > 0:
        raise SystemExit(
            f"FEHLER: {path} hat {n_conflict} Pixelschluessel mit MEHREREN unterschiedlichen "
            f"split-Werten INNERHALB derselben Datei - Datei selbst inkonsistent. Abbruch."
        )
    return df.drop_duplicates(subset=KEY_COLS)


def verify_pair(name_a: str, path_a: Path, name_b: str, path_b: Path) -> None:
    if not path_a.exists() or not path_b.exists():
        missing = name_a if not path_a.exists() else name_b
        print(f"  [{name_a} <-> {name_b}] uebersprungen - {missing} existiert nicht.")
        return

    a = _load_keys(path_a)
    b = _load_keys(path_b)
    merged = a.merge(b, on=KEY_COLS, suffixes=(f"_{name_a}", f"_{name_b}"), how="inner")
    n_overlap = len(merged)
    if n_overlap == 0:
        print(f"  [{name_a} <-> {name_b}] keine gemeinsamen Pixelschluessel - nichts zu pruefen.")
        return

    col_a, col_b = f"split_{name_a}", f"split_{name_b}"
    mismatch = merged[col_a] != merged[col_b]
    n_mismatch = int(mismatch.sum())

    if n_mismatch > 0:
        rate = n_mismatch / n_overlap
        print(f"  [{name_a} <-> {name_b}] {n_overlap} gemeinsame Pixel, {n_mismatch} "
              f"({rate:.2%}) mit abweichendem split.")
        raise SystemExit(
            f"FEHLER: {name_a} ({path_a}) und {name_b} ({path_b}) haben fuer "
            f"{n_mismatch}/{n_overlap} ({rate:.2%}) gemeinsamer Pixelschluessel "
            f"UNTERSCHIEDLICHE split-Werte - vermutlich aus unterschiedlichen "
            f"splits_london.csv-Staenden gebaut (siehe data/processed/MANIFEST.md). Abbruch."
        )

    print(f"  [{name_a} <-> {name_b}] OK - {n_overlap} gemeinsame Pixel, split stimmt exakt ueberein.")


def main() -> None:
    print("=== verify_provenance.py: Split-Konsistenz ueber gemeinsame Pixelschluessel ===")
    for name_a, path_a, name_b, path_b in PAIRS:
        verify_pair(name_a, path_a, name_b, path_b)
    print("\nAlle vorhandenen Paare konsistent.")


if __name__ == "__main__":
    main()
