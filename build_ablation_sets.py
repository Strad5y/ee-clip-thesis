"""
build_ablation_sets.py

Baut die V1/V2/V3-Klassensets (F4: entkoppelt vom HILUCS-Crosswalk, der als totes
Artefakt liegen bleibt, aber in keinem aktiven Codepfad mehr referenziert werden darf -
siehe CLAUDE.md/A3-Audit). Klassen sind ab jetzt (key, value)-Paare aus der Klassentabelle
(class_table.py, keep=True), NICHT mehr HILUCS-Codes.

  V1 = alle behaltenen (key, value)-Klassen aus der Klassentabelle
  V2 = V1 minus wenig informative Klassen, entfernt per IDF-Filter ueber die
       POI-Haeufigkeit je (key, value)-Klasse
  V3 = nur Klassen, die sich einer Urban-Atlas-Kategorie zuordnen lassen (Ground-Truth-
       assoziiert) - blockiert, bis die Urban-Atlas-Klassenliste vorliegt (siehe eval_landuse.py)

idf_threshold fuer V2 bleibt bewusst unkalibriert (siehe compute_idf()/main()) - erst am
vollen POI-Bestand sinnvoll bestimmbar, nicht am Mini-London-Testsample.
"""
import numpy as np
import pandas as pd

import config as cfg
from class_table import load_class_table

CLASS_TABLE = cfg.CLASS_TABLE


def load_classes(class_table_path=CLASS_TABLE) -> pd.DataFrame:
    return load_class_table(class_table_path)


def build_v1(classes: pd.DataFrame) -> set:
    """Alle behaltenen (key, value)-Klassen aus der Klassentabelle."""
    return set(zip(classes["key"], classes["value"]))


def compute_idf(pois: pd.DataFrame, classes: pd.DataFrame) -> pd.Series:
    """IDF je (key, value)-Klasse ueber die POI-Haeufigkeit (log(N / df), N = Anzahl POIs,
    df = Anzahl POIs mit dieser Klasse).

    TODO: 'Dokument' ist hier als einzelnes POI definiert - falls im urspruenglichen
    Opus-Chat ein anderer Dokumentbegriff gemeint war (z.B. raeumliche Zelle/Tile), muss das
    hier angepasst werden. Nicht verifizierbar ohne den Chat-Kontext.
    """
    merged = pois.merge(classes[["key", "value"]], on=["key", "value"], how="inner")
    n_docs = merged["poi_id"].nunique()
    df = merged.groupby(["key", "value"])["poi_id"].nunique()
    return np.log(n_docs / df)


def build_v2(classes: pd.DataFrame, pois: pd.DataFrame, idf_threshold: float) -> set:
    """V1 minus Klassen mit IDF unterhalb idf_threshold (= zu haeufig/uninformativ).

    TODO: idf_threshold ist noch nicht kalibriert - erst nach echtem POI-Bestand (nicht nur
    dem Mini-London-Testsample) sinnvoll bestimmbar.
    """
    v1 = build_v1(classes)
    idf = compute_idf(pois, classes)
    keep = set(idf[idf >= idf_threshold].index)
    return v1 & keep


def build_v3(classes: pd.DataFrame, urban_atlas_classes: set) -> set:
    """Nur Klassen, die auch in der Urban-Atlas-Ground-Truth vorkommen (Ground-Truth-assoziiert).

    TODO: Urban-Atlas-Klassenliste liegt noch nicht vor - siehe eval_landuse.py.
    urban_atlas_classes wird als Menge von (key, value)-Tupeln erwartet, sobald diese
    Zuordnung existiert.
    """
    return build_v1(classes) & urban_atlas_classes


def main():
    classes = load_classes()
    v1 = build_v1(classes)
    print(f"V1: {len(v1)} behaltene (key, value)-Klassen: {sorted(v1)}")

    pois_path = cfg.DATA / "interim" / "pois_london_test_filtered.parquet"
    pois = pd.read_parquet(pois_path, columns=["poi_id", "key", "value"])
    idf = compute_idf(pois, classes).sort_values()
    print("\nIDF je (key, value)-Klasse (Mini-London-Testsample, NICHT der finale Datensatz):")
    print(idf.to_string())
    print(
        "\nV2: idf_threshold noch nicht kalibriert (siehe build_v2()-Docstring - erst am "
        "vollen POI-Bestand sinnvoll bestimmbar, nicht am Mini-Testsample). Kein Aufruf hier, "
        "um keinen willkuerlichen Threshold als 'final' erscheinen zu lassen."
    )
    print(
        "\nV3: blockiert - Urban-Atlas-Klassenliste liegt noch nicht vor (siehe eval_landuse.py)."
    )


def _smoke_test():
    """Synthetische Klassentabelle + POIs statt echter Daten (Klassentabelle wird gerade
    ersetzt, siehe A3-Audit; echte POIs existieren noch nicht - Paarbau ist diese Session
    bewusst nicht gestartet)."""
    classes = pd.DataFrame({
        "key": ["amenity", "amenity", "shop", "leisure"],
        "value": ["cafe", "bench", "bakery", "park"],
        "keep": [True, True, True, True],
    })
    # bench kommt sehr haeufig vor (uninformativ), park ist selten
    pois = pd.DataFrame({
        "poi_id": [f"node/{i}" for i in range(10)],
        "key": ["amenity"] * 6 + ["amenity"] + ["shop"] * 2 + ["leisure"],
        "value": ["bench"] * 6 + ["cafe"] + ["bakery"] * 2 + ["park"],
    })

    v1 = build_v1(classes)
    assert v1 == {("amenity", "cafe"), ("amenity", "bench"), ("shop", "bakery"), ("leisure", "park")}

    idf = compute_idf(pois, classes)
    assert idf[("amenity", "bench")] < idf[("leisure", "park")], "haeufige Klasse muss niedrigeren IDF haben"

    v2 = build_v2(classes, pois, idf_threshold=idf.median())
    assert ("amenity", "bench") not in v2, "haeufige/uninformative Klasse sollte in V2 gefiltert werden"
    assert v2 <= v1

    urban_atlas_classes = {("leisure", "park"), ("amenity", "cafe")}
    v3 = build_v3(classes, urban_atlas_classes)
    assert v3 == {("leisure", "park"), ("amenity", "cafe")}

    print(f"Smoke-Test OK: V1={len(v1)}, V2={len(v2)} (bench gefiltert), V3={len(v3)} "
          f"(Ground-Truth-assoziiert) - alles ueber (key,value)-Klassen, kein Crosswalk mehr involviert.")


if __name__ == "__main__":
    import sys
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        main()
