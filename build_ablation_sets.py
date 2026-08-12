"""
build_ablation_sets.py

Baut die V1/V2/V3-Klassensets aus dem HILUCS-Crosswalk (Definition vom Nutzer vorgegeben):
  V1 = alle mappbaren HILUCS-Klassen
  V2 = V1 minus wenig informative Klassen, entfernt per IDF-Filter (inverse document
       frequency ueber die Klassenhaeufigkeit in den POIs)
  V3 = nur HILUCS-Klassen, die mit der Evaluations-Ground-Truth (Urban Atlas) assoziiert sind

cfg.CROSSWALK (crosswalk/osm_hilucs_crosswalk.csv) existiert jetzt - aus Tabelle1 von
"wikidata tags/OSM_Classes_final_enriched.xlsx" gebaut (amenity/shop/leisure/tourism, 97
Zeilen). Deckt NUR den aktiven Key-Set ab, kein landuse (siehe CLAUDE.md: landuse laeuft
separat komplett ueber den Crosswalk, dessen HILUCS-Mapping fuer landuse-Werte noch fehlt -
⚠ SEBASTIAN, u.a. fuer allotments/meadow/grass/cemetery/military/garages).

Bekannte Luecke IM aktiven Key-Set: 7 Zeilen (theatre, cinema, arts_centre,
community_centre, museum, gallery, artwork) haben hilucs_label="Cultural services" aber
KEIN hilucs_code - in Tabelle1 fehlt fuer dieses Label durchgehend der Code. Nicht selbst
erfunden (waere eine echte Klassifikationsentscheidung) - macht diese 7 (key,value)
aktuell in V1 unmappbar.

V1 ist damit real ausfuehrbar. V2 braucht einen idf_threshold - noch nicht kalibriert
(siehe compute_idf()-Docstring), main() zeigt daher nur die IDF-Verteilung, waehlt aber
keinen finalen Threshold. V3 bleibt blockiert (Urban-Atlas-Klassenliste fehlt, siehe
eval_landuse.py).
"""
import numpy as np
import pandas as pd

import config as cfg


def load_crosswalk() -> pd.DataFrame:
    return pd.read_csv(cfg.CROSSWALK)


def build_v1(crosswalk: pd.DataFrame) -> set:
    """Alle HILUCS-Klassen, die im Crosswalk ueberhaupt gemappt sind."""
    return set(crosswalk["hilucs_code"].dropna().unique())


def compute_idf(pois: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.Series:
    """IDF je HILUCS-Klasse ueber die POI-Haeufigkeit (log(N / df), N = Anzahl POIs,
    df = Anzahl POIs mit dieser Klasse).

    TODO: 'Dokument' ist hier als einzelnes POI definiert - falls im urspruenglichen
    Opus-Chat ein anderer Dokumentbegriff gemeint war (z.B. raeumliche Zelle/Tile), muss das
    hier angepasst werden. Nicht verifizierbar ohne den Chat-Kontext.
    """
    merged = pois.merge(crosswalk[["key", "value", "hilucs_code"]], on=["key", "value"], how="inner")
    n_docs = merged["poi_id"].nunique()
    df = merged.groupby("hilucs_code")["poi_id"].nunique()
    return np.log(n_docs / df)


def build_v2(crosswalk: pd.DataFrame, pois: pd.DataFrame, idf_threshold: float) -> set:
    """V1 minus Klassen mit IDF unterhalb idf_threshold (= zu haeufig/uninformativ).

    TODO: idf_threshold ist noch nicht kalibriert - erst nach echtem POI-Bestand (nicht nur
    dem Mini-London-Testsample) sinnvoll bestimmbar.
    """
    v1 = build_v1(crosswalk)
    idf = compute_idf(pois, crosswalk)
    keep = set(idf[idf >= idf_threshold].index)
    return v1 & keep


def build_v3(crosswalk: pd.DataFrame, urban_atlas_classes: set) -> set:
    """Nur HILUCS-Klassen, die auch in der Urban-Atlas-Ground-Truth vorkommen.

    TODO: Urban-Atlas-Klassenliste liegt noch nicht vor - siehe eval_landuse.py.
    """
    return build_v1(crosswalk) & urban_atlas_classes


def main():
    crosswalk = load_crosswalk()
    v1 = build_v1(crosswalk)
    print(f"V1: {len(v1)} mappbare HILUCS-Klassen: {sorted(v1)}")

    n_unmapped_label = crosswalk["hilucs_code"].isna().sum()
    if n_unmapped_label > 0:
        print(f"  Warnung: {n_unmapped_label} Crosswalk-Zeilen ohne hilucs_code "
              f"(hilucs_label={crosswalk.loc[crosswalk['hilucs_code'].isna(), 'hilucs_label'].unique().tolist()}) "
              f"- siehe Modul-Docstring, nicht selbst ergaenzt.")

    pois_path = cfg.DATA / "interim" / "pois_london_test_filtered.parquet"
    pois = pd.read_parquet(pois_path, columns=["poi_id", "key", "value"])
    idf = compute_idf(pois, crosswalk).sort_values()
    print(f"\nIDF je HILUCS-Klasse (Mini-London-Testsample, NICHT der finale Datensatz):")
    print(idf.to_string())
    print(
        "\nV2: idf_threshold noch nicht kalibriert (siehe build_v2()-Docstring - erst am "
        "vollen POI-Bestand sinnvoll bestimmbar, nicht am Mini-Testsample). Kein Aufruf hier, "
        "um keinen willkuerlichen Threshold als 'final' erscheinen zu lassen."
    )
    print(
        "\nV3: blockiert - Urban-Atlas-Klassenliste liegt noch nicht vor (siehe eval_landuse.py)."
    )


if __name__ == "__main__":
    main()
