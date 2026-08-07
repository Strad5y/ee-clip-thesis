"""
build_ablation_sets.py

GERUEST - noch nicht ausgefuehrt. Baut die V1/V2/V3-Klassensets aus dem HILUCS-Crosswalk
(Definition vom Nutzer vorgegeben):
  V1 = alle mappbaren HILUCS-Klassen
  V2 = V1 minus wenig informative Klassen, entfernt per IDF-Filter (inverse document
       frequency ueber die Klassenhaeufigkeit in den POIs)
  V3 = nur HILUCS-Klassen, die mit der Evaluations-Ground-Truth (Urban Atlas) assoziiert sind

Setzt cfg.CROSSWALK voraus (siehe crosswalk_coverage.py), das noch nicht existiert. V3
setzt zusaetzlich die Urban-Atlas-Klassenliste voraus (siehe eval_landuse.py).
"""
import numpy as np
import pandas as pd

import config as cfg


def load_crosswalk() -> pd.DataFrame:
    # TODO: cfg.CROSSWALK existiert noch nicht - siehe crosswalk_coverage.py Docstring.
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
    # TODO: NICHT ausfuehren - Crosswalk (crosswalk_coverage.py) und Urban-Atlas-Klassenliste
    # (eval_landuse.py) fehlen noch.
    raise NotImplementedError(
        "build_ablation_sets.py ist ein Geruest. Voraussetzungen fehlen: "
        "cfg.CROSSWALK (siehe crosswalk_coverage.py) und Urban-Atlas-Klassenliste (siehe eval_landuse.py)."
    )


if __name__ == "__main__":
    main()
