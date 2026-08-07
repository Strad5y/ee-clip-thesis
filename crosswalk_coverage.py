"""
crosswalk_coverage.py

GERUEST - noch nicht ausgefuehrt.
Misst, welcher Anteil der extrahierten OSM (key, value)-Kombinationen ein HILUCS-Mapping im
Crosswalk hat. Braucht cfg.CROSSWALK (crosswalk/osm_hilucs_crosswalk.csv), die es noch nicht
gibt.

TODO: Crosswalk-CSV aus "wikidata tags/OSM_Classes_final_enriched.xlsx" bauen:
  - Tabelle1 hat bereits volles key/value/hilucs_label/hilucs_code-Mapping
    (amenity/shop/tourism/leisure, 97 Zeilen).
  - Tabelle3 deckt landuse-Werte nur mit Wikidata-QID + Label ab, OHNE hilucs_code
    (365 Zeilen, hilucs_label/hilucs_code durchgehend leer) - muss noch ergaenzt werden,
    sonst zeigt dieser Report fuer alle landuse-Werte 0% Coverage.
"""
import pandas as pd

import config as cfg


def load_crosswalk(path=cfg.CROSSWALK) -> pd.DataFrame:
    # TODO: cfg.CROSSWALK existiert noch nicht - siehe Modul-Docstring.
    return pd.read_csv(path)


def coverage(pois_path, crosswalk_path=cfg.CROSSWALK) -> pd.DataFrame:
    """Pro (key, value): n_total, n_mapped, coverage_pct."""
    pois = pd.read_parquet(pois_path)
    crosswalk = load_crosswalk(crosswalk_path)

    merged = pois.merge(crosswalk[["key", "value", "hilucs_code"]], on=["key", "value"], how="left")
    merged["mapped"] = merged["hilucs_code"].notna()

    by_tag = merged.groupby(["key", "value"]).agg(n=("mapped", "size"), n_mapped=("mapped", "sum"))
    by_tag["coverage_pct"] = 100 * by_tag["n_mapped"] / by_tag["n"]
    return by_tag.sort_values("coverage_pct")


def main():
    # TODO: NICHT ausfuehren, bevor cfg.CROSSWALK existiert (siehe Modul-Docstring).
    pois_path = cfg.DATA / "interim" / "pois_london_test_filtered.parquet"
    report = coverage(pois_path)
    print(report)
    total_pct = 100 * report["n_mapped"].sum() / report["n"].sum()
    print(f"Gesamt-Coverage: {total_pct:.1f}%")


if __name__ == "__main__":
    main()
