"""
wikidata_extraction.py

Holt Wikidata-Beschreibungen fuer alle QIDs aus der Klassentabelle (Table_True, Spalte
"wikidata"). Sprache: en mit de-Fallback (Nutzerentscheidung, da nicht eindeutig in
CLAUDE.md festgelegt - KEY_FRAME-Templates in build_poi_text.py sind ebenfalls englisch).
REST-Endpoint wbgetentities, gebatcht (API-Limit: 50 IDs/Request).
"""
import time
import urllib.error
import urllib.request
import json

import pandas as pd

import config as cfg
from class_table import load_class_table

CLASS_TABLE = cfg.DATA / "OSM_Classes_final.xlsx"
OUT_PATH = cfg.DATA / "interim" / "qid_descriptions.parquet"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "ee-clip-thesis/0.1 (Bachelorarbeit, jakob.l.strade@stud.leuphana.de)"
LANGUAGES = ["en", "de"]  # Praeferenzreihenfolge: en, Fallback de
BATCH_SIZE = 50  # wbgetentities-Limit pro Request
NO_DESCRIPTION_FALLBACK = "(keine Wikidata-Beschreibung verfuegbar)"


def _fetch_batch(qids: list[str]) -> dict:
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "descriptions",
        "languages": "|".join(LANGUAGES),
        "format": "json",
    }
    query = "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in params.items())
    req = urllib.request.Request(f"{WIKIDATA_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_descriptions(qids: list[str]) -> pd.DataFrame:
    """QIDs gebatcht abfragen -> DataFrame(wikidata, description). Fallback-String bei
    fehlender Beschreibung in beiden Sprachen (kein Fehler, keine erfundenen Werte)."""
    records = []
    for i in range(0, len(qids), BATCH_SIZE):
        batch = qids[i:i + BATCH_SIZE]
        try:
            data = _fetch_batch(batch)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Wikidata-API nicht erreichbar (Batch {i}-{i + len(batch)}): {e}") from e

        entities = data.get("entities", {})
        for qid in batch:
            ent = entities.get(qid, {})
            descs = ent.get("descriptions", {})
            desc = None
            for lang in LANGUAGES:
                if lang in descs:
                    desc = descs[lang]["value"]
                    break
            records.append({"wikidata": qid, "description": desc or NO_DESCRIPTION_FALLBACK})

        print(f"  Batch {i // BATCH_SIZE + 1}/{-(-len(qids) // BATCH_SIZE)}: {len(batch)} QIDs abgefragt")
        if i + BATCH_SIZE < len(qids):
            time.sleep(0.2)  # freundlich zur API, kein Rate-Limit-Handling noetig bei 272 QIDs

    return pd.DataFrame.from_records(records)


def main():
    cls = load_class_table(CLASS_TABLE)  # bereinigt "wikidata" bereits (Whitespace, QID-Tippfehler)
    qids = sorted(cls["wikidata"].dropna().unique().tolist())
    print(f"{len(qids)} eindeutige QIDs aus Table_True")

    descriptions = fetch_descriptions(qids)

    n_fallback = int((descriptions["description"] == NO_DESCRIPTION_FALLBACK).sum())
    print(f"Beschreibungen geholt: {len(descriptions)}, davon {n_fallback} ohne echte Beschreibung (Fallback)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptions.to_parquet(OUT_PATH)
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
