"""
wikidata_extraction.py

Holt Wikidata-Beschreibungen fuer alle QIDs aus einer flachen key/value/wikidata-Tabelle
(z.B. "wikidata tags/osm_wikidata_keys.xlsx") - EIN Sheet, Spalten key/value/wikidata, kein
Table_True-Aufbau und keine keep-Spalte wie bei class_table.py::load_class_table(), daher
hier direkter pandas-Read statt Wiederverwendung dieser Funktion. Sprache: en mit
de-Fallback (Nutzerentscheidung, da nicht eindeutig in CLAUDE.md festgelegt -
KEY_FRAME-Templates in build_poi_text.py sind ebenfalls englisch).
REST-Endpoint wbgetentities, gebatcht (API-Limit: 50 IDs/Request).
"""
import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
import json
from pathlib import Path

import pandas as pd

import config as cfg

DEFAULT_INPUT = cfg.ROOT / "wikidata tags" / "osm_wikidata_keys.xlsx"
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


QID_PATTERN = r"(Q\d+)"  # gleiche Robustheit wie class_table.py: Tippfehler wie "Q123)" abfangen


def load_source_table(xlsx_path: str | Path) -> pd.DataFrame:
    """Liest die flache key/value/wikidata-Tabelle, bereinigt die QID-Spalte (Whitespace,
    Tippfehler wie 'Q123)'). Behaelt key/value - werden am Ende wieder an die Beschreibung
    angehaengt, nicht nur die eindeutigen QIDs."""
    df = pd.read_excel(xlsx_path)
    required = {"key", "value", "wikidata"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spalten fehlen in {xlsx_path}: {missing} (vorhanden: {list(df.columns)})")

    cleaned = df["wikidata"].astype(str).str.strip().str.extract(QID_PATTERN, expand=False)
    n_bad = int((df["wikidata"].notna() & cleaned.isna()).sum())
    if n_bad > 0:
        print(f"  Warnung: {n_bad} 'wikidata'-Werte ohne gueltiges QID-Muster (Q\\d+), auf NaN gesetzt")

    df = df.copy()
    df["wikidata"] = cleaned
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Wikidata-Beschreibungen fuer QIDs aus einer key/value/wikidata-Tabelle holen."
    )
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT),
                         help=f"Pfad zur xlsx mit Spalten key/value/wikidata (default: {DEFAULT_INPUT})")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    source = load_source_table(args.input)
    qids = sorted(source["wikidata"].dropna().unique().tolist())
    print(f"{len(qids)} eindeutige QIDs aus {len(source)} (key,value)-Zeilen in {args.input}")

    descriptions = fetch_descriptions(qids)  # (wikidata, description), einmal pro eindeutiger QID

    # Beschreibung zurueck auf ALLE (key,value)-Zeilen broadcasten (mehrere Klassen koennen
    # sich dieselbe QID teilen - das ist gewollt, siehe CLAUDE.md 1.1), nicht nur die
    # deduplizierten QIDs ausgeben
    out = source.merge(descriptions, on="wikidata", how="left")
    n_no_qid = int(out["wikidata"].isna().sum())
    if n_no_qid > 0:
        print(f"  Warnung: {n_no_qid} (key,value)-Zeilen ohne gueltige QID (description bleibt leer)")

    n_fallback = int((out["description"] == NO_DESCRIPTION_FALLBACK).sum())
    print(f"Beschreibungen geholt: {len(descriptions)} eindeutige QIDs -> {len(out)} (key,value)-Zeilen, "
          f"davon {n_fallback} ohne echte Beschreibung (Fallback)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"Gespeichert: {out_path}")

    # Zusaetzlich als xlsx (CLAUDE.md 1.1 verlangt .parquet als Pipeline-Format, das allein
    # ist aber nicht manuell durchsichtbar - xlsx daneben zum Pruefen)
    xlsx_path = out_path.with_suffix(".xlsx")
    out.to_excel(xlsx_path, index=False)
    print(f"Gespeichert: {xlsx_path}")


if __name__ == "__main__":
    main()
