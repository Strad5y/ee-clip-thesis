#!/usr/bin/env python3
"""
Liest die Wikidata-Q-IDs aus Spalte 3 des zweiten Tabellenblatts einer Excel-Datei
und holt dazu das englische Label + die englische Beschreibung von Wikidata.

Ausgabe: <input>_enriched.xlsx (Originalblatt + zwei neue Spalten) und optional CSV.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import requests
from openpyxl import load_workbook

API = "https://www.wikidata.org/w/api.php"
QID_RE = re.compile(r"Q\d+")

# >>> KONFIGURIEREN: echte Kontaktadresse eintragen <
HEADERS = {"User-Agent": "OSM-Class-Enricher/1.0 (Bachelor thesis; jakob.l.strade@stud.leuphana.de)"}


def extract_qids(cell_value):
    """Alle Q-IDs aus einer Zelle (fängt auch 'Q123)' oder 'Q1; Q2' ab)."""
    if cell_value is None:
        return []
    return QID_RE.findall(str(cell_value))


def fetch_batch(qids, lang="en", session=None, retries=3):
    """wbgetentities: max. 50 IDs pro Request."""
    session = session or requests.Session()
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels|descriptions",
        "languages": lang,
        "format": "json",
    }
    entities = {}
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            entities = r.json().get("entities", {})
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry nach Fehler: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))

    out = {}
    for qid in qids:
        ent = entities.get(qid, {})
        if "missing" in ent:
            out[qid] = ("", "(nicht gefunden / geloescht)")
            continue
        label = ent.get("labels", {}).get(lang, {}).get("value", "")
        desc = ent.get("descriptions", {}).get(lang, {}).get("value", "")
        out[qid] = (label, desc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--sheet", default=None, help="Blattname (Default: zweites Blatt)")
    ap.add_argument("--col", type=int, default=3, help="Spalte mit den Q-IDs (1-basiert)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="nur extrahieren, nichts abrufen")
    args = ap.parse_args()

    wb = load_workbook(args.xlsx)
    ws = wb[args.sheet] if args.sheet else wb[wb.sheetnames[1]]
    print(f"Blatt: {ws.title} ({ws.max_row} Zeilen), Spalte {args.col}")

    # 1. Q-IDs einsammeln
    row_qids = {}
    for row in range(1, ws.max_row + 1):
        qids = extract_qids(ws.cell(row, args.col).value)
        if qids:
            row_qids[row] = qids

    unique = sorted({q for qs in row_qids.values() for q in qs}, key=lambda x: int(x[1:]))
    print(f"{len(row_qids)} Zeilen mit Q-IDs, {len(unique)} eindeutige Entitaeten")

    if args.dry_run:
        print(unique[:20], "...")
        return

    # 2. In 50er-Batches abrufen
    lookup = {}
    session = requests.Session()
    for i in range(0, len(unique), 50):
        batch = unique[i : i + 50]
        print(f"Abruf {i + 1}-{i + len(batch)} von {len(unique)} ...")
        lookup.update(fetch_batch(batch, args.lang, session))
        time.sleep(0.2)

    # 3. Zurueckschreiben
    label_col = ws.max_column + 1
    desc_col = ws.max_column + 2
    for row, qids in row_qids.items():
        labels = [lookup.get(q, ("", ""))[0] for q in qids]
        descs = [lookup.get(q, ("", ""))[1] for q in qids]
        ws.cell(row, label_col).value = " | ".join(filter(None, labels))
        ws.cell(row, desc_col).value = " | ".join(filter(None, descs))

    out_xlsx = args.xlsx.with_name(args.xlsx.stem + "_enriched.xlsx")
    wb.save(out_xlsx)
    print(f"Geschrieben: {out_xlsx}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qid", f"label_{args.lang}", f"description_{args.lang}"])
            for q in unique:
                lab, desc = lookup.get(q, ("", ""))
                w.writerow([q, lab, desc])
        print(f"Geschrieben: {args.csv}")

    missing = [q for q in unique if not lookup.get(q, ("", ""))[1]]
    if missing:
        print(f"Ohne {args.lang}-Beschreibung: {len(missing)} -> {missing[:15]}")


if __name__ == "__main__":
    main()