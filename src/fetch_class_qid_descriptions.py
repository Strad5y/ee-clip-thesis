"""
fetch_class_qid_descriptions.py

Holt Label + Beschreibung fuer die KLASSEN-QIDs aus der Klassentabelle (Spalte 'wikidata',
siehe class_table.py) - NICHT zu verwechseln mit object_qid_descriptions.parquet
(src/fetch_object_qid_descriptions.py), das die objekteigenen POI-QIDs abdeckt. Ersetzt
wikidata_extraction.py (deckte nur den 4-Key-Tabellenstand ab, keine Labels, nicht idempotent).

Wiederverwendet die Batch-/Retry-/Fallback-Logik 1:1 aus fetch_object_qid_descriptions.py
(dort bereits gegen einen echten API-Fehlerfall verifiziert - ein einzelnes ungueltiges QID
im Batch kann die GESAMTE Antwort auf ein Top-Level-"error"-Objekt kippen) statt sie zu
duplizieren. Nur die QID-Quelle (Klassentabelle statt POI-Extraktion) und die Ausgabespalte
('wikidata' statt 'object_qid') unterscheiden sich.

Idempotent: bereits vorhandene QIDs werden nicht erneut abgefragt, nach jedem Batch wird
persistiert.
"""
import argparse
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))   # damit config gefunden wird

import pandas as pd

import config as cfg
from class_table import load_class_table
from fetch_object_qid_descriptions import (
    BATCH_SIZE, PAUSE_S, _fetch_batch_entities,
)

OUT_PATH = cfg.DATA / "interim" / "qid_descriptions.parquet"


def load_target_qids(class_table_path=cfg.CLASS_TABLE) -> list[str]:
    cls = load_class_table(class_table_path)
    return sorted(cls["wikidata"].dropna().unique().tolist())


def _parse_entity(qid: str, ent: dict) -> dict:
    if ent.get("missing") is not None:
        return {"wikidata": qid, "label_en": None, "description_en": None,
                "label_de": None, "description_de": None, "missing": True}

    labels = ent.get("labels", {})
    descs = ent.get("descriptions", {})
    return {
        "wikidata": qid,
        "label_en": labels.get("en", {}).get("value"),
        "description_en": descs.get("en", {}).get("value"),
        "label_de": labels.get("de", {}).get("value"),
        "description_de": descs.get("de", {}).get("value"),
        "missing": False,
    }


def load_existing(out_path: Path = OUT_PATH) -> pd.DataFrame:
    """Achtung: die ALTE qid_descriptions.parquet (wikidata_extraction.py) hat ebenfalls eine
    'wikidata'-Spalte, aber ein anderes Schema (key/value/wikidata/description, 4-Key-Stand) -
    'label_en' ist der eindeutige Marker fuer das NEUE Schema. Ohne diese Unterscheidung wuerde
    load_existing() die veraltete Datei faelschlich als bereits-vollstaendigen Cache lesen."""
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if "label_en" in existing.columns:
            return existing
    return pd.DataFrame(columns=["wikidata", "label_en", "description_en",
                                  "label_de", "description_de", "missing"])


def fetch_missing(target_qids: list[str], out_path: Path = OUT_PATH) -> pd.DataFrame:
    existing = load_existing(out_path)
    done = set(existing["wikidata"])
    todo = [q for q in target_qids if q not in done]

    print(f"{len(target_qids)} Ziel-QIDs, {len(done)} bereits vorhanden, "
          f"{len(todo)} noch abzurufen")

    if not todo:
        print("Nichts zu tun - alle Ziel-QIDs bereits im Cache.")
        return existing

    n_batches = -(-len(todo) // BATCH_SIZE)
    result = existing.copy()

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        batch_no = i // BATCH_SIZE + 1
        try:
            entities = _fetch_batch_entities(batch)
        except Exception as e:
            print(f"  Batch {batch_no}/{n_batches} endgueltig fehlgeschlagen ({e}) - "
                  f"breche ab, {len(result) - len(existing)} neue QIDs bereits gesichert "
                  f"({out_path}). Re-Run holt den Rest nach.")
            raise

        records = [_parse_entity(qid, entities.get(qid, {"missing": ""})) for qid in batch]
        result = pd.concat([result, pd.DataFrame.from_records(records)], ignore_index=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path)

        n_missing_batch = sum(r["missing"] for r in records)
        print(f"  Batch {batch_no}/{n_batches}: {len(batch)} QIDs "
              f"({n_missing_batch} missing) - gespeichert ({len(result)} total)")

        if i + BATCH_SIZE < len(todo):
            time.sleep(PAUSE_S)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Label + Beschreibung fuer Klassen-QIDs der Klassentabelle holen (wbgetentities, idempotent)."
    )
    parser.add_argument("--class-table", default=str(cfg.CLASS_TABLE))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    target_qids = load_target_qids(args.class_table)
    result = fetch_missing(target_qids, Path(args.out))

    covered = result[result["wikidata"].isin(target_qids)]
    n_missing = int(covered["missing"].sum())
    n_has_en_desc = int(covered["description_en"].notna().sum())
    n_has_de_desc = int(covered["description_de"].notna().sum())
    n_has_any_desc = int((covered["description_en"].notna() | covered["description_de"].notna()).sum())

    print(f"\nAbdeckung fuer {len(target_qids)} Ziel-QIDs:")
    print(f"  missing (QID existiert nicht/geloescht): {n_missing}")
    print(f"  Beschreibung: en={n_has_en_desc}, de={n_has_de_desc}, en-oder-de={n_has_any_desc} "
          f"({100*n_has_any_desc/len(target_qids):.1f}%)")
    if n_missing > 0:
        print(f"  QIDs ohne Ergebnis: {sorted(covered[covered['missing']]['wikidata'].tolist())}")

    print(f"\nGespeichert: {args.out}")


if __name__ == "__main__":
    main()
