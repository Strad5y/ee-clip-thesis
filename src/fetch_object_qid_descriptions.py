"""
fetch_object_qid_descriptions.py

Holt Label + Beschreibung fuer die objekteigenen Wikidata-QIDs (Spalte object_qid, siehe
src/extract_pois.py) - NICHT zu verwechseln mit qid_descriptions.parquet (wikidata_extraction.py),
das die Klassen-QIDs aus der Klassentabelle abdeckt. Beide Quellen sind bewusst getrennt
(build_poi_text.py::Phase 4 unterscheidet Klassen- und Objekt-QID).

Idempotent: bereits in object_qid_descriptions.parquet vorhandene QIDs werden nicht erneut
abgefragt. Nach JEDEM Batch wird der aktuelle Stand geschrieben (~166 Requests bei 8.254 QIDs -
ein Abbruch mittendrin darf nicht alles verwerfen, ein Re-Run muss dort weitermachen koennen,
wo der letzte Lauf aufgehoert hat).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))   # damit config gefunden wird

import pandas as pd

import config as cfg

POIS_PATH = cfg.DATA / "interim" / "pois_london_object_qid_check.parquet"
OUT_PATH = cfg.DATA / "interim" / "object_qid_descriptions.parquet"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "ee-clip-thesis/0.1 (Bachelorarbeit, jakob.l.strade@stud.leuphana.de)"
LANGUAGES = ["en", "de"]
BATCH_SIZE = 50           # wbgetentities-Limit pro Request
PAUSE_S = 0.3             # hoefliche Pause zwischen Requests, unabhaengig von Retries
MAX_RETRIES = 6
BACKOFF_BASE_S = 2.0      # exponentiell: 2, 4, 8, 16, 32, 64s


def _fetch_batch(qids: list[str]) -> dict:
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels|descriptions",
        "languages": "|".join(LANGUAGES),
        "format": "json",
    }
    query = "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in params.items())
    req = urllib.request.Request(f"{WIKIDATA_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _fetch_batch_with_retry(qids: list[str]) -> dict:
    """Retry mit exponentiellem Backoff bei HTTP 429 (Rate-Limit) und 5xx (Server-Fehler) -
    genau die Faelle, in denen ein erneuter Versuch sinnvoll ist. Andere HTTPErrors (4xx wie
    400/404) sind ein Anfrageproblem, kein transientes - die eskalieren sofort."""
    delay = BACKOFF_BASE_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _fetch_batch(qids)
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                print(f"    HTTP {e.code} (Versuch {attempt}/{MAX_RETRIES}), "
                      f"warte {delay:.0f}s...")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    Netzwerkfehler ({e}) (Versuch {attempt}/{MAX_RETRIES}), "
                  f"warte {delay:.0f}s...")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")  # Schleife verlaesst sich immer per return/raise


def _fetch_batch_entities(qids: list[str]) -> dict:
    """Wie _fetch_batch_with_retry, faengt aber zusaetzlich ab, dass die wbgetentities-API bei
    EINER einzigen fehlerhaften ID (z.B. syntaktisch gueltig nach Q[0-9]+, aber ausserhalb des
    zulaessigen Entity-Bereichs) fuer den GESAMTEN Batch ein Top-Level-"error"-Objekt statt
    "entities" liefert - beobachtet mit einer 16-stelligen Test-ID. Ohne diesen Fallback wuerden
    dann ALLE ~50 QIDs des Batches faelschlich als 'missing' durchgehen. Fallback: bei fehlendem
    "entities"-Key den Batch einzeln (1 ID/Request) abfragen, damit nur die tatsaechlich defekte
    ID betroffen ist."""
    data = _fetch_batch_with_retry(qids)
    if "entities" in data:
        return data["entities"]

    print(f"    Batch-weiter Fehler ({data.get('error', {}).get('code', '?')}) - "
          f"falle auf Einzelabfragen zurueck ({len(qids)} QIDs)")
    entities = {}
    for qid in qids:
        try:
            single = _fetch_batch_with_retry([qid])
            entities.update(single.get("entities", {qid: {"id": qid, "missing": ""}}))
        except urllib.error.HTTPError as e:
            print(f"      {qid}: dauerhaft fehlerhaft ({e}), als missing gewertet")
            entities[qid] = {"id": qid, "missing": ""}
    return entities


def _parse_entity(qid: str, ent: dict) -> dict:
    if ent.get("missing") is not None:
        return {"object_qid": qid, "label_en": None, "description_en": None,
                "label_de": None, "description_de": None, "missing": True}

    labels = ent.get("labels", {})
    descs = ent.get("descriptions", {})
    return {
        "object_qid": qid,
        "label_en": labels.get("en", {}).get("value"),
        "description_en": descs.get("en", {}).get("value"),
        "label_de": labels.get("de", {}).get("value"),
        "description_de": descs.get("de", {}).get("value"),
        "missing": False,
    }


def load_target_qids(pois_path: Path = POIS_PATH) -> list[str]:
    df = pd.read_parquet(pois_path, columns=["object_qid"])
    return sorted(df["object_qid"].dropna().unique().tolist())


def load_existing(out_path: Path = OUT_PATH) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_parquet(out_path)
    return pd.DataFrame(columns=["object_qid", "label_en", "description_en",
                                  "label_de", "description_de", "missing"])


def fetch_missing(target_qids: list[str], out_path: Path = OUT_PATH) -> pd.DataFrame:
    existing = load_existing(out_path)
    done = set(existing["object_qid"])
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

        # Nach JEDEM Batch persistieren - Abbruch bei Batch 140/166 darf die ersten 139
        # nicht verwerfen.
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
        description="Label + Beschreibung fuer objekteigene Wikidata-QIDs holen (wbgetentities, idempotent)."
    )
    parser.add_argument("--pois", default=str(POIS_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    target_qids = load_target_qids(Path(args.pois))
    result = fetch_missing(target_qids, Path(args.out))

    covered = result[result["object_qid"].isin(target_qids)]
    n_missing = int(covered["missing"].sum())
    n_has_en_desc = int(covered["description_en"].notna().sum())
    n_has_de_desc = int(covered["description_de"].notna().sum())
    n_has_any_desc = int((covered["description_en"].notna() | covered["description_de"].notna()).sum())
    n_has_en_label = int(covered["label_en"].notna().sum())
    n_has_de_label = int(covered["label_de"].notna().sum())
    n_has_any_label = int((covered["label_en"].notna() | covered["label_de"].notna()).sum())

    print(f"\nAbdeckung fuer {len(target_qids)} Ziel-QIDs:")
    print(f"  missing (QID existiert nicht/geloescht): {n_missing}")
    print(f"  Label: en={n_has_en_label}, de={n_has_de_label}, en-oder-de={n_has_any_label}")
    print(f"  Beschreibung: en={n_has_en_desc}, de={n_has_de_desc}, en-oder-de={n_has_any_desc}")
    print(f"\nGespeichert: {args.out}")


if __name__ == "__main__":
    main()
