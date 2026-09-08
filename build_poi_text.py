"""
build_poi_text.py

Phase 4 (CLAUDE_CODE_TASKS_phase4_7.md): Baut aus den langformatigen POI-Tags GENAU EINEN Text
pro poi_id. Ersetzt den alten Inner-Join-gegen-Klassentabelle (verwarf POIs ohne gelistete
Klasse) durch die Aufnahmeregel A/B/C/D:

  Fall A: gelistete Klasse + object_qid -> aufnehmen, objekteigene Beschreibung ERSETZT die
          Klassenbeschreibung (nicht zusaetzlich), faellt zurueck auf Klassenbeschreibung(en)
          nur wenn keine objekteigene Beschreibung existiert.
  Fall B: gelistete Klasse, keine object_qid -> aufnehmen, unveraendert (Klassenbeschreibung
          falls vorhanden).
  Fall C: keine gelistete Klasse, aber object_qid -> NEU aufgenommen. Text aus den (nicht
          gelisteten) Tags des POI + objekteigener Beschreibung.
  Fall D: weder noch -> verworfen.

Kein train_{backbone}.parquet mehr hier (das war der alte POI-Ebene-Merge auf pairs_*.parquet) -
die Pixel-Ebene-Aggregation (Phase 5, src/aggregate_pixel_pairs.py) uebernimmt diesen Schritt
jetzt direkt auf Pixel-Ebene und konsumiert poi_text_full.parquet + pairs_*_full.parquet.
"""
import pandas as pd

import config as cfg
from class_table import load_class_table

POIS_IN = cfg.DATA / "interim" / "pois_london_full_filtered.parquet"
CLASS_TABLE = cfg.CLASS_TABLE
CLASS_DESCRIPTIONS = cfg.DATA / "interim" / "qid_descriptions.parquet"          # Klassen-QID-Ebene
OBJECT_DESCRIPTIONS = cfg.DATA / "interim" / "object_qid_descriptions.parquet"  # Objekt-QID-Ebene
POI_TEXT_OUT = cfg.DATA / "interim" / "poi_text_full.parquet"  # NEU, altes poi_text.parquet unveraendert

KEY_FRAME = {
    "shop": "a retail shop",
    "amenity": "a public amenity or service",
    "leisure": "a leisure or recreation place",
    "tourism": "a tourism site or accommodation",
    "building": "a building",
    "healthcare": "a healthcare facility",
    "historic": "a historic site",
    "office": "an office or workplace",
    "sport": "a sports facility",
    "water": "a body of water",
    "waterway": "a watercourse",
    "military": "a military site",
    "natural": "a natural feature",
}

# Nur fuer die MULTI-Tag-Form ('{value} {frame_noun}', z.B. 'a bakery shop') - die
# Single-Tag-Form nutzt weiterhin KEY_FRAME (Doppelpunkt-Form, nie kaputt gewesen). Nutzer-
# entschieden: 'amenity' bleibt bewusst generisch (Values zu heterogen fuer ein spezifisches
# Nomen wie 'restaurant').
FRAME_NOUN = {
    "shop": "shop",
    "amenity": "amenity",
    "leisure": "leisure spot",
    "tourism": "tourism site",
    "building": "building",
    "healthcare": "healthcare facility",
    "historic": "historic site",
    "office": "office",
    "sport": "sports facility",
    "water": "water feature",
    "waterway": "watercourse",
    "military": "military site",
    "natural": "natural feature",
}


def clean_val(v):
    return str(v).replace("_", " ")


def _prefix_name(name, body):
    if name and str(name).strip() and str(name).lower() != "nan":
        return f"{name} — {body}"
    return body[0].upper() + body[1:]   # ohne Name: ersten Buchstaben gross


def _join_glieder(glieder: list[str]) -> str:
    """2 Glieder: nur 'and'. 3+: Kommas + finales 'and' (Oxford-Komma)."""
    if len(glieder) == 2:
        return f"{glieder[0]} and {glieder[1]}"
    return ", ".join(glieder[:-1]) + f", and {glieder[-1]}"


def _frame_and_values(tags):
    """Single-Tag: unveraendert die alte Doppelpunkt-Form (nie kaputt gewesen). Multi-Tag-FIX:
    frueher verwarf ' and '.join(vals) die Keys komplett ('a dentist and dentist' bei
    amenity=dentist + healthcare=dentist, ununterscheidbar). Jetzt bekommt JEDES Tag ein
    eigenes Glied '{value} {frame_noun}' - KEIN Value-Dedup hier (Nutzervorgabe: beide Keys
    bleiben im Glied-Teil sichtbar, z.B. 'a dentist amenity and a dentist healthcare
    facility' - die unterschiedlichen frame_nouns machen die Wiederholung nicht mehr wie ein
    Bug lesbar). Dedup passiert stattdessen nur bei den Beschreibungen, siehe compose_listed."""
    if len(tags) == 1:
        frame = KEY_FRAME.get(tags[0]["key"], "a place")
        return f"{frame}: {clean_val(tags[0]['value'])}."
    glieder = [f"a {clean_val(t['value'])} {FRAME_NOUN.get(t['key'], 'place')}" for t in tags]
    return _join_glieder(glieder) + "."


def _ordered_descriptions(tags) -> list[str]:
    """Beschreibungs-Dedup auf dem NORMALISIERTEN Value (nicht (key,value)): teilen sich
    mehrere Tags denselben Value (z.B. amenity=dentist + healthcare=dentist), wird nur die
    Beschreibung des ERSTEN Vorkommens durchgelassen. tags kommt bereits alphabetisch nach
    key sortiert an (siehe _load_poi_tags) - "erstes Vorkommen" ist damit automatisch der
    alphabetisch erste Key, deterministisch. Reihenfolge der Rueckgabe = Reihenfolge der
    Glieder (fuer die Semikolon-Verkettung in compose_listed)."""
    seen_values = set()
    descs = []
    for t in tags:
        v = clean_val(t["value"])
        if v in seen_values:
            continue
        seen_values.add(v)
        desc = t.get("class_description")
        if desc and str(desc).lower() != "nan":
            descs.append(desc)
    return descs


def compose_listed(name, tags, object_description):
    """Fall A/B: Text aus den GELISTETEN Tags des POI (nicht gelistete Tags desselben POI
    tragen wie im alten Code nichts bei). object_description ist nur bei Fall A gesetzt
    (Fall B hat per Definition keine object_qid) und ERSETZT dann die Klassenbeschreibung(en)
    komplett, statt sie zu ergaenzen - faellt auf die Klassenbeschreibung(en) zurueck, wenn
    object_description leer ist (Objekt-QID vorhanden, aber Wikidata liefert keine Beschreibung).
    Klassenbeschreibungen werden mit Semikolon verkettet (frueher: Leerzeichen, verlor die
    Satzgrenze), in Glied-Reihenfolge, deduped auf den Value (siehe _ordered_descriptions).
    Rueckgabe (text, body): body = OHNE Namensprefix (Phase 5 braucht das fuer POIs ab
    Position 2 in einem Pixel - 'nur Frame + Value + Beschreibung', kein Name)."""
    body = _frame_and_values(tags)

    if object_description:
        desc_str = object_description
    else:
        desc_str = "; ".join(_ordered_descriptions(tags))

    if desc_str:
        body = f"{body} {desc_str}"
    return _prefix_name(name, body), body


def compose_unlisted(name, tags, object_description):
    """Fall C: alle (nicht gelisteten) Tags des POI, EINE gemeinsame objekteigene Beschreibung
    (es gibt nur eine QID/Beschreibung je POI, nicht je Tag - anders als Fall B's
    Klassenbeschreibungen, die je Tag existieren koennen)."""
    body = _frame_and_values(tags)
    if object_description:
        body = f"{body} {object_description}"
    return _prefix_name(name, body), body


def _load_poi_tags(pois_path=POIS_IN) -> pd.DataFrame:
    df = pd.read_parquet(pois_path, columns=["poi_id", "key", "value", "name", "object_qid"])
    df = df.drop_duplicates(subset=["poi_id", "key", "value"])
    return df.sort_values(["poi_id", "key", "value"]).reset_index(drop=True)


def _attach_descriptions(poi_tags: pd.DataFrame, class_table_path=CLASS_TABLE,
                          class_desc_path=CLASS_DESCRIPTIONS,
                          object_desc_path=OBJECT_DESCRIPTIONS) -> pd.DataFrame:
    """LEFT JOIN statt des alten Inner-Joins - nicht gelistete Tags (Fall C/D) muessen
    erhalten bleiben, nicht verworfen werden. Haengt has_class (bool je Tag-Zeile),
    class_description (falls (key,value) gelistet UND eine Beschreibung existiert, en mit
    de-Fallback) sowie label_en/description_en/description_de (objekteigen, je poi_id via
    object_qid) an. qid_descriptions.parquet (Schritt 1, Phase-4-Nachtrag) traegt seit dem
    Regenerieren gegen die neue Klassentabelle KEINE key/value-Spalten mehr, nur noch
    wikidata/label_en/description_en/description_de/missing (analog object_qid_descriptions.
    parquet) - Merge deshalb ueber die Klassentabelle selbst auf 'wikidata', nicht mehr
    direkt auf (key,value)."""
    cls = load_class_table(class_table_path)
    cls_pairs = cls[["key", "value", "wikidata"]].drop_duplicates(["key", "value"])
    cls_pairs = cls_pairs.assign(has_class=True)
    tagged = poi_tags.merge(cls_pairs, on=["key", "value"], how="left")
    tagged["has_class"] = tagged["has_class"].fillna(False)

    class_desc = pd.read_parquet(class_desc_path, columns=["wikidata", "description_en", "description_de"])
    class_desc = class_desc.drop_duplicates("wikidata").rename(
        columns={"description_en": "class_description_en", "description_de": "class_description_de"})
    tagged = tagged.merge(class_desc, on="wikidata", how="left")
    tagged["class_description"] = tagged["class_description_en"].where(
        tagged["class_description_en"].notna(), tagged["class_description_de"])

    obj_desc = pd.read_parquet(
        object_desc_path, columns=["object_qid", "label_en", "description_en", "description_de"])
    tagged = tagged.merge(obj_desc, on="object_qid", how="left")
    return tagged


def _object_description(row_first) -> str | None:
    """en mit de-Fallback (Projektkonvention, siehe wikidata_extraction.py)."""
    for col in ("description_en", "description_de"):
        v = row_first.get(col)
        if v and str(v).lower() != "nan":
            return v
    return None


def build_poi_text(tagged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pro poi_id genau eine Zeile (Faelle A/B/C), Fall D wird verworfen. counters: fuer den
    Bericht (Faelle, Namensslot-Stufe-2-Treffer, Fall-C-POIs ohne Beschreibung)."""
    records = []
    counters = {"A": 0, "B": 0, "C": 0, "D": 0, "name_from_label_en": 0, "c_without_description": 0}

    for poi_id, group in tagged.groupby("poi_id", sort=True):
        has_qid = group["object_qid"].notna().any()
        has_class_any = group["has_class"].any()

        if has_class_any and has_qid:
            case = "A"
        elif has_class_any and not has_qid:
            case = "B"
        elif not has_class_any and has_qid:
            case = "C"
        else:
            case = "D"
        counters[case] += 1
        if case == "D":
            continue

        # Namensslot: 1) OSM-name, 2) sonst Wikidata-label_en (nur wenn object_qid vorhanden),
        # 3) sonst entfaellt der Namensslot.
        names = group["name"].dropna()
        name = names.iloc[0] if len(names) else None
        used_label_fallback = False
        if not name:
            labels = group["label_en"].dropna()
            if len(labels):
                name = labels.iloc[0]
                used_label_fallback = True
        if used_label_fallback:
            counters["name_from_label_en"] += 1

        object_description = _object_description(group.iloc[0]) if has_qid else None

        if case in ("A", "B"):
            listed = group[group["has_class"]]
            tag_dicts = listed[["key", "value", "class_description"]].to_dict("records")
            text, body = compose_listed(name, tag_dicts,
                                         object_description if case == "A" else None)
        else:  # Fall C: ALLE Tags des POI (keiner ist gelistet)
            tag_dicts = group[["key", "value"]].to_dict("records")
            text, body = compose_unlisted(name, tag_dicts, object_description)
            if not object_description:
                counters["c_without_description"] += 1

        records.append({"poi_id": poi_id, "text": text, "body": body, "case": case,
                         "n_tags": len(tag_dicts)})

    return pd.DataFrame.from_records(records), counters


def main():
    poi_tags = _load_poi_tags()
    tagged = _attach_descriptions(poi_tags)
    poi_text, counters = build_poi_text(tagged)

    n_pois_total = counters["A"] + counters["B"] + counters["C"] + counters["D"]
    n_kept = len(poi_text)
    print(f"POIs gesamt: {n_pois_total}")
    print(f"  Fall A (Klasse+QID):   {counters['A']:>8}  ({100*counters['A']/n_pois_total:.2f}%)")
    print(f"  Fall B (nur Klasse):   {counters['B']:>8}  ({100*counters['B']/n_pois_total:.2f}%)")
    print(f"  Fall C (nur QID, NEU): {counters['C']:>8}  ({100*counters['C']/n_pois_total:.2f}%)")
    print(f"  Fall D (weder noch, verworfen): {counters['D']:>8}  ({100*counters['D']/n_pois_total:.2f}%)")
    print(f"Aufgenommen (A+B+C): {n_kept}")
    print(f"Eindeutige Texte: {poi_text['text'].nunique()} "
          f"({100*poi_text['text'].nunique()/n_kept:.2f}% von {n_kept})")
    print(f"Namensslot Stufe 2 (label_en-Fallback) gegriffen: {counters['name_from_label_en']}")
    print(f"Fall-C-POIs ohne verfuegbare Beschreibung: {counters['c_without_description']} "
          f"von {counters['C']} ({100*counters['c_without_description']/max(counters['C'],1):.2f}%)")

    POI_TEXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    poi_text.to_parquet(POI_TEXT_OUT)
    print(f"\nGespeichert (NEU, altes poi_text.parquet unveraendert): {POI_TEXT_OUT}")


if __name__ == "__main__":
    main()
