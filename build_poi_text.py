"""
build_poi_text.py

Baut aus den langformatigen, klassengefilterten POI-Tags GENAU EINEN Text pro poi_id
(Klassenfilter via class_table -> Wikidata-Description-Merge -> deterministische
Aggregation zu einem Template-Satz). Merged den Text danach auf die Pixel-Paare
(pairs_{backbone}.parquet) -> train_{backbone}.parquet.
"""
import pandas as pd

import config as cfg
from class_table import attach_labels, load_class_table
from spatial_join import BACKBONES

POIS_IN = cfg.DATA / "interim" / "pois_london_test_filtered.parquet"
CLASS_TABLE = cfg.DATA / "OSM_Classes_final.xlsx"
QID_DESCRIPTIONS = cfg.DATA / "interim" / "qid_descriptions.parquet"
POI_TEXT_OUT = cfg.DATA / "interim" / "poi_text.parquet"

PAIRS_DIR = cfg.DATA / "processed"
TRAIN_DIR = cfg.DATA / "processed"

KEY_FRAME = {
    "shop": "a retail shop",
    "amenity": "a public amenity or service",
    "leisure": "a leisure or recreation place",
    "tourism": "a tourism site or accommodation",
}


def clean_val(v):
    return str(v).replace("_", " ")


def compose_text(name, tags):
    # tags: list[dict] mit keys key/value/description, dedupliziert + sortiert
    vals = [clean_val(t["value"]) for t in tags]
    descs = [t["description"] for t in tags
             if t.get("description") and str(t["description"]).lower() != "nan"]
    desc_str = " ".join(descs)

    if len(tags) == 1:
        frame = KEY_FRAME.get(tags[0]["key"], "a place")
        body = f"{frame}: {vals[0]}."
    else:
        body = "a " + " and ".join(vals) + "."

    if desc_str:
        body = f"{body} {desc_str}"

    if name and str(name).strip() and str(name).lower() != "nan":
        return f"{name} — {body}"
    return body[0].upper() + body[1:]   # ohne Name: ersten Buchstaben groß


def _load_tagged(pois_path=POIS_IN, class_table_path=CLASS_TABLE,
                  qid_descriptions_path=QID_DESCRIPTIONS) -> pd.DataFrame:
    """poi_tags laden -> auf behaltene Klassen filtern -> Wikidata-Description mergen."""
    poi_tags = pd.read_parquet(pois_path, columns=["poi_id", "key", "value", "name"])

    cls = load_class_table(class_table_path)
    tagged = attach_labels(poi_tags, cls)  # inner join (key,value) -> haengt "wikidata" an

    qid_desc = pd.read_parquet(qid_descriptions_path)
    tagged = tagged.merge(qid_desc, on="wikidata", how="left")

    # (key,value) je poi_id deduplizieren + deterministisch sortieren (fuer stabile Texte)
    tagged = tagged.drop_duplicates(subset=["poi_id", "key", "value"])
    tagged = tagged.sort_values(["poi_id", "key", "value"]).reset_index(drop=True)
    return tagged


def build_poi_text(tagged: pd.DataFrame) -> pd.DataFrame:
    """Pro poi_id genau eine Zeile: name = erster non-null Wert, Text via compose_text."""
    records = []
    for poi_id, group in tagged.groupby("poi_id", sort=True):
        names = group["name"].dropna()
        name = names.iloc[0] if len(names) else None
        tag_dicts = group[["key", "value", "description"]].to_dict("records")
        text = compose_text(name, tag_dicts)
        records.append({"poi_id": poi_id, "text": text, "n_tags": len(tag_dicts)})
    return pd.DataFrame.from_records(records)


def merge_into_pairs(poi_text: pd.DataFrame):
    """Fuer jeden vorhandenen Backbone: text per Inner-Join auf poi_id an pairs_{backbone} haengen."""
    text_map = poi_text[["poi_id", "text"]]

    for name, raster_path in BACKBONES.items():
        pairs_path = PAIRS_DIR / f"pairs_{name}.parquet"
        if not pairs_path.exists():
            print(f"\n=== {name}: übersprungen (kein {pairs_path.name}) ===")
            continue

        pairs = pd.read_parquet(pairs_path)
        train = pairs.merge(text_map, on="poi_id", how="inner")

        # Sanity: der Merge darf Paare nur droppen (POIs ohne Text), nie vervielfachen
        expected_n = len(pairs.merge(poi_text[["poi_id"]], on="poi_id", how="inner"))
        ok = expected_n == len(train)

        print(f"\n=== {name} ===")
        print(f"  Trainingspaare: {len(train)}  (aus {len(pairs)} Pixel-Paaren)")
        print(f"  nunique(poi_id): {train['poi_id'].nunique()}  nunique(text): {train['text'].nunique()}")
        print(f"  Sanity (keine Vervielfachung durch Merge): {'OK' if ok else 'FEHLER'}")

        out_path = TRAIN_DIR / f"train_{name}.parquet"
        train.to_parquet(out_path)
        print(f"  Gespeichert: {out_path}")


def main():
    tagged = _load_tagged()
    poi_text = build_poi_text(tagged)

    n_pois = len(poi_text)
    n_no_name = int(tagged.drop_duplicates("poi_id")["name"].isna().sum())
    n_no_desc = int((~tagged.groupby("poi_id")["description"].apply(lambda s: s.notna().any())).sum())

    print(f"poi_text: {n_pois} POIs, davon {n_no_name} ohne Name, {n_no_desc} ohne Description")

    POI_TEXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    poi_text.to_parquet(POI_TEXT_OUT)
    print(f"Gespeichert: {POI_TEXT_OUT}")

    merge_into_pairs(poi_text)


if __name__ == "__main__":
    main()
