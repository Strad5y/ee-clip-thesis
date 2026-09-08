"""
poi_classes.py

Gemeinsames Parsing-Modul fuer OSM-Klassen aus TEMPLATE-generiertem pixel_text (siehe
build_poi_text.py/aggregate_pixel_pairs.py) - Nutzervorgabe T6: eine Stelle fuer dieses
Vokabular, damit run_retrieval_baselines.py und run_class_retrieval.py (und kuenftige
Auswertungen) dieselbe Erkennungslogik verwenden, keine zweite Implementierung.

extract_primary_class() (verschoben aus run_retrieval_baselines.py, Verhalten unveraendert):
liefert NUR das primaere (key, value)-Tag - der Top-Coverage-POI (aggregate_pixel_pairs.py-
Sortierung: coverage desc, poi_id asc), und davon nur das erste Tag, falls dieser POI selbst
mehrere simultane OSM-Tags traegt (Multi-Tag-Form).

extract_all_classes() (neu, Nutzervorgabe T6): liefert die VOLLSTAENDIGE Menge aller
(key, value)-Tags ueber ALLE POIs an diesem Pixel (alle ";"-Segmente, siehe
compose_pixel_texts_multi()) und ALLE Glieder jedes Multi-Tag-Segments - nicht nur das erste.
Nutzt DIESELBEN Erkennungsregeln (_KEY_FRAME_BY_LEN/_FRAME_NOUN_BY_LEN/_split_glieder) wie
extract_primary_class(), nur mit groesserem Iterationsumfang.
"""
from build_poi_text import FRAME_NOUN, KEY_FRAME

_KEY_FRAME_BY_LEN = sorted(KEY_FRAME.items(), key=lambda kv: -len(kv[1]))
_FRAME_NOUN_BY_LEN = sorted(FRAME_NOUN.items(), key=lambda kv: -len(kv[1]))


def _split_glieder(t: str) -> list[str]:
    """Kehrt build_poi_text.py::_join_glieder() um: 2 Glieder 'A and B', 3+ Glieder
    'A, B, ..., and Z' (Oxford-Komma) -> Liste der einzelnen Glieder in Original-Reihenfolge."""
    if " and " not in t:
        return [t]
    if ", " not in t:
        return t.split(" and ", 1)
    parts = t.split(", ")
    last = parts[-1]
    if last.startswith("and "):
        last = last[4:]
    return parts[:-1] + [last]


def _match_single_tag(s_lower: str, s: str) -> tuple[str, str] | None:
    """Single-Tag-Form '{frame}: {value}.' -> (key, value), sonst None."""
    for key, frame in _KEY_FRAME_BY_LEN:
        prefix = frame.lower() + ":"
        if s_lower.startswith(prefix):
            value = s[len(prefix):].split(".")[0].strip()
            return (key, value) if value else None
    return None


def _match_glied(g_lower: str, g: str) -> tuple[str, str] | None:
    """Ein Glied der Multi-Tag-Form 'a {value} {noun}' -> (key, value), sonst None."""
    if not g_lower.startswith("a "):
        return None
    for key, noun in _FRAME_NOUN_BY_LEN:
        if g_lower.endswith(noun.lower()) and len(g) > 2 + len(noun):
            value = g[2:len(g) - len(noun)].strip()
            return (key, value) if value else None
    return None


def extract_primary_class(text: str) -> tuple[str | None, str | None]:
    """Wie dataset.py::extract_primary_key(), liefert zusaetzlich den Value des primaeren Tags
    (Top-Coverage-POI bei Multi-POI-Pixeln, siehe aggregate_pixel_pairs.py-Sortierung).
    Single-Tag-Form '{frame}: {value}.' -> Value direkt hinter dem Doppelpunkt. Multi-Tag-Form
    'a {value1} {noun1} and a {value2} {noun2}...' -> ERSTES Glied nach _split_glieder(), dessen
    Praefix 'a '/'A ' und Suffix-Nomen abgestreift werden. Heuristisch wie extract_primary_key
    (Substring-/Template-Parsing), aber zuverlaessig genug fuer eine Diagnose-Metrik - keine
    Grundlage fuer Trainingsdaten."""
    t = text
    if t.startswith("A location containing "):
        t = t[len("A location containing "):]
    t = t.split("; ")[0]
    if " — " in t:
        t = t.split(" — ", 1)[1]
    t = t.rstrip(".")
    t_lower = t.lower()

    hit = _match_single_tag(t_lower, t)
    if hit is not None:
        return hit

    # Nutzervorgabe (Fix uebertragen aus extract_all_classes()): nur der Teil VOR dem ersten
    # Punkt (die Tag-Aufzaehlung) geht in _split_glieder() - eine angehaengte Wikidata-
    # Beschreibung mit Komma wuerde sonst den Oxford-Komma-Zweig fehlleiten und glieder[0]
    # korrumpieren (verifiziert: ~3,2% aller Test-Pixel lieferten dadurch faelschlich
    # (None, None), obwohl die Klasse eigentlich vorhanden war).
    tag_part = t.split(".", 1)[0]
    glieder = _split_glieder(tag_part)
    first = glieder[0].strip()
    hit = _match_glied(first.lower(), first)
    return hit if hit is not None else (None, None)


def extract_all_classes(text: str) -> tuple[set[tuple[str, str]], int]:
    """Vollstaendige Menge aller (key, value)-Tags an diesem Pixel, ueber ALLE POIs (alle
    ';'-Segmente von pixel_text) und ALLE Glieder jeder Multi-Tag-Form - nicht nur der
    Top-Coverage-POI/das erste Glied wie extract_primary_class(). Namenspraefix ('Name — ')
    kann laut compose_pixel_texts_multi() NUR im ersten Segment auftreten, wird daher nur dort
    geprueft.

    Rueckgabe: (Menge der (key,value)-Paare, Anzahl NICHT aufloesbarer Segmente/Glieder -
    weder Single-Tag- noch Multi-Tag-Muster hat gegriffen). Unaufloesbare Einheiten werden
    gezaehlt, nicht still verworfen (Nutzervorgabe T6)."""
    t = text
    if t.startswith("A location containing "):
        t = t[len("A location containing "):]
    t = t.rstrip(".")
    segments = t.split("; ")

    classes: set[tuple[str, str]] = set()
    n_unparsed = 0

    for idx, seg in enumerate(segments):
        s = seg
        if idx == 0 and " — " in s:
            s = s.split(" — ", 1)[1]
        s = s.rstrip(".")
        s_lower = s.lower()

        hit = _match_single_tag(s_lower, s)
        if hit is not None:
            classes.add(hit)
            continue

        # Multi-Tag-Form: build_poi_text.py::compose_listed() haengt bei mehreren Tags eine mit
        # ';' verkettete Beschreibung AN den Glieder-Teil an ('a X und a Y. desc1; desc2') - und
        # diese Beschreibungen koennen selbst Kommas enthalten. _split_glieder() auf den ROHEN
        # Rest anzuwenden (inkl. Beschreibung) laesst den Oxford-Komma-Zweig auf Kommas in der
        # Beschreibung anspringen und zerlegt sie fehlerhaft (verifiziert an echten Beispielen,
        # z.B. Ortsnamen mit Komma in der Wikidata-Beschreibung) - deshalb: nur der Teil VOR dem
        # ersten Punkt geht in _split_glieder(), die Beschreibung selbst wird verworfen (wird fuer
        # (key,value) nicht gebraucht, analog zum Single-Tag-Zweig oben, der genauso nur bis zum
        # ersten Punkt liest).
        tag_part = s.split(".", 1)[0]
        glieder = _split_glieder(tag_part)
        seg_matched = False
        for g in glieder:
            g = g.strip()
            hit = _match_glied(g.lower(), g)
            if hit is not None:
                classes.add(hit)
                seg_matched = True
            else:
                n_unparsed += 1

        if not seg_matched and not glieder:
            n_unparsed += 1

    return classes, n_unparsed


def _smoke_test() -> None:
    assert extract_primary_class("A building: house. building usually intended for living in") \
        == ("building", "house")
    assert extract_primary_class("Greggs — a public amenity or service: fast food.") \
        == ("amenity", "fast food")
    assert extract_primary_class("A pitch leisure spot and a soccer sports facility.") \
        == ("leisure", "pitch")

    classes, n_bad = extract_all_classes(
        "A location containing King's College London Honor Oak Park Sports Ground — a leisure "
        "or recreation place: sports centre. facility...; a leisure or recreation place: pitch. "
        "flat area of grass..."
    )
    assert classes == {("leisure", "sports centre"), ("leisure", "pitch")}, classes
    assert n_bad == 0

    classes2, n_bad2 = extract_all_classes("A pitch leisure spot and a soccer sports facility.")
    assert classes2 == {("leisure", "pitch"), ("sport", "soccer")}, classes2
    assert n_bad2 == 0

    print("Smoke-Test OK: extract_primary_class() und extract_all_classes() konsistent auf "
          "Single-/Multi-Tag-/Multi-POI-Beispielen.")


if __name__ == "__main__":
    _smoke_test()
