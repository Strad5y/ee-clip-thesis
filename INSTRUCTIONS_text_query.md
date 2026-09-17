# INSTRUCTIONS_text_query.md — Text-Query-Evaluation über ganz London

Ersetzt die frühere Fassung dieser Datei.

## Ziel

Mit den **bereits trainierten** Checkpoints wird der gemeinsame 256-d-Raum (Text + EO) für ganz Greater London abgefragt. Die Auswertung hat drei Blöcke:

- **Block A – Kategorien (quantitativ):** generischer Kategoriename vs. Klassennamen vs. Template-Beschreibungen, alle auf derselben Ground Truth.
- **Block B – Einzelklassen (quantitativ):** Klassenname vs. Template, jeweils gegen die Ground Truth der Klasse.
- **Block C – Tätigkeiten/Eigenschaften (qualitativ):** Heatmaps plus Steckbriefe der Top-5-Hotspots, die Jakob kurz kommentiert.

Dazu kommen reine Karten-Queries: die AETHER-Phrasen und „desert“ als Kontrolle.

**Kein Training. Trainingscode und Checkpoints werden nicht verändert.**

## Regeln für die gesamte Aufgabe

- Code und Kommentare auf Englisch, Berichte an Jakob auf Deutsch.
- Am Ende jeder Phase **STOP**: berichten und auf Freigabe warten.
- Nichts neu implementieren, was es im Repo schon gibt: Dequantisierung, Text-Encoding-Pfad, Template-Funktion, Rasterisierungsregel der Paarbildung, Laden der Checkpoints. Diese Funktionen werden importiert.
- Jede Modellkennzahl steht neben ihrer Zufallsbaseline. Ergebnisse werden neutral berichtet, ohne positive Deutung. Keine zusätzlichen Varianten oder Ablationen.
- Große Outputs kommen nach `.gitignore`. Code und Configs werden committet, `MANIFEST.md` wird gepflegt.
- **Configs mit `TODO_JAKOB` blockieren die zugehörigen Phasen.** Enthält eine Config noch diesen Platzhalter, wird die Phase nicht gestartet.

---

# TEIL 1 — ohne Jakobs Input ausführbar

## Phase 0 — Repo-Stand prüfen (nur lesen)

1. `git pull` ausführen, dann `verify_provenance.py` als Gate laufen lassen.
2. Folgende Stellen mit Pfad und Funktionsname berichten:
   - Head-Klassen und Checkpoint-Ladelogik (`checkpoints/ae_v1/best.pt`, `checkpoints/tessera_v1/best.pt`).
   - Den **exakten** Text-Encoding-Pfad des Trainings:
     - Modell
     - Instruction-Prefix ja/nein, mit Wortlaut
     - Pooling
     - MRL-Kürzung
     - Reihenfolge von Kürzung und Normalisierung
   - Die Template-Funktion und ob sie sich ohne Namen aufrufen lässt.
   - Die Klassentabelle und die POI-Tabelle (Geometrien, OSM-ID, Name) für London.
   - Die EO-Mosaike für London pro Backbone (Pfad, CRS, Auflösung, NoData, dtype, Pixelzahl) und die Dequantisierungs- bzw. Ladefunktion.
   - Die Rasterisierungsregel der Paarbildung: Punkte, Polygone, Linien, `all_touched`.
   - Die Grenze von Greater London (AOI-Polygon) und das 1-km-Split-Raster.
   - Ob bereits projizierte 256-d-Embeddings existieren.
3. **Speicherplatz prüfen.** Schätze den Bedarf für die projizierten Embeddings: Pixelzahl × 256 × 2 Byte (float16) pro Backbone. Vergleiche mit dem freien Platz.
4. **STOP.** Berichte als Tabelle. Unklare Punkte markierst du deutlich.

## Phase 1 — Projizierte 256-d-Embeddings für ganz London

**Entscheidung (2026-09-17, Jakob):** Phase 1 rechnet NICHTS neu. Die bereits vorhandenen
Arm-C-Felder aus dem SDM-Projekt (`build_sdm_projected_field.py`, trainierter Kopf) werden
**unverändert** als die projizierten 256-d-Embeddings für ganz London verwendet:

- `data/interim/sdm_projected/tessera_armc/*_proj256.tif` (50 Kacheln)
- `data/interim/sdm_projected/ae_armc/*_proj256.tif` (2 Kacheln)

Kein Neurechnen, kein Umformatieren (bleibt float32-GeoTIFF pro Kachel, NoData=`nan` im
Rasterband, kein separates Valid-Maske/JSON-Artefakt, kein `data/projected/{backbone}/`).
Grund: die Felder benutzen bereits denselben Lade-/Normalisierungs-/Projektionscodepfad
(`run_landuse_probe.py::load_trained_head`/`project`) und decken die AOI vollständig ab.

1. **Loader:** `src/eval/text_query/tiles.py` (oder vergleichbar) — stellt pro Backbone die
   Liste der Arm-C-Kacheln einheitlich bereit (Pfad, natives Kachel-CRS, Transform). Arbeit
   erfolgt pro Kachel **im jeweiligen Kachel-CRS** (keine Reprojektion hier); eine gemeinsame
   Reprojektion für Karten (Phase `maps.py`) passiert erst dort, getrennt.
2. **Plausibilitätsprüfung (Stichprobe, kein Vollscan):**
   - L2-Norm der Vektoren liegt bei 1 ± 1e-3 (Stichprobe über mehrere Kacheln je Backbone).
   - 1.000 zufällige **gültige** Pixel werden aus den Roh-Mosaiken (inkl. `dequantize_ae` bzw.
     Z-Score mit den echten `norm_stats`) über `run_landuse_probe.py::load_trained_head`/`project`
     neu projiziert. Kosinus-Ähnlichkeit zum gespeicherten Arm-C-Wert an derselben Pixelposition
     muss ≥ 0.9999 sein, sonst abbrechen.
3. **Kachel-Überlappung prüfen:** Insbesondere an der UTM-Zonengrenze 32630/31 (siehe bekannte
   Leakage-Quelle in `spatial_join.py`-Docstring). Prüfe geometrisch, ob sich Kachel-Footprints
   räumlich überlappen (gemeinsame Fläche in Weltkoordinaten). Falls ja: Ausmaß berichten
   (Fläche/Pixelanteil) und dokumentieren, wie die Paarbildung (`spatial_join.py::assign_utm_zone`)
   bzw. `build_splits_pixels.py` damit umgehen — **nicht selbst entscheiden**, ob/wie das für die
   Text-Query-Auswertung (Similarity-GeoTIFF, Ground Truth) korrigiert werden muss.
4. `MANIFEST.md` ergänzen: verwendete Arm-C-Kachel-Pfade, Checkpoint-Hash/Epoche, Datum der Prüfung.

**STOP.** Berichte die Ergebnisse aller drei Prüfungen (L2-Norm, Reprojektions-Kosinus,
Kachel-Überlappung).

## Phase 2 — Query-Encoder und Ähnlichkeits-Werkzeug

1. Lege `src/eval/text_query/encode.py` an: `encode_queries(texts, backbone) -> [n, 256]`, über denselben Encoding-Pfad wie im Training, danach Text-Head und L2-Normalisierung.
   - Ensembling-Funktion: Die Texte werden einzeln kodiert und normalisiert, dann gemittelt und erneut normalisiert.
   - **Pflicht-Check:** 5 Texte aus dem Text-Cache neu kodieren und vor dem Head mit dem Cache vergleichen. Die Kosinus-Ähnlichkeit muss ≥ 0.9999 sein, sonst brichst du ab.
2. Lege `src/eval/text_query/similarity.py` an: Es erzeugt aus den projizierten Embeddings (Phase 1) und einer Query-Matrix ein mehrbandiges float32-GeoTIFF (ein Band pro Query) im Raster des jeweiligen Backbones.
3. **Testlauf** mit den reinen Karten-Queries, die schon fest sind:
   - „Green Park and Natural Reserve“ (wörtlich aus AETHER v2)
   - „Central Business District“ (wörtlich aus AETHER v2)
   - „desert“ (Kontrolle)

**STOP.** Berichte das Ergebnis des Pflicht-Checks sowie Minimum, Maximum, Mittelwert und 99. Perzentil pro Band.

## Phase 3 — Werkzeuge für Ground Truth, Kennzahlen, Karten und Hotspots

Alles wird generisch gebaut, damit später nur noch Configs eingesetzt werden müssen.

1. **`gt.py`**
   - Nimmt eine Liste von OSM-Klassen, dedupliziert nach `poi_id` und puffert jede Geometrie um **25 m** im metrischen CRS (dadurch werden Punkte zu Polygonen).
   - **Entscheidung (2026-09-17, Jakob):** `all_touched` entfällt ersatzlos. Ground-Truth-Regel = Regel der Paarbildung, nicht separat definiert: die gepufferten Geometrien laufen über exakt denselben Polygon-Pfad wie `spatial_join.py::_join_polygons()` (`exactextract.exact_extract`, flächengewichtete Coverage-Fraction). Positiv ist jedes Pixel mit Coverage-Fraction > 0 — verifiziert gegen die Paarbildung: `coverage` ist dort laut `aggregate_pixel_pairs.py`-Docstring ausschließlich Sortierkriterium, keine Mindestschwelle/Ausschluss, also keine andere Schwelle als „> 0" zu übernehmen.
   - **Linien:** werden übersprungen, exakt wie in der Paarbildung (`spatial_join.py` behandelt nur `Point`/`Polygon`/`MultiPolygon`, `LineString`/`MultiLineString` fallen durch keinen der beiden Zweige). `gt.py` loggt pro Query die Anzahl übersprungener Linien-Instanzen.
   - Gibt zurück: eine binäre Maske und `instance_id → Pixelindizes`.
2. **`metrics.py`**
   - **AP** über alle gültigen Pixel Londons, ohne Subsampling. Auf 1 Mio. Pixeln gegen `sklearn.metrics.average_precision_score` verifizieren (Abweichung < 1e-6).
   - **Recall@Top-k%** für k ∈ {1, 5, 10}. Schwelle ist das (100−k)-Perzentil. Eine Instanz gilt als gefunden, wenn mindestens eines ihrer Pixel ≥ Schwelle ist.
   - **Zufallsbaseline:** 20 Permutationen der Scores; berichtet werden Mittelwert und Standardabweichung.
   - **Bootstrap** über die 1-km-Zellen mit B = 200 und 95-%-Perzentil-Intervall. Für AP werden die Pixel der gezogenen Zellen verwendet, für Recall wird jede Instanz der Zelle ihres Zentroids zugeordnet. **Gepaarte Differenzen:** Alle Varianten einer Query verwenden dieselben Resamples.
3. **`maps.py`**
   - Heatmap über ganz London mit der Grenze von Greater London, optional mit OSM-Ground-Truth als Panel daneben.
   - **Gemeinsame Farbskala:** `vmin` und `vmax` sind das globale 2. und 98. Perzentil über alle Queries und beide Backbones und werden in die Config geschrieben.
   - Für die Darstellung darf auf 50 m gemittelt werden; das wird vermerkt.
   - Ausgabe als PNG mit 300 dpi.
4. **`hotspots.py`**
   - Markiert alle Pixel im obersten 1 % einer Query.
   - Bildet zusammenhängende Flächen mit 8er-Nachbarschaft und verwirft Flächen unter 4 Pixeln.
   - Sortiert nach mittlerem Score und nimmt die **Top 5**.
   - Pro Hotspot entsteht ein Steckbrief (Markdown plus PNG):
     - Fläche in m², mittlerer und maximaler Score, Zentroid
     - OSM-Klassen in der Fläche mit Anzahl
     - Namen der POIs, falls vorhanden
     - Kartenausschnitt mit Umriss auf der Hintergrundkarte, deren Quelle in Teil 2 festgelegt wird. Bis dahin nur Heatmap-Ausschnitt plus OSM-Geometrien.
5. Unit-Tests auf einem kleinen synthetischen Raster: bekannte AP, bekannter Recall, ein bekannter Hotspot.

**STOP.** Berichte die Testergebnisse und die Verifikation gegen sklearn.

---

# TEIL 2 — braucht Jakobs Input

## Phase 4 — Configs einsetzen und prüfen

Jakob füllt `configs/text_query.yaml` (Vorlage siehe unten).

1. Prüfe, ob jede zugeordnete OSM-Klasse in der Klassentabelle vorkommt. Fehlende Klassen meldest du, statt sie zu entfernen.
2. Berichte pro Kategorie und pro Klasse:
   - Instanzzahl in London (Punkt/Polygon)
   - Anteil positiver Pixel mit Puffer
   - die erzeugten Template-Texte wörtlich
3. Prüfe, ob sich Kategorien überlappen, also ob eine Klasse in mehreren Kategorien steht, und melde das.

**STOP.** Jakob bestätigt.

```yaml
block_a_categories:            # 5 Kategorien, Name wörtlich aus dem OS-POI-Schema
  - id: TODO_JAKOB
    os_category_name: "TODO_JAKOB"
    osm_classes: [TODO_JAKOB]  # z. B. "amenity=hospital"
    class_names: [TODO_JAKOB]  # nackte Klassennamen für A2, z. B. "hospital"
  # ... 5 Einträge

block_c_terms:                 # englisch, wie die Trainingstexte
  - term: "TODO_JAKOB"
    expectation: "TODO_JAKOB"  # ein Satz, VOR dem ersten Blick auf Karten festlegen
    expected_osm_classes: [TODO_JAKOB]   # optional, nur für den Steckbrief

map_only:
  - "Green Park and Natural Reserve"
  - "Central Business District"
  - "desert"

hotspot_close_reading_backbones: [TODO_JAKOB]   # z. B. [alphaearth] oder beide
basemap: TODO_JAKOB
```

Jakob committet die Config **vor** Phase 6. Der Commit dient als Zeitstempel für die vorab festgelegten Erwartungen.

## Phase 5 — Block A und B

1. **Block A:** Für jede Kategorie gibt es drei Queries auf **derselben** Ground Truth (Vereinigung aller `osm_classes`).
   - **A1:** `os_category_name`
   - **A2:** Ensemble der `class_names`
   - **A3:** Ensemble der Template-Texte, ohne Namen
2. **Block B:** Für jede Klasse werden Klassenname und Template einzeln gegen die Ground Truth der Klasse ausgewertet.
3. **Kennzahlen:**
   - AP, Recall@Top-{1, 5, 10}%, Zufallsbaseline und Konfidenzintervalle, für beide Backbones.
   - mAP pro Stufe als Mittel über die 5 Kategorien.
   - Gepaarte Differenzen A1−A3, A2−A3 und A1−A2 mit Konfidenzintervall.
4. **Ausgaben:**
   - `outputs/text_query/metrics_block_a.csv` und `metrics_block_b.csv` mit den Spalten `backbone, level, query_id, variant, metric, value, ci_low, ci_high, random_mean, random_std, n_instances, pos_pixel_frac`
   - `diffs_block_a.csv`
5. **Karten:** pro Kategorie eine Reihe A1 | A2 | A3 | Ground Truth, pro Backbone.

**STOP.** Berichte die Tabellen neutral.

## Phase 6 — Block C

1. Heatmaps für alle `block_c_terms` und `map_only`, für beide Backbones, mit gemeinsamer Farbskala.
2. Hotspot-Steckbriefe (Top 5) für die Backbones aus `hotspot_close_reading_backbones`.
3. Lege eine Kommentar-Vorlage `outputs/text_query/block_c_review.md` an. Pro Begriff enthält sie die Erwartung aus der Config und eine Tabelle mit den Spalten `Hotspot | Ort/Namen | OSM-Klassen | passt (ja/teilweise/nein) | Begründung (1 Satz)`. Die letzten beiden Spalten bleiben **leer**, die füllt Jakob aus.

**STOP.**

## Phase 7 — Vergleich mit dem unveränderten Embedding

**GESPERRT, bis Jakob die Methode nach Rücksprache mit Kai freigibt.**

Arbeitsvorschlag, der nicht ohne Freigabe umgesetzt wird: eine Suche per Beispiel.

1. Die Instanzen einer Kategorie werden per festem Seed in zwei Hälften geteilt.
2. Aus Hälfte 1 wird ein Prototyp gebildet (mittleres Embedding ihrer Pixel).
3. Alle Pixel außerhalb von Hälfte 1 werden nach Kosinus-Ähnlichkeit zum Prototyp gerankt.
4. Die AP wird gegen Hälfte 2 gemessen.
5. Das Ganze läuft identisch im rohen Raum und im projizierten 256-d-Raum, mit denselben Hälften.

## Phase 8 — Abschluss

- Logische Commits.
- `MANIFEST.md` ergänzen: Checkpoints, Config-Hash, OSM-Snapshot, Laufzeiten.
- `git push`.
- Kurzbericht an Jakob.
