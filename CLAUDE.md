# CLAUDE_CODE_TASKS.md — Restarbeiten ee-clip-thesis

Arbeitsanweisung für Claude Code. Reihenfolge = Abhängigkeitsreihenfolge. Jede Aufgabe:
**Ziel → Umsetzung → Erwarteter Output → Abnahme-Check**. Immer *einen* Schritt umsetzen,
erwarteten Output nennen, dann ausführen. Bei echten methodischen Zweifelsfällen: **nicht selbst
entscheiden, an Kai eskalieren** (Punkte unten markiert mit `⚠ KAI`).

---

## 0. Fixe Constraints (NICHT ändern, nicht neu herleiten)

Diese Entscheidungen sind settled und stehen in `CLAUDE.md`. Sie gelten für alle Aufgaben.

| Bereich | Festlegung |
|---|---|
| Backbones | Zwei **eingefrorene** EE-Backbones: TESSERA (128-dim, 10 m), AlphaEarth (64-dim, bereits L2-normalisiert). Getrennte Two-Tower-Modelle pro Backbone (faire Vergleichbarkeit). |
| Text-Encoder | Qwen3-Embedding-4B (Apache-2.0, decoder-only, **last-token pooling**, MRL: 2560 → auf 1024 truncaten, *dann* projizieren). |
| Trainierbar | **Nur** Projection Heads: `Linear → GELU → Dropout → Linear → L2-normalize`, gemeinsame Latentdim `d = 256`. |
| Loss | **SigLIP** (pairwise sigmoid), learnbare Temperatur + Bias. **NICHT** InfoNCE/Softmax. Behandelt Multi-Positives pro Zelle ohne Masking. |
| Normalisierung | z-score für TESSERA-Inputs. AlphaEarth **nicht** normalisieren (bereits unit-norm). |
| Trainingspaare | Ein Eintrag pro geschnittenem 10-m-Pixel; Coverage-Fraction als Subsampling-Gewicht. |
| POI-Daten | Long-Format (eine Zeile pro key-value-Tag pro `poi_id`). Tag-Linkage per `poi_id`-Merge **nach** dem Join. |
| Aktive OSM-Keys | Maßgeblich ist der Code (`build_poi_text.py::KEY_FRAME`/`FRAME_NOUN`), **13 Keys**: `amenity`, `shop`, `leisure`, `tourism`, `building`, `healthcare`, `historic`, `office`, `sport`, `water`, `waterway`, `military`, `natural`. Korrigiert 2026-09-17 — frühere Fassung nannte hier nur 4 Keys, das war nicht mehr der Ist-Stand des Codes. `landuse` **nicht** im aktiven Key-Set — läuft komplett über den HILUCS-Crosswalk. |
| Multi-Tag | MultiPolygons **nicht** explodieren (sonst Duplikat-Paare). Carry-Columns beim Join dürfen `key`/`value` **nicht** enthalten. |
| Vergleich | **Kein** direkter Zahlenvergleich gegen publizierte Paper (andere Regionen/Tasks/Splits). Nur Ablation auf eigenen Eval-Tasks: Delta raw ↔ enriched. |

---

## 0.5 Phase 0 — Sync & Ist-Zustand

- **Ziel:** Sauberer Start, echten Code-Stand feststellen (Memory kann veraltet sein).
- **Umsetzung:** `git pull`; conda env `eeclip` aktivieren; Repo-Baum ausgeben; prüfen welche der unten genannten Dateien schon existieren und welchen Reifegrad sie haben (fertig / TODO-Gerüst / fehlt).
- **Erwarteter Output:** Kurze Statustabelle je Datei: `spatial_join.py, class_table.py, build_training_pairs.py, validate_pairs.py, models/*, train.py, eval/*, ablation/*`.
- **Abnahme:** `pairs_tessera.parquet` existiert und hat 104.483 Zeilen, Embedding-Dim 128, NaN-Count 0, Coverage-Mean ≈ 0.69. Falls nicht → stoppen und melden.

---

## 1. Phase 1 — Textseite (Daten)

### 1.1 Wikidata-Beschreibungen holen
- **Ziel:** `qid_descriptions.parquet` für alle QIDs aus der Klassentabelle.
- **Umsetzung:** Eindeutige QIDs aus `Table_True` (`class_table.load_class_table()`, Spalte `wikidata`) sammeln. Beschreibungen per Wikidata (SPARQL `wikibase:description` @ `de` bzw. `en`, oder REST `wbgetentities`) batchen. Sprache konsistent wählen (Empfehlung: `de` mit `en`-Fallback — falls unklar `⚠ KAI`, welche Sprache in den Text-Templates). Geteilte QIDs über mehrere Klassen sind **gewollt**, kein Fehler.
- **Erwarteter Output:** `data/interim/qid_descriptions.parquet`, Spalten `wikidata, description`.
- **Abnahme:** Zeilenzahl = Anzahl eindeutiger QIDs; keine leeren `description` (bzw. dokumentierter Fallback-String für fehlende).

### 1.2 `build_training_pairs.py` fertigstellen (Tag-Merge + Text)
- **Ziel:** Aus `pairs_tessera.parquet` + Klassentabelle + QID-Beschreibungen die fertigen Trainingszeilen mit Text bauen.
- **Umsetzung:**
  1. Post-Join Tag-Merge über `poi_id` (Long-Format), Labels via `class_table.attach_labels()`.
  2. Text-Templates anwenden:
     - Single-Tag: `{name} — {key_frame}: {value}. {wikidata_description}`
     - Multi-Tag: `{name} — a {value1} and {value2}. {desc1} {desc2}`
     - Key-Frames pro aktivem Key wie in `CLAUDE.md` definiert.
  3. QID-Beschreibungen per `wikidata`-Merge einhängen.
  4. Constraints beachten: keine `key`/`value` als Carry-Cols beim Join; MultiPolygons nicht explodieren.
- **Erwarteter Output:** `data/processed/pairs_tessera_text.parquet` mit `poi_id, embedding[128], text, key, value, coverage, label, wikidata`.
- **Abnahme:** Zeilenzahl gegenüber Input erklärbar (kein spuriöses Multiplizieren); jede Zeile hat nicht-leeren `text`; NaN-Count Embedding = 0. Anschließend `validate_pairs.py` (8 Checks) grün.

### 1.3 Text-Embeddings cachen
- **Ziel:** Texte einmal mit Qwen3-Embedding-4B encodieren und cachen (Encoder ist frozen).
- **Umsetzung:** Eindeutige Texte encodieren; last-token pooling; MRL 2560 → 1024 truncaten; L2-normalisieren. Über Text-Hash cachen, damit Ablationen nicht neu encodieren.
- **Erwarteter Output:** `data/processed/text_emb.parquet` (`text_hash, emb[1024]`).
- **Abnahme:** Dim = 1024, L2-Norm ≈ 1.0, Count = #eindeutige Texte.

---

## 2. Phase 2 — AlphaEarth beschaffen & joinen

### 2.1 AlphaEarth-COGs herunterladen (empfohlener Weg: GCS)
- **Ziel:** AlphaEarth Satellite Embedding V1 (Annual), Jahr **2025**, 64-dim, 10 m für beide Studienregionen lokal als COG.
- **Warum GCS statt GEE-Export:** Öffentlicher Bucket `gs://alphaearth_foundations`, seit Juli 2026 *provider-pays* (kein Egress-Kostenrisiko), COG-Format → identisch zur bestehenden rasterio/exactextract-TESSERA-Pipeline, kein GEE-Export-Job/Quota nötig.
- **Umsetzung:**
  1. `gcloud` CLI (ARM64-Linux, läuft auf DGX Spark) oder `gsutil`; ggf. `gcloud auth login`.
  2. Bucket-Struktur: `Jahr/UTM-Zone/…` (Top-Level pro Jahr, darunter 120 UTM-Zonen-Unterordner).
     - **Niedersachsen:** UTM-Zone **32N**.
     - **Greater London:** liegt am 0-Meridian → **30N und 31N** (beide Zonen ziehen).
  3. Nur die Kacheln laden, die die jeweilige Region-Bounding-Box schneiden (`gcloud storage cp` bzw. gezieltes Listing), nach `data/raw/alphaearth/2025/`.
  4. Alternativen falls GCS klemmt: EE-ImageCollection `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` + `Export.image.toDrive/toCloudStorage` (dein ursprünglich notierter Weg); oder AWS Registry of Open Data Mirror (`aef-source`, Source Cooperative).
- **Erwarteter Output:** Lokale COGs, 64 Bänder, 10 m, die beide Regionen abdecken.
- **Abnahme:** `rasterio` liest Datei, `count == 64`, CRS = passende UTM-Zone, Auflösung 10 m. Attribution im Thesis-Datenkapitel vermerken: „The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."

### 2.2 AlphaEarth Spatial Join
- **Ziel:** Analog `spatial_join.py` die POI-Geometrien gegen AlphaEarth schneiden.
- **Umsetzung:** `spatial_join.py` wiederverwenden/parametrisieren (Backbone als Argument). **Kein** z-score (AlphaEarth ist unit-norm). Gleiche Dedup-Logik (`poi_id`) wie bei TESSERA — der früher gefixte Bug (Multi-Row-POI verschiebt Geometrie-Zuordnung) darf nicht zurückkommen.
- **Erwarteter Output:** `pairs_alphaearth.parquet`, Embedding-Dim 64, NaN-Count 0.
- **Abnahme:** Gleiche `poi_id`-Abdeckung wie TESSERA (Differenzen nur durch Raster-Extent erklärbar); `validate_pairs.py` grün.

### 2.3 AlphaEarth-Textpaare
- **Ziel:** Schritt 1.2 für AlphaEarth spiegeln.
- **Erwarteter Output:** `pairs_alphaearth_text.parquet` (analog 1.2, Embedding-Dim 64).
- **Abnahme:** Wie 1.2.

---

## 3. Phase 3 — Modell & Training (prüfen + verdrahten)

> Laut Stand existieren Heads, SigLIP-Loss, Dataset/Dataloader und Trainingsskript bereits. Hier **nicht neu schreiben**, sondern gegen die fixen Constraints prüfen und Lücken schließen.

### 3.1 Modellkomponenten verifizieren
- **Check:** Heads = `Linear → GELU → Dropout → Linear → L2-norm`, `d=256`. SigLIP-Loss = pairwise sigmoid mit learnbarer Temp **und** Bias, Multi-Positives ohne Masking. Backbones eingefroren. Zwei getrennte Modelle pro Backbone.
- **Abnahme:** Kurzer Assert-Report; Abweichung von Constraints → melden, nicht eigenmächtig „verbessern".

### 3.2 Trainingsskript vervollständigen
- **Ziel:** Config-gesteuerter Lauf pro Backbone.
- **Umsetzung:** Backbone-Auswahl per Config; **coverage-gewichtetes** Sampling; learnbare Temp+Bias im Optimizer; Train/Val-Split; Checkpointing; Logging (Loss, LR, Temp, Bias). Nur Head-Parameter im Optimizer.
- **Erwarteter Output:** `train.py` lauffähig mit einer Config-Datei.
- **Abnahme:** Sanity-Overfit auf 200 Paaren → Loss geht klar runter, Retrieval-Recall@1 auf demselben Subset → nahe 1.0.

### 3.3 Training
- **Umsetzung:** Erst London-Mini-Region (Smoke-Test end-to-end), dann Niedersachsen. Beide Tower getrennt (TESSERA, AlphaEarth).
- **Erwarteter Output:** Checkpoints je Backbone/Region unter `checkpoints/`.
- **Abnahme:** Val-Loss sinkt und stabilisiert; keine NaNs in Temp/Bias.

---

## 4. Phase 4 — Evaluation (TODOs mit echten Daten füllen)

Grundprinzip überall: **linear probe / zero-shot auf frozen Embeddings**, gemessen als **Delta raw-Baseline ↔ enriched**. Ground Truth Land Use: Copernicus Urban Atlas.

### 4.1 Land-Use-Klassifikation
- **Metriken:** Macro-F1, OA. Linear Probe auf enriched vs. raw Backbone-Embeddings, Ground Truth Urban Atlas.
- **Abnahme:** Tabelle raw vs. enriched je Backbone; Splits dokumentiert.

### 4.2 Sozioökonomische Verteilung
- **Metrik:** KL-Divergenz gegen Referenz. London: Census 2021 / IMD. Niedersachsen: Zensus 2022.
- **Abnahme:** KL-Werte raw vs. enriched, Mapping der Aggregationseinheiten dokumentiert.

### 4.3 Text-zu-EE-Retrieval (strukturell stärkstes Ergebnis)
- **Metriken:** Recall@k, mAP@k, Zero-Shot-F1. Raw-Baseline hat **kein** Text-Interface → dort nur enriched sinnvoll; das ist genau der Punkt.
- **Abnahme:** Recall@{1,5,10}, mAP@k, Zero-Shot-F1 als Tabelle; Query-Set dokumentiert.

---

## 5. Phase 5 — Ablationen

- **V1:** alle semantischen Klassen.
- **V2:** IDF-gefiltert (Rarität ist ein **statistisches** Kriterium hier — **nicht** Klassen aus V1 entfernen).
- **V3:** nur ground-truth-assoziierte Klassen.
- **Umsetzung:** Jede Config durch Phase 4 laufen lassen; gecachte Text-Embeddings wiederverwenden. **Nicht** nach Satelliten-Detektierbarkeit filtern (würde den Kernbeitrag der Thesis untergraben).
- **Erwarteter Output:** Ergebnismatrix V1/V2/V3 × {LandUse, Sozio, Retrieval} × {TESSERA, AlphaEarth}, jeweils raw vs. enriched.

---

## 6. Phase 6 — DGX Spark Full Run

- **Umsetzung:** Blackwell-GPU-Torch gesondert installieren (aarch64/GB10, Special Handling). Full-Pipeline: komplettes London + Niedersachsen. Workflow `git pull` → Lauf → `git add/commit/push`; große Datenfiles via `.gitignore` ausgeschlossen. User `minorai` ist **nicht** in sudoers → keine sudo-Schritte einplanen.
- **Abnahme:** Alle Checkpoints + Eval-Tabellen erzeugt, reproduzierbar über Config.

---

## ⚠ Nicht-Code — an Kai eskalieren (nicht selbst entscheiden)
- HILUCS-Crosswalk Zweifelsfälle: `allotments, meadow, grass, cemetery, military, garages`.
- Textsprache der Wikidata-Beschreibungen (de vs. en), falls nicht eindeutig.
- Jede sonstige methodische Grenzentscheidung → dokumentieren + fragen, nicht annehmen.