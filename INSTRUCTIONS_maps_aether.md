# INSTRUCTIONS: AETHER-Vergleichskarten (Grünflächen, Flughäfen)

Ziel: Similarity-Karten über ganz London für zwei Text-Queries, je
Backbone, plus Kontroll-Query und Referenzkarten. Nur Karten, keine
Metriken (AP/Recall/Bootstrap kommen später).

## Grundlagen (nicht neu bauen)
- 256-d-Felder: data/interim/sdm_projected_tessera_v1dedup/tessera_armc/
  und data/interim/sdm_projected_ae_v1dedup/ae_armc/ (kanonische
  Dedup-Checkpoints, in Punkt 1 verifiziert)
- Masken: data/interim/text_query_masks/{ae,tessera}/*_mask.tif
- Referenzen: data/interim/greencover_2024_valid.parquet, Flughäfen aus
  greater_london-latest.osm.pbf (aeroway=aerodrome, alle 7)
- Vorhandener Code: src/eval/text_query/{tiles,masks,verify_projected_fields}.py

## Queries (Wortlaut exakt so)
- green: "Green Park and Natural Reserve."
- airport: "Airport and Transportation Hub."
- control: "desert"
- cbd: "Central Business District" (ergänzt 2026-09-23, Jakob — Phasen 1/2/4
  für Phasen 1-3 bereits abgeschlossenen Stand nachgezogen, siehe MANIFEST.md.
  Keine Referenzkarte und keine Pflichtprüfung/Kennzahl für diese Query — dafür
  gibt es kein Layer. Skalengrenzen wie bei den übrigen Queries: 1./99. Perzentil
  über die Vereinigung beider Backbone-Felder dieser Query.)

## Phase 1: Query-Encoder
src/eval/text_query/encode.py. Query mit demselben Sentence-Transformer
kodieren wie im Training, dann durch den Text-Projektor des jeweiligen
Checkpoints, dann L2-normieren.
**Korrektur (2026-09-23, Rueckfrage Jakob/Kai):** ckpt['cfg'] enthaelt
KEINE Encoder-Kennung (nur text_dim=1024, die Dimension nach MRL-
Truncation) - der urspruengliche Satz "Encoder-Name und Projektor
kommen aus ckpt['cfg']" traf nicht zu. Der Encoder-Name ist eine
Modulkonstante in compute_text_embeddings.py (MODEL_NAME/MRL_DIM,
"Qwen/Qwen3-Embedding-4B"/1024) - importieren statt duplizieren,
identisch zum bestehenden Weg in run_class_retrieval.py. Der Projektor
(model.text_head) kommt weiterhin korrekt aus dem jeweiligen
Checkpoint. Zusaetzliche Pflichtpruefung: ckpt['cfg']['text_dim'] muss
mit MRL_DIM uebereinstimmen, sonst klarer Abbruch.
Prüfung: Nimm einen Trainings-Text aus pixels_*_dedup.parquet, kodiere
ihn neu und vergleiche mit dem im Training verwendeten Text-Embedding.
Kosinus muss **>= 0,999** sein (Toleranz statt exakt 1,0 — Ursache:
Batch-Zusammensetzung beim Encoding beeinflusst das Ergebnis eines
4B-Transformers minimal, siehe MANIFEST.md; für ein Ranking über
Millionen Pixel bedeutungslos, keine Nachbildung der ursprünglichen
Batch-Zusammensetzung). Gemessen (2026-09-23): TESSERA 0,99982113,
AE 0,99986976 — beide bestanden. Ergebnis berichten.
STOPP nach Phase 1.

## Phase 2: Similarity-Felder
src/eval/text_query/similarity.py. Kosinus zwischen Query-Embedding und
jedem 256-d-Pixel, kachelweise, Ergebnis als GeoTIFF float32 pro Kachel
unter outputs/text_query/similarity/{backbone}/{query}/.
Ungültige Pixel (Maske) als NoData, nicht als 0.
Für beide Backbones, alle drei Queries.
Berichten: Wertebereich, Anteil NoData, Rechenzeit.
STOPP nach Phase 2.

## Phase 3: Referenzkarten
Rasterisierte Referenz als GeoTIFF (uint8, 1 = positiv), Regel >50 %
Flächenanteil, exactextract wie in Nachtrag 1, Arm-C-Grid, je Backbone:
- greencover, airports
Unter outputs/text_query/reference/{backbone}/.
Pixelzahlen müssen mit Nachtrag 1 übereinstimmen (7.825.648 / 144.030
TESSERA, 7.823.743 / 144.023 AE) — hart prüfen, Abweichung melden.

## Phase 4: Karten
src/eval/text_query/maps.py, Ausgabe PNG (300 dpi) unter
outputs/text_query/maps/.
- cmap "magma", Colorbar rechts, beschriftet "cosine similarity"
- Skalengrenzen je Query FEST und für beide Backbones identisch:
  1.- und 99.-Perzentil über die Vereinigung beider Backbone-Felder
  dieser Query. Die verwendeten Grenzen in den Dateinamen-Metadaten oder
  einer begleitenden CSV festhalten.
- Ausserhalb AOI transparent, Umriss der Greater-London-Grenze dünn
  darüber
- Gleiche Bildgrösse und Ausschnitt für alle Karten
- Referenzkarten separat: positive Pixel grün auf weiss, gleicher
  Ausschnitt

Erwartete Dateien: 6 Similarity-Karten (2 Backbones × 3 Queries) und
4 Referenzkarten (2 Backbones × 2 Referenzen). Nach Ergänzung der
vierten Query (cbd, 2026-09-23): 8 Similarity-Karten (2 Backbones ×
4 Queries), weiterhin 4 Referenzkarten (cbd hat keine).

## Nicht tun
- Keine AP, kein Recall, kein Bootstrap
- Keine Prototyp-Baseline
- Keine Hotspot-Auswahl
- Nichts an bestehenden Feldern, Masken oder Referenzdateien ändern

Am Ende: Working Tree sauber, committen, Dateiliste berichten.