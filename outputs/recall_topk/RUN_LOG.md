# RUN_LOG — Recall@Top-k% (AETHER-Stil), Nachtlauf

Instruktion: `INSTRUCTIONS_recall_topk_overnight.md` (als Chat-Paste übergeben, liegt nicht als Datei im Repo).
Branch: `recall-topk` (von `main` @ c3f8f4f, `git pull`: already up to date).
Nicht committete Fremdänderungen im Working Tree beim Start (NICHT angefasst, NICHT mitcommittet):
`data/processed/MANIFEST.md` (modified), `catboost_info/`, `eval_landuse_heads.py`,
`logs/landuse_probe_heads_full_run.log`, `run_landuse_probe_heads.py` (untracked).

---

## Phase 0 — Bestandsaufnahme

Start: 2026-09-30 00:00 (CEST)

**Gate:** `python src/verify_provenance.py` → OK (pixels_ae↔landuse_pairs_ae 1.915.903 Pixel,
pixels_tessera↔landuse_pairs_tessera 1.945.979 Pixel, split identisch). Hinweis: das Skript
registriert nur die NICHT-dedup-Tabellen; die hier genutzten `pixels_{ae,tessera}_dedup.parquet`
prüft es nicht. Ergänzend prüft `eval/recall_topk.py` vor jeder Rechnung selbst, dass die aus
`splits_london.csv` rekonstruierte Zellzuordnung den `split` jedes GT-Pixels in der Dedup-Tabelle
exakt reproduziert (Abbruch bei Abweichung).

**1. 256-d-Felder für ganz London** (wiederverwendet, nicht neu projiziert):

| Backbone | Feld | Checkpoint | Provenienz |
|---|---|---|---|
| AlphaEarth | `data/interim/sdm_projected_ae_v1dedup/ae_armc/*_proj256.tif` (2 Kacheln) | `checkpoints/ae_v1_dedup/best.pt` (Epoche 32) | MANIFEST „Arm-C-Feld-Provenienz (Nachtrag 2026-09-22)“: Kosinus 1,000000 gegen Dedup-Checkpoint; mtime Feld 21.9. 17:29 > best.pt 21.9. 15:50 |
| TESSERA | `data/interim/sdm_projected_tessera_v1dedup/tessera_armc/*_proj256.tif` (50 Kacheln) | `checkpoints/tessera_v1_dedup/best.pt` (Epoche 48) | dito; best.pt 19.9. 02:05 |

Projektion erfolgte pixelweise vor jeder Aggregation (build_sdm_projected_field.py). Die alten Felder
unter `data/interim/sdm_projected/` (Nicht-Dedup-Checkpoints) werden NICHT verwendet.
Zusätzlich wiederverwendet: Eindeutigkeits+AOI-Polygon-Masken `data/interim/text_query_masks/{ae,tessera}/`
(INSTRUCTIONS_text_query.md Phase 2, ~15,95 Mio gültige Pixel je Backbone laut MANIFEST).
Die bestehenden Similarity-GeoTIFFs unter `outputs/text_query/similarity/` betreffen andere Queries
und werden nicht benutzt. Eine eigene Stichproben-Reprojektion (Kosinus ≥ 0,9999) läuft als
Preflight in `eval/recall_topk.py`.

**2. Prototyp-Baseline-Code:** existiert NICHT (INSTRUCTIONS_text_query.md Phase 7 ist „GESPERRT“,
grep nach prototyp/prototype im Repo ohne Treffer). → Neuimplementierung nach Definition Phase 2
dieser Instruktion (Mittel der ROHEN Embeddings aller GT-Pixel in Train-Zellen; AE L2-normiert,
TESSERA z-standardisiert mit `data/processed/norm_stats_tessera_pixels_dedup.json`).

**3. Weitere Befunde:**
- Klassentabelle: `wikidata tags/osm_wikidata_keys.xlsx` (config.CLASS_TABLE), 780 Klassen (key=value).
- Trainingspaare pro Pixel: `data/processed/pixels_{ae,tessera}_dedup.parquet` (Spalten tile,row,col,
  poi_ids,split; Rasterkoordinaten der ROH-Kachel). Split-Verteilung Zellen (`splits_london.csv`):
  1454 train / 182 val / 182 test.
- POI-Tags: `data/interim/pois_london_full_filtered.parquet` (Long-Format poi_id,key,value).
- Kein Top-Level-Verzeichnis `eval/` vorhanden (bisheriger Eval-Code liegt unter `src/eval/`).
  Instruktion nennt explizit `eval/recall_topk.py` → dort angelegt.

Ende: 2026-09-30 00:03. Ergebnis: ok.

---

## Phase 1 — Klassenzuordnung

Start: 2026-09-30 00:03 · Ende: 00:05 — Commit 87dc23a
"recall_topk: pre-registered AETHER category mapping" (nur YAML + feasibility.md).

Zuordnung (`queries/aether_categories.yaml`) wurde geschrieben, BEVOR irgendeine Instanz- oder
Pixelzahl ausgezählt wurde; Maßstab nur OSM-Wiki-Kernbedeutung/Funktion des Ortes.
- Art and Antiques: shop=art, shop=antiques (unsicher: tourism=gallery, shop=collector, shop=frame, shop=craft, shop=pottery)
- Religious Organisations: amenity=place_of_worship, amenity=monastery, office=religion (unsicher: 11× building=<religiöse Bauform>, 4× historic=*, shop=religion)
- Medical Equipment, Supplies and Pharmaceuticals: shop=medical_supply, amenity=pharmacy, healthcare=pharmacy (unsicher: shop=chemist, hearing_aids, optician, nutrition_supplements, herbalist)
- Sports Clubs and Associations: leer (club=* nicht in Klassentabelle; unsicher: Sportanlagen, office=association, sport=*)
- Import and Export Services: leer (unsicher: office=logistics, shop=wholesale, office=courier)

Machbarkeit (`feasibility.md`, `feasibility.csv`): N = 15.954.033 (AE) / 15.954.158 (TESSERA) Kandidatenpixel
(identisch mit den MANIFEST-Zahlen der Text-Query-Masken). Status beide Backbones: art_antiques ok
(Test 351/358 Pixel, 16 Zellen), religious_orgs ok, medical_supplies ok, sports_clubs no_gt, import_export no_gt.
0 GT-Pixel lagen außerhalb des Kandidatenraums.

Ergebnis: ok.

---

## Phase 2 — Implementierung `eval/recall_topk.py`

Start: 00:05 · Ende: 00:08 (Commit fd7fd83)

- Kandidatenraum: Text-Query-Maske (Eindeutigkeit + AOI-Polygon) UND endliches Arm-C-Feld; N wie oben.
- Top-Menge: K = ceil(k·N) per `np.argpartition`, k ∈ {0.01, 0.05, 0.10}; Schwelle immer über ganz London.
- GT: siehe Problemtabelle (Dedup-Trainingspixeltabelle, kein Puffer, Vereinigung der osm_classes).
- Arme: T (Query → `src/eval/text_query/encode.py::encode_queries`, identischer Encoder-/MRL-/Normierungs-/
  Projektor-Pfad wie im Training, Kosinus gegen Arm-C-Feld), P (Mittel der rohen Train-GT-Pixel, AE dequantisiert
  + L2, TESSERA z-Score mit `norm_stats_tessera_pixels_dedup.json`, Kosinus im Rohraum), R = K/N.
- Preflight je Backbone: 300 zufällige Kandidatenpixel aus Roh neu projiziert (Dedup-Checkpoint), Kosinus zum
  gespeicherten Arm-C-Wert ≥ 0,9999, sonst Abbruch.
- Zell-Split-Provenienz: rekonstruierte Zelle→Split (aggregate_pixel_pairs.compute_pixel_cells) muss die split-Spalte
  aller GT-Pixel exakt reproduzieren, sonst Abbruch.
- CIs: Zellebenen-Bootstrap, B=1000, Seed 0, gepaarte Differenz T−P mit denselben Gewichten.

**Selbsttest (synthetisch, N=2 Mio, |GT|=100.000, `selftest_synthetic.txt`): PASSED**
- Zufallsscores: Recall 0,01068 / 0,05001 / 0,10070 bei K/N = 0,01 / 0,05 / 0,10 (alle innerhalb 5 SD).
- Orakel: 0,2 / 1,0 / 1,0 = min(1, K/|GT|) exakt; Zusatzfall |GT| > K: 0,25 = K/|GT| exakt.
- Zusätzlich läuft derselbe Zufalls-/Orakel-Test im echten Lauf auf dem echten Kandidatenraum mit der echten GT
  jeder Kategorie (`selftest_real_{bb}.txt`); Abbruch bei Fehlschlag.

Ergebnis: ok.

---

## Phase 3 — Lauf

Start: 00:08 · Ende: 00:17

- AlphaEarth (`run_ae.log`): Gate ok, Preflight-Reprojektion min. Kosinus 0,9999999 (≥ 0,9999), Echtdaten-Selbsttest
  (Zufall ≈ K/N, Orakel = min(1,K/|GT|)) für alle 3 Kategorien mit GT und alle k PASS, 196 s. ok.
- TESSERA (`run_tessera.log`): Gate ok, Preflight 1,000000, Echtdaten-Selbsttest PASS, 203 s. ok.
- Prototyp-Stützpixel (Train-GT): AE 808 / 30.949 / 2.637, TESSERA siehe `run_meta_tessera.json`.
- `finalize`: `recall_topk.csv`, `diffs_recall_topk.csv`, `table_aether_style.md`, 10 Karten `maps/`,
  Skalengrenzen `map_scale_limits.csv` (1./99. Perzentil über die Vereinigung beider Backbones je Query). ok.
- Freier Speicher vor Feldschreiben: 2,3 TB (> 50 GB). Felder `fields/` (1,4 GB: Similarity-T + GT-GeoTIFFs je
  Kachel, `scoresT_*.npy`) werden NICHT committet (Größe).
- Konzentrationsbefund (nur markiert, nichts entfernt): Test-GT von art_antiques liegt zu 73 % (AE) / 74 % (TESSERA)
  in EINER 1-km-Zelle, Top-3-Zellen 87 % / 89 % (16 Testzellen). Die Test-Recall-Werte dieser Kategorie spiegeln
  daher überwiegend einen Ort; die Bootstrap-CIs sind entsprechend breit/schief. religious_orgs: größte Zelle 5 %,
  medical_supplies: 8 %.

Ergebnis: ok.

---

## Probleme / Entscheidungen

Format: Problem | Ursache | Entscheidung | verworfene Alternative | betroffene Dateien

| Problem | Ursache | Entscheidung | verworfene Alternative | betroffene Dateien |
|---|---|---|---|---|
| Kein `eval/`-Verzeichnis im Repo | bisheriger Eval-Code unter `src/eval/` | `eval/recall_topk.py` wie in der Instruktion wörtlich angelegt (erlaubter Schreibpfad) | `src/eval/recall_topk/` (Konvention, aber nicht erlaubter Schreibpfad) | eval/recall_topk.py |
| verify_provenance.py prüft die Dedup-Tabellen nicht | PAIRS-Liste nur mit Nicht-Dedup-Dateien; Datei darf hier nicht geändert werden | Gate trotzdem vor jeder Rechnung laufen lassen + eigene harte Prüfung (Zell-Split rekonstruiert == split-Spalte der Dedup-Tabelle) im Skript | verify_provenance.py erweitern (nicht erlaubter Schreibpfad) | eval/recall_topk.py |
| Prototyp-Code fehlt | Phase 7 der Text-Query-Instruktion gesperrt | Neuimplementierung nach Phase 2 dieser Instruktion | Phase-7-Arbeitsvorschlag (Hälften-Split der Instanzen) – widerspricht der hier vorgegebenen Definition | eval/recall_topk.py |
| GT-Quelle nicht eindeutig benannt („gleiche Regel wie bei den Trainingspaaren“) | Paarbildung = spatial_join.py/exactextract → aggregate_pixel_pairs.py | GT direkt aus `pixels_{bb}_dedup.parquet` (Spalte poi_ids = alle POIs, die das Pixel schneiden); Pixel positiv, wenn ein POI mit ≥1 Tag aus osm_classes darunter ist. Enthält damit implizit auch den Ausschluss von POIs ohne Text (Fall D) und die Dedup-/AOI-Filter der Trainingspixel | GT neu per exactextract aus POI-Geometrien rastern (INSTRUCTIONS_text_query gt.py-Weg, dort mit 25-m-Puffer – hier explizit „kein Puffer“) | eval/recall_topk.py |
| „n_pois“ mehrdeutig | POIs der Klasse in London vs. POIs mit ≥1 GT-Pixel | CSV-Spalte n_pois = POIs mit ≥1 GT-Pixel im Kandidatenraum; feasibility.md zeigt beide Zahlen | nur Tabellenzahl | recall_topk.csv, feasibility.md |
| Split „val“ | Instruktion nennt nur train/test/all | val-Pixel gehen nur in „all“ ein, kein eigener Block | eigener val-Block | recall_topk.csv |
| Welche Zellen werden im Bootstrap gezogen? | Zellen ohne GT tragen 0/0 bei | Resampling über die Zellen mit ≥1 GT-Pixel der Kategorie im jeweiligen Split (n_gt_cells), mit Zurücklegen, B=1000, Seed 0; identische Gewichte für T und P und alle k (gepaart) | Resampling über alle Zellen des Splits (fügt nur Nullzellen hinzu, ändert Gewichtung der GT-Zellzahl) | eval/recall_topk.py |
| feasibility.csv | Hilfsdatei zum Report | in Phase 1 nicht committet (Vorgabe „nur YAML + Report“), wird mit Phase 3 committet | – | outputs/recall_topk/feasibility.csv |
| Karten-GT bei 50-m-Darstellungsgitter | Punkt-POIs (1 Pixel) verschwinden bei nearest-Resampling | GT-Panel mit Resampling.max (50-m-Zelle grün, wenn ≥1 GT-Pixel); nur Darstellung, Kennzahlen auf nativem Raster | nearest wie bei Referenzkarten der AETHER-Karten | eval/recall_topk.py, maps/ |
| Kartenlayout | Instruktion: Similarity (T) neben GT | eine PNG je Kategorie×Backbone mit 2 Panels; no_gt-Kategorien: GT-Panel leer mit Vermerk | getrennte PNGs | maps/ |
| art_antiques Test-GT konzentriert (73–74 % in einer Zelle) | wenige, räumlich geballte Test-POIs | Status bleibt regelgemäß „ok“, Befund in Log/Report markiert | Kategorie als thin umdeklarieren (nicht durch Regel gedeckt) | REPORT.md |

---

## Phase 4 — Abschlussbericht (nachgetragen unter der Problemtabelle)

Start: 00:17 · Ende: 00:20. `REPORT.md` geschrieben, Commit "recall_topk: overnight report", Push des Branches `recall-topk`. Ergebnis: ok.

---

## Nachtrag v2 (2026-09-30 abends) — GT-Zuordnung nach Vorgabe Jakob

Nach Vorlage der vorregistrierten Ergebnisse hat Jakob eine andere GT-Zuordnung festgelegt
(`queries/aether_categories_v2_jakob.yaml`): Art = shop=art, shop=antiques (unverändert); Religious = nur
office=religion; Medical = nur shop=medical_supply; Sports Clubs = club=sport (außerhalb der Klassentabelle);
Import/Export = office=logistics. Nur als Evaluations-GT dieser Auswertung, keine Änderung an Klassentabelle/Training.
Die vorregistrierten Ergebnisse in `outputs/recall_topk/` bleiben unverändert; v2 liegt in `outputs/recall_topk/v2_jakob/`.

Code: `eval/recall_topk.py` um `extra_osm_tags` (Extraktion aus `config.PBF_LONDON` per pyrosm, Rasterung mit
`spatial_join.join_pois_to_tiles` = Paarbildungsregel) sowie `--categories/--out` erweitert. Konsistenzcheck: art_antiques
in v2 zahlengleich mit v1. Gate, Preflight (Kosinus 1,000000) und Echtdaten-Selbsttest beide Backbones PASS.

| Problem | Ursache | Entscheidung | verworfene Alternative | betroffene Dateien |
|---|---|---|---|---|
| Jakobs Tabelle nahm an, `shop` sei kein aktiver Key | veralteter Stand (früher 4 Keys) | Korrigiert gemeldet: shop ist aktiv, shop=art/antiques/medical_supply sind in der Pipeline; Zuordnung trotzdem wie vorgegeben | – | – |
| club=sport nicht in den Trainingspixeln | club ist kein aktiver Key | eigene Rasterung mit der Paarbildungsregel gegen dieselben Rohkacheln; Pixel außerhalb des Kandidatenraums (Kachelüberlappung/AOI) verworfen: 1266 AE / 2317 TESSERA; 322 von 328 Objekten mit GT-Pixel; 0 Pixel ohne Split-Zelle | GT aus Pixeltabelle (unmöglich) | eval/recall_topk.py |
| religious_orgs, medical_supplies, import_export in v2 `thin` (Test 0–3 Pixel) | nur 13–25 POIs je Klasse in London | nach Regel markiert, nichts entfernt; Test-Werte nicht aussagekräftig, all-Split zusätzlich berichtet (Prototyp dort nicht unabhängig) | – | v2_jakob/ |
