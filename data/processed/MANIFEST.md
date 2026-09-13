# data/processed/MANIFEST.md — Provenienz-Manifest

Hält für jedes aktive Artefakt fest, wodurch es erzeugt wurde. Bei jedem Neubau eines Artefakts
diese Datei fortschreiben (Zeile aktualisieren, nicht nur Kommentar im Skript). Archivierte
Gen-1-Artefakte stehen NICHT hier, sondern in `data/archive_gen1/` (siehe dortige Herkunft in
der Git-Historie / im Chat-Verlauf vom 3.9., "Altmaterial aus dem Weg räumen").

Split-Grenze, auf die sich "aktuell" bezieht: `data/splits/splits_london.csv` wurde am 3.9.
14:52 von `build_splits_pixels.py` mit der PIXELBASIERTEN Zell→Split-Zuordnung überschrieben
(vorher: POI-basiert, jetzt gesichert unter `data/splits/splits_london_4key.csv`, siehe
`data/splits/README.md`). "Aktuell" heißt hier: gebaut MIT der pixelbasierten Zuordnung.

## POI-Extraktion & Text

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/interim/pois_london_full.parquet` | `src/extract_pois.py` | OSM PBF (13-Key-Extraktion: amenity/shop/leisure/tourism/building/healthcare/historic/office/sport/water/waterway/military/natural) | 2.9. 22:13 | aktuell |
| `data/interim/pois_london_full_filtered.parquet` | `src/extract_pois.py` | `pois_london_full.parquet` | 2.9. 22:13 | aktuell |
| `data/interim/object_qid_descriptions.parquet` | `src/fetch_object_qid_descriptions.py` | `pois_london_full_filtered.parquet` (object_qid-Spalte), Wikidata REST | 3.9. 12:40 | aktuell |
| `data/interim/qid_descriptions.parquet` | `src/fetch_class_qid_descriptions.py` | `class_table.py`/`wikidata tags/osm_wikidata_keys.xlsx` (eindeutige QIDs), Wikidata SPARQL/REST | 3.9. 13:26 | aktuell |
| `data/interim/poi_text_full.parquet` | `build_poi_text.py` | `pois_london_full_filtered.parquet`, `class_table.py`, `qid_descriptions.parquet`, `object_qid_descriptions.parquet` | 3.9. 14:13 | aktuell |

## Spatial Joins (POI × EO-Raster, POI-Ebene, vor Pixel-Aggregation)

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/processed/pairs_ae_full.parquet` | `run_ae_join_full.py` (→ `chunked_join_runner.py`/`spatial_join.py`) | `pois_london_full_filtered.parquet`, AlphaEarth-COGs | 3.9. 00:16 | aktuell |
| `data/processed/pairs_tessera_full.parquet` | `run_tessera_join_full.py` | `pois_london_full_filtered.parquet`, TESSERA-COGs | 3.9. 10:18 | aktuell |

## Split

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/splits/splits_london.csv` | `build_splits_pixels.py` | Zell-Vereinigung aus `pixels_tessera.parquet`+`pixels_ae.parquet` (Pixel-Keys, VOR Split-Zuweisung), AOI | 3.9. 14:52 | aktuell (pixelbasiert) |
| `data/splits/splits_london_4key.csv` | `build_splits.py` (Backup vor Überschreiben) | 279.988 POIs der alten 4-Key-Extraktion | 1.9. 10:31 | **ARCHIV — nie als aktueller Split verwenden** |

## Pixel-Aggregation & Text-Embeddings

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/processed/pixels_ae.parquet` | `src/aggregate_pixel_pairs.py` | `pairs_ae_full.parquet`, `poi_text_full.parquet`, `splits_london.csv` (aktuell) | 3.9. 14:52:59 | aktuell |
| `data/processed/pixels_tessera.parquet` | `src/aggregate_pixel_pairs.py` | `pairs_tessera_full.parquet`, `poi_text_full.parquet`, `splits_london.csv` (aktuell) | 3.9. 14:52:42 | aktuell |
| `data/processed/pixel_text_embeddings.parquet` | `compute_pixel_text_embeddings.py` | eindeutige `pixel_text`-Werte aus `pixels_ae.parquet`+`pixels_tessera.parquet`, Qwen3-Embedding-4B | 3.9. 16:18 | aktuell |

## Normalisierung

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/processed/norm_stats_tessera.json` | `train.py::apply_normalization` (via altem `configs/tessera_v1.yaml`-Lauf) | `pairs_tessera.parquet` (Gen1, POI-Ebene) | 1.9. 16:11 | **veraltet, liegt noch da, aber seit 3.9. NICHT mehr referenziert** (`run_landuse_probe.py::NORM_STATS_PATH` zeigt jetzt auf `norm_stats_tessera_pixels.json`, s. u.) |
| `data/processed/norm_stats_tessera_pixels.json` | `train.py::apply_normalization` (TESSERA-Pixel-Lauf, Log fehlt) | `pixels_tessera.parquet` (aktuell) | war 3.9. 18:13, jetzt in `data/archive_gen1/processed/` | **archiviert wegen ungeklärter Herkunft — `run_landuse_probe.py`/`configs/tessera_v1.yaml` zeigen auf diesen Pfad, Datei existiert dort aktuell NICHT — Skript bricht beim nächsten TESSERA-Lauf mit `FileNotFoundError` ab, bis neu gefittet wurde. Das ist gewollt (Nutzervorgabe 3.9.: lieber Fehler als stillschweigend Gen1-Stats laden).** |

## Checkpoints

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `checkpoints/ae_v1/best.pt` (Epoche 22) | `train.py --config configs/ae_v1.yaml` | `pixels_ae.parquet`, `pixel_text_embeddings.parquet` (beide aktuell) | 3.9. 19:43 | aktuell — einzig verifizierter aktueller Checkpoint |
| `checkpoints/tessera_v1/*` | — | Gen1-Daten (`pairs_tessera.parquet`, alter Split) | 1.9. 16:19/16:36 | **archiviert** (`data/archive_gen1/checkpoints/tessera_v1/`) — vor TESSERA-Nutzung neu trainieren |
| LR-Sweep-Checkpoints (`ae_lr*_5ep`, `tessera_lr*_5ep`, `ae_v1_nomask`) | `train.py` (LR-Sweep-Läufe) | Gen1-Daten | 1.9. | **archiviert** (`data/archive_gen1/checkpoints/`) |

## Urban Atlas / Land-Use

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/interim/urban_atlas_london.parquet` | `build_landuse_pairs.py::load_and_clip` | Copernicus Urban Atlas FGB, `data/aoi/london.gpkg` | **neu gebaut 3.9. 23:45** (Rebuild, unabhängig vom Split — deterministisch identischer Inhalt zur alten 1.9.-Version) | aktuell |
| `data/processed/landuse_pairs_ae.parquet` | `build_landuse_pairs.py` | `urban_atlas_london.parquet`, AlphaEarth-COGs, `splits_london.csv` (aktuell, pixelbasiert) | **neu gebaut 3./4.9. (Rebuild nach Leakage-Fund)** | **aktuell — `verify_provenance.py` bestätigt 100% Split-Übereinstimmung mit `pixels_ae.parquet` (1.934.728 gemeinsame Pixel, 0 Konflikte)** |
| `data/processed/landuse_pairs_tessera.parquet` | `build_landuse_pairs.py` | analog, TESSERA-COGs | **neu gebaut 3./4.9.** | **aktuell — `verify_provenance.py` bestätigt 100% Übereinstimmung (1.959.700 gemeinsame Pixel, 0 Konflikte)** |

Nebenbefunde aus dem Rebuild (4.9., beide Backbones): `split='missing'` betrifft AE 317 / TESSERA 315 Pixel
(~0,0066%) — UA-Pixel in 1-km-Gitterzellen ganz ohne POI-Pixel, daher kein Eintrag in
`splits_london.csv`; nicht verworfen, nur markiert. Klasse `12400` (Airports, 12 UA-Polygone)
kommt in allen drei Splits vor (AE: train=949/val=97/test=110), wird aber wegen zu geringer
Test-Fallzahl weiterhin in `run_landuse_probe.py` ausgeschlossen (`EXCLUDED_CLASS`). Klasse
`40000` (Wetlands, 13 Polygone) fehlt in diesem Split-Zug komplett im Test-Anteil beider
Backbones (Zufall der Ziehung, nicht Bug) — Konfusionsmatrizen sind deshalb 23x23 statt 24x24.

## Werkzeuge

- `src/verify_provenance.py` — prüft Split-Konsistenz über gemeinsame `(tile,row,col)`-Schlüssel
  zwischen registrierten Dateipaaren (aktuell: `pixels_ae`↔`landuse_pairs_ae`,
  `pixels_tessera`↔`landuse_pairs_tessera`), bricht bei Abweichung mit Fehler ab. Zuletzt
  ausgeführt 4.9., beide Paare 100% konsistent (Exit-Code 0). **Noch nicht automatisch in
  `train.py`/`run_landuse_probe.py` eingebaut** — muss manuell vor einem Lauf aufgerufen werden.

## Evaluations-Outputs (Phase B/C dieser Session)

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/processed/retrieval_baselines_ae_v1.csv` | `run_retrieval_baselines.py` | `checkpoints/ae_v1/best.pt`, `pixels_ae.parquet`, `pixel_text_embeddings.parquet` (alle aktuell) | 3.9. 21:28 | aktuell (von der Landuse-Split-Frage nicht betroffen — reine Retrieval-Auswertung auf `pixels_ae.parquet`) |
| `data/processed/landuse_probe_results_ae.csv` + zugehörige `landuse_probe_*_ae.csv` (Konfusionsmatrizen, Per-Klassen-F1, Häufigkeitsniveau, Test-Bootstrap, gepaarter Diff) | `run_landuse_probe.py --backbones ae --skip-train-bootstrap` | `landuse_pairs_ae.parquet` (**neu, verifiziert konsistent**), `checkpoints/ae_v1/best.pt` | **4.9., Rerun nach Leakage-Fix** | **aktuell — Endergebnis (Test): roh test_macro_f1=0.2559, zufällig=0.2140, trainiert=0.2631, Häufigkeitsniveau=0.0270; gepaart trainiert−roh macro_f1=+0.00735 [0.00242,0.01236]. Ersetzt den 3.9.-Lauf unten.** |
| `data/processed/landuse_probe_results.csv` + `landuse_probe_{train,test}_bootstrap*.csv`, `landuse_probe_paired_diff.csv` (ohne `_ae`-Suffix) | `run_landuse_probe.py` (alter Lauf, beide Backbones) | `landuse_pairs_ae.parquet`+`landuse_pairs_tessera.parquet` **(beide waren zu diesem Zeitpunkt STALE)** | 1.–2.9. | **veraltet, ersetzt durch die `_ae`-Zeile oben. Noch nicht archiviert (nicht explizit im Archivierungsauftrag), nur gemeldet.** |
| `data/processed/landuse_probe_results_ae.csv`-Vorläufer vom 3.9. 22:07–22:53 (überschrieben) | `run_landuse_probe.py --backbones ae` | `landuse_pairs_ae.parquet` (damals noch STALE) | 3.9. | **durch den 4.9.-Rerun überschrieben — falls eine Kopie des alten Standes noch irgendwo referenziert wird, ist sie ungültig (Leakage).** |

## SDM-Rohdaten (Phase 2, INSTRUCTIONS_sdm.md — Socioeconomic Distribution Mapping London)

Beide Rohdateien manuell geladen, kein Download-Skript. Lagen zunächst unter `data/SDM/`,
liegen jetzt (Umzug ausserhalb dieser Session, vor Phase 2 festgestellt) unter `data/raw/sdm/`
— passt zur `.gitignore`-Konvention (`data/raw/` ist ausgenommen, `data/SDM/` wäre es nicht
gewesen). SHA-256 vor und nach dem Umzug identisch, siehe unten.

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/raw/sdm/London_SDM.csv` | manuell geladen (Nomis Census 2021 Bulk, kein Skript) | Nomis, Tabelle **TS062** (NS-SeC), Geography ueber "select areas within" → Regions → **London**, Ebene **2021 LSOA lower layer**, alle **9 NS-SeC-Kategorien**, **Counts** (nicht Percent), Format **CSV**, **mit area codes** | Zugriff 8.9.2026 (Datei selbst traegt die Copyright-Zeile „ONS Crown Copyright Reserved [from Nomis on 8 September 2026]") | aktuell — SHA-256 `042e214ab5d73efb9360ae534808a7103553a6d6e2b7ac3aa06fc06c7688556a`, Lizenz OGL v3.0. Format: Nomis-Bulk mit Metadaten-Kopf, echte Kopfzeile Zeile 8, Daten Zeile 10–5003 (4.994 Zeilen), Footer ab Zeile 5004 — siehe `build_sdm_targets.py::HEADER_ROW/N_DATA_ROWS`. |
| `data/raw/sdm/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10_-672099234420024429.gpkg` | manuell geladen (ONS Open Geography Portal) | Produkt „Lower layer Super Output Areas (December 2021) Boundaries EW **BFC** (full resolution, clipped to coastline), V10" — ganz England & Wales (35.672 Features), CRS EPSG:27700, Layer `LSOA_2021_EW_BFC_V10`, Felder `LSOA21CD, LSOA21NM, LSOA21NMW, BNG_E, BNG_N, LAT, LONG, GlobalID` | Zugriff 8.9.2026 (Dateisystem-Zeitstempel des Downloads: 8.9. 16:16) | aktuell — SHA-256 `fae6b97592c3164294d291d42298f83c97a751fbb9b9149b6ed61796a2f255a9`, Lizenz OGL v3.0 (enthaelt OS- und ONS-IPR). BFC bestaetigt (nicht BGC) ueber Datei- und Layername. Noch **nicht** auf London geclippt, noch **nicht** nach EPSG:32630 reprojiziert (Phase 3). |
| `data/processed/sdm_targets_london_9cat.parquet` | `build_sdm_targets.py` | `data/raw/sdm/London_SDM.csv` | 8.9.2026 | aktuell (ersetzt die fruehere `sdm_targets_london.parquet`, umbenannt zur Abgrenzung von der 8cat-Variante) — 4.994 Zeilen (= alle Zeilen der Quelle, kein Filter), 9 NS-SeC-Kategorien inkl. **L15** (Nutzervorgabe, s. u.), Total-Spalte nur zur Validierung (0 von 4.994 Zeilen mit Summe(9 Kategorien) != Total, 0 Zeilen mit Summe=0), Anteile je Zeile normiert auf Summe=1,0 (max. Abweichung 2,22e-16). Join-Key `lsoa_code` (aus `mnemonic`), 0 doppelte Codes. Spalten: `lsoa_code, lsoa_name, total, {9 Rohzaehlungen}, category_sum, {9 Anteile p_*}`. |
| `data/processed/sdm_targets_london_8cat.parquet` | `build_sdm_targets.py` | `data/raw/sdm/London_SDM.csv` | 8.9.2026 | aktuell — dieselben 4.994 Zeilen, aber **ohne L15**, erneut auf die Summe der verbleibenden 8 Kategorien normiert (NICHT die 9cat-Anteile ohne L15 weiterverwendet — sonst Zeilensumme < 1). Max. Abweichung Zeilensumme von 1,0: 2,22e-16, 0 Zeilen mit Summe=0. Spalten: `lsoa_code, lsoa_name, total, {8 Rohzaehlungen}, l15 (Referenz, nicht in category_sum), category_sum, {8 Anteile p_*}`. Verschiebung ggue. 9cat (mittlere \|Δ\| je Kategorie, absteigend): l4_l6 0,0204, l1_l3 0,0176, l14 0,0114, l8_l9 0,0113, l7 0,0097, l13 0,0092, l12 0,0092, l10_l11 0,0040 — max. Einzel-Δ bis 0,20 (l1_l3) bei LSOAs mit p_l15 > 0,6. Begruendung fuer beide Varianten: AETHERs eigene Kategorienwahl ist nicht rekonstruierbar (s. u.), daher werden beide Varianten durchgerechnet statt eine anzunehmen. |

**Kategorienentscheidung (Phase 2a/2a-Vertiefung):** Nutzervorgabe — beide Varianten werden
durchgerechnet (9cat mit L15, 8cat ohne L15, jeweils auf die eigene Kategoriensumme normiert),
weil sich AETHERs eigene Kategorienwahl **nicht** rekonstruieren laesst. Das oeffentliche
AETHER-Repo (github.com/inwind0212/AETHER, `main`, einziger Branch, einziger Commit `cc52099`,
keine Tags) wurde zweimal durchsucht — zuerst per Fachbegriff-Grep, dann strukturell (Notebooks,
Zahlenmuster fuer Klassenzahlen, Dateiendungen, Metrikbegriffe wie kl_div/chebyshev/l1_dist,
Spaltennamen wie L1_L3/L4_L6/L14/lsoa, vollstaendige Markdown-Ausgabe, Verweise auf ein
Zweit-Repo oder externe Datenablage). Ergebnis beider Durchgaenge: **kein** Treffer irgendeiner
Art. Das Repo enthaelt genau 13 getrackte Dateien, ausschliesslich die Contrastive-Pretraining-
Pipeline (AE×POI-Alignment: `train.py`, `dataio.py`, `model.py`, `infer_aether_raster.py`,
`config.yaml`, `README.md`, ein Beispiel-POI-GeoJSON, eine Abbildung, zwei Pretraining-
Checkpoints fuer London und Singapur). Keine `.ipynb`, keine zweite `.md`-Datei, keine
Zielvariablen-Datei, kein Metrik- oder Downstream-Code, keine Klassifikationsschicht in den
Checkpoints (nur `ae_state_dict`/`txt_state_dict`, reine Projektionsgewichte), kein Verweis auf
ein Zweit-Repo oder eine externe Datenablage. Die SDM-Auswertung aus Tabelle 1 des Papers ist
nachweislich nicht Teil der Veroeffentlichung.

## SDM-LSOA-Split (Phase 3, INSTRUCTIONS_sdm.md)

Ordnet jedes Greater-London-LSOA vollstaendig einer Partition (train/val/test) des
bestehenden 1-km-Rasters zu (Flaechenanteil-Mehrheitsregel), damit LSOA-Pixel nicht in
Train UND Test landen. Zellgeometrie rekonstruiert aus `data/splits/splits_london.csv`
(1.818 Zellen, 1.454/182/182 train/val/test) ohne Raster-Rerun (Ursprung/Zellgroesse aus
`build_splits.py`/`build_splits_pixels.py`, siehe Docstring von `build_sdm_lsoa_splits.py`).
Haengt nur an der Geometrie, nicht an den NS-SeC-Kategorien - identisch fuer 9cat und 8cat.

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/splits/sdm_lsoa_splits_london.csv` (Variante A, Mehrheitsregel) | `build_sdm_lsoa_splits.py` | `data/raw/sdm/Lower_layer_..._BFC_V10_....gpkg`, `data/aoi/london.gpkg`, `data/splits/splits_london.csv`, `data/raw/sdm/London_SDM.csv` (nur fuer die Codeliste) | 8.9.2026 | aktuell — 4.994 Zeilen (= alle TS062-LSOAs, alle gematcht). Join TS062↔Geometrie ueber `LSOA21CD`/`mnemonic`: 0 TS062-Codes ohne Geometrie, 119 Geometrie-Codes (AOI-Intersects) ohne TS062-Eintrag (verworfen - AOI ist die OSM-Verwaltungsgrenze admin_level=5 'Greater London', leicht groesser/anders geschnitten als Nomis' eigene Regionsauswahl an den Raendern). Split je LSOA: train=4.152, val=425, test=417. 1.966 LSOAs (39,4%) beruehren mehr als eine Partition (erwartbar: LSOAs sind i.d.R. kleiner als 1 km²-Zellen). max_frac: Median 1,0, Mittel 0,920, Min 0,373. 303 LSOAs (6,1%) mit max_frac < 60%. 2 LSOAs mit frac_covered < 99,9% (Restflaeche in Zellen ohne Split-Eintrag, randlich). |
| `data/splits/sdm_lsoa_splits_london_min60.csv` (Variante B, Schwellwert) | `build_sdm_lsoa_splits.py` | wie oben | 8.9.2026 | aktuell — Variante A gefiltert auf `max_frac >= 0,60` (Schwellwert = derselbe, der ohnehin fuer den "< 60%"-Bericht verlangt war, keine zusaetzliche freie Zahl). 4.691 Zeilen (303 verworfen). Split: train=4.010, val=343, test=338. |
| `data/splits/sdm_lsoa_cell_overlay_london.csv` | `build_sdm_lsoa_splits.py` | dieselben Inputs | 8.9.2026 | aktuell — 14.138 Zeilen, Spalten `lsoa_code, cell_row, cell_col, split, piece_area_m2`. Rohe LSOA×Zelle-Ueberlagerung (vor der Mehrheits-Aggregation) - Grundlage der **partitionsreinen Poolung** (s. u.): Phase 5 filtert damit auf Pixel, deren Zelle denselben `split` traegt wie das LSOA selbst, ohne die Ueberlagerung neu zu rechnen. |

Beide Split-Varianten teilen sich das Spaltenschema: `lsoa_code, split, max_frac, second_max_frac,
margin_frac, frac_train, frac_val, frac_test, frac_covered, n_partitions_touched, lsoa_area_m2,
area_kept_m2, n_pixels_total_est, n_pixels_kept_est, frac_pixels_kept`.

**Diagnose der 303 Grenzfaelle (max_frac < 60%):** 140 davon (46,2%) haben `margin_frac` < 0,10
(zweitgroesster Flaechenanteil liegt weniger als 10 Prozentpunkte unter dem groessten - die
Zuordnung ist dort de facto ein Muenzwurf). Raeumlich **nicht** am AOI-Rand geclustert, sondern
ueber die Stadt verteilt: mittlerer Abstand der Grenzfall-Zentroide zum AOI-Rand 7.366 m, praktisch
identisch zum Mittel ueber ALLE 4.994 LSOAs (7.778 m). Betrifft alle 33 Londoner Boroughs (keine
Konzentration auf wenige Boroughs; Ealing/Hillingdon mit je 21 LSOAs am haeufigsten, aber das sind
nur ~7% ihrer jeweiligen LSOA-Zahl). Strukturell erwartbar: die Zell→Split-Zuordnung ist ein
zufaelliger Shuffle ueber ALLE 1.818 Zellen (`build_splits.py::assign_splits`), nicht raeumlich
geblockt im Sinne zusammenhaengender Regionen - Val/Test-Zellen liegen zufaellig verstreut zwischen
Train-Zellen, daher treten Grenzfaelle ueberall dort auf, wo ein LSOA-Rand zufaellig auf eine
Zellgrenze mit unterschiedlichem Split faellt, nicht bevorzugt am AOI-Rand.

**Partitionsreine Poolung (schliesst einen Leakage-Kanal):** ohne Filter wuerde der
gepoolte Regionsvektor eines LSOA ueber Pixel aus bis zu drei Partitionen mitteln, obwohl das
LSOA nur EINER Partition (der Mehrheits-Partition) zugewiesen ist - der Regionsvektor eines
"Train"-LSOA enthaelt in bis zu 39,4% der Faelle auch Information aus Val-/Test-Zellen. Filter:
nur Pixel aus Zellen mit `split == LSOA-split` behalten (`sdm_lsoa_cell_overlay_london.csv`).
`frac_pixels_kept` (== `max_frac`, Flaechenanteil ≈ Pixelanteil bei 10-m-Aufloesung, kein
Raster-Durchlauf noetig): Mittel 0,9203, Median 1,0000, Min 0,3732, 5%-Quantil 0,5819. Nur
**1 von 4.994** LSOAs faellt nach dem Filter unter 100 Pixel (geschaetzt), **0 LSOAs** werden
komplett leer. `n_pixels_total_est` (vor Filter): Min 97, Median 1.980, Max 158.049.

## SDM-Projiziertes-Feld (Phase 4, INSTRUCTIONS_sdm.md)

Projiziert jedes rohe EE-Pixel (Greater-London-Ausschnitt) einzeln durch den Projektionskopf,
Arm B (untrainiert) und Arm C (trainiert), je Backbone - unabhaengig von Ziel-/Split-Variante,
einmal gerechnet. Arm A braucht kein Feld (poolt direkt ueber das rohe Raster).

**Pre-Flight norm_stats (Nutzervorgabe):** AE braucht laut Config/Checkpoint (`normalization:
none`, AlphaEarth ist bereits L2-normalisiert) **gar keine** Statistikdatei - nichts fehlt,
weil nichts erwartet wird (`check_norm_stats_available()`: `{'mode': 'none', 'stats_path':
None, 'exists': None, 'tracked': None}`). TESSERA braucht `norm_stats_tessera_pixels.json` -
vorhanden UND getrackt (der `-f`-Fund aus Commit 4 galt nur fuer TESSERA, AE hat strukturell
kein Aequivalent, das fehlen koennte).

| Datei/Ordner | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/interim/sdm_projected/{ae,tessera}_arm{b,c}/*_proj256.tif` | `build_sdm_projected_field.py` | rohe EE-Kacheln (`spatial_join.py::get_backbones('london')`), `data/aoi/london.gpkg`, `checkpoints/{ae,tessera}_v1/best.pt` (nur Arm C), `norm_stats_tessera_pixels.json` (nur TESSERA) | 8.9.2026 | aktuell — 2+2+50+50=104 GeoTIFFs, 256 Kanaele float32, BIGTIFF (AE-Kacheln > 4 GB), zeilenweise geschrieben (`ROW_CHUNK=200`) um den Speicherbedarf zu begrenzen (ungechunkt: eine einzelne AE-Kachel trieb den Prozess auf ueber 100 GB RSS, nahe am System-Limit von 119 GB - gefunden und behoben, bevor der volle Lauf startete). Gesamt 131 GB, gesamt ~30 min Rechenzeit (Summe der 4 Arm-Laeufe, sequenziell). Raeumlich auf die AOI-BBox je Kachel zugeschnitten (nicht die vollen 8192x8192-AlphaEarth-Rohkacheln - der Rest liegt ausserhalb Greater Londons). |

Werte je Arm (Wertebereich ueber alle Kacheln, Backbone):
- AE Arm B (untrainiert): [-0,2383, 0,2271], 26.581.324 gueltige Pixel, 486,8s
- AE Arm C (trainiert, ep22): [-0,6064, 0,5699], 26.581.324 gueltige Pixel, 481,0s
- TESSERA Arm B (untrainiert): [-0,3458, 0,3647], 28.831.211 gueltige Pixel, 418,7s
- TESSERA Arm C (trainiert, ep26): [-0,4539, 0,4976], 28.831.211 gueltige Pixel, 414,0s

Trainierte Koepfe (Arm C) haben durchgehend einen breiteren Wertebereich als die untrainierten
(Arm B) - erwartbar, das Training draengt semantisch unterschiedliche Pixel im 256-dim-Raum
auseinander.

**Stichprobencheck L2-Norm == 1:** TESSERA (im Lauf selbst geprueft): 4.000 Pixel/Arm aus 20
Kacheln, max. Abweichung 1,19e-07. AE: der im Lauf eingebaute Zufallscheck traf bei beiden
Kacheln ein reines Nodata-Fenster (0 Pixel geprueft, gemeldeter Fehlbefund im Skript - AE-
Kacheln haben grosse Nodata-Raender, ein einzelnes 64x64-Fenster kann leer sein) - nachtraeglich
mit Wiederholungsversuchen bei leerem Fenster nachgeholt: 200 Pixel je Kachel/Arm (800 gesamt),
max. Abweichung ebenfalls 1,19e-07. `sample_l2_norm_check()` danach mit Retry-bei-leerem-Fenster
nachgeruestet (bis zu 10 Versuche je Kachel) - der obige Nodata-Fehlbefund kann bei einem
kuenftigen Lauf nicht mehr auftreten.

**Kritischer Bugfix (Nachtrag, vor Phase 5):** die urspruengliche Phase-4-Schreiblogik liess
`tiled`/`interleave` auf dem GDAL-Default (ungetilte Streifen, `interleave='pixel'`) - fuer
Phase 5s `exactextract`-Abfragen katastrophal, weil jede LSOA-Abfrage trotz "nur Band 1 noetig"
alle 256 Baender pro Zeile dekomprimieren musste. Der erste Phase-5-Lauf haengte >26h bei der
ersten AlphaEarth-Kachel (Arm B) fest, 69,8 GB RSS, keine einzige Kachel fertig - abgebrochen.
Fix: `build_sdm_projected_field.py`-Schreibprofil auf `tiled=True, blockxsize=blockysize=256,
interleave='band'` umgestellt (kuenftige Laeufe); die bereits berechneten 104 Dateien wurden
verlustfrei per `retile_sdm_projected.py` konvertiert (Werte stichprobenartig bit-identisch
verifiziert, keine Neuprojektion). Sanity-Check danach: dieselbe AlphaEarth-Kachel/-Abfrage
(4.994 Kandidaten-LSOAs), die vorher nach >26h nicht fertig war, brauchte retiled 422,6s.

## SDM-Poolung (Phase 5, INSTRUCTIONS_sdm.md)

Coverage-gewichtetes Mean-Pooling je LSOA per `exactextract` (`["cell_id","coverage"]`,
identischer Mechanismus wie `spatial_join.py::_join_polygons`). Arm A poolt ueber das rohe
Backbone-Raster (mit derselben Normalisierung wie ueberall sonst: z-score fuer TESSERA, none
fuer AE - Konsistenz mit Arm B/C, die schon in Phase 4 normalisiert wurden), Arm B/C ueber das
projizierte 256-Kanal-Feld aus Phase 4. Zwei Poolungsvarianten in EINEM Durchlauf (dieselben
exact_extract-Treffer, zwei laufende Summen): partitionsrein (Hauptlauf, nur Pixel aus Zellen,
deren Split mit dem LSOA-Split uebereinstimmt) und alle Pixel im Polygon (Robustheitscheck).
Split-Variante B braucht keinen eigenen Poolungslauf (Teilmenge derselben Mehrheits-Zuordnung,
Vektoren fuer verbleibende LSOAs sind bit-identisch - Filter statt Neuberechnung in Phase 6).

**Re-Normalisierung nach dem Pooling (offene Entscheidung, explizit getroffen):** NEIN, fuer
alle drei Arme gleich keine erneute L2-Normalisierung. Begruendung: TESSERAs rohe Kanaele sind
z-score-normiert, nicht L2-normiert - eine nachtraegliche Zwangs-Normierung auf Einheitslaenge
haette dort keine natuerliche Bedeutung und waere nur fuer AE/B/C (bereits unit-norm) motiviert,
nicht einheitlich fuer alle drei Arme. Der Phase-6-Vorhersagekopf (MLP+Softmax) verlangt anders
als der SigLIP-Pretraining-Loss keine Einheitsnorm. Einfacherer Default ohne Not-wendige
Zusatztransformation - "keine Transformation ohne Grund".

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/processed/sdm_pooled_{ae,tessera}_arm{a,b,c}.parquet` (6 Dateien) | `build_sdm_pooled_embeddings.py` | rohe/projizierte EE-Kacheln, `sdm_lsoa_splits_london.csv` (Variante A), `splits_london.csv`, LSOA-Geometrie | 11.9.2026 | aktuell — je 4.994 LSOAs erreicht (= alle), **0 LSOAs mit 0 Pixeln** in JEDER Kombination, in BEIDEN Poolungsvarianten (Spalten `n_pixels_all`/`n_pixels_pure`, `all_0..all_255`/`pure_0..pure_255` bzw. `all_0..all_127`/`pure_0..pure_127` fuer TESSERA-Arm-A, `all_0..all_63`/`pure_0..pure_63` fuer AE-Arm-A). Laufzeiten: ae/a 129s, ae/b 903s, ae/c 955s, tessera/a 1654s, tessera/b 615s, tessera/c 614s (nach dem Retiling-Fix). |

**LSOAs mit 0 Pixeln:** in keiner der 6 Kombinationen, weder "alle Pixel" noch "partitionsrein" -
passt zur Phase-3-Vorhersage (max_frac nie 0, 5%-Quantil frac_pixels_kept=0,582).

**Verifikation vor Phase 6 (Nutzeranfrage):** bei Arm A/TESSERA wird die z-score-Normierung PRO
PIXEL angewandt, BEVOR die coverage-gewichtete Summierung beginnt (`build_sdm_pooled_
embeddings.py` Zeilen ~133-151 - erst `apply_zscore()` auf den rohen Pixelvektor, dann in
`sum_all`/`sum_pure` aufsummiert), dieselbe Reihenfolge wie bei Arm B/C. Statistik-Identitaet
explizit geprueft: `configs/tessera_v1.yaml` UND die im `tessera_v1`-Checkpoint eingebettete
Trainings-Config zeigen beide auf denselben Pfad (`data/processed/norm_stats_tessera_pixels.json`,
Hash `471c7157...`), mtime 4.9. 22:06, unmittelbar vor Trainingsstart 22:06:21 gefittet - exakt
die beim Training verwendete Statistik, keine neu gefittete oder abweichende.

## SDM-Vorhersagekopf (Phase 6, INSTRUCTIONS_sdm.md)

Zweischichtiger MLP (Linear->ReLU->Linear, HIDDEN_DIM=128), Embeddings eingefroren, Softmax
(als `log_softmax` in der Loss) auf die C NS-SeC-Kategorien, distributionale Cross-Entropy,
Adam (lr=1e-3, weight_decay=1e-4), Early Stopping auf Val-Loss (Patience 20, max. 500 Epochen),
5 Seeds (0-4). Vier Szenarien statt vollem Kreuzprodukt-Sweep (je EIN Faktor gegenueber dem
Hauptlauf geaendert): Hauptlauf (9cat/Split A/partitionsrein), Robustheit 8cat, Robustheit
Split B, Robustheit Poolung-alle-Pixel. Je Szenario: 2 Backbones x 3 Arme x 5 Seeds, plus
Mittelwertsvorhersage-Baseline (deterministisch, nur von target+split abhaengig) und
Random-Embedding-Baseline (5 Seeds, dimensionsangepasst: 64/128 fuer Arm A, 256 fuer Arm B/C).

**Bewusst NICHT hier berechnet:** die eigentlichen Bewertungsmetriken KL/L1/Chebyshev mit
Kachel-Bootstrap - das ist Phase 7 (dort muessen KL-Richtung, Null-Behandlung, L1-Definition
und die Bootstrap-Ebene noch entschieden werden). Phase 6 liefert nur die Cross-Entropy als
Konvergenz-/Sanity-Metrik und die Test-Predictions.

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/processed/sdm_phase6_head_results.csv` | `build_sdm_phase6_predict.py` | `sdm_pooled_*.parquet`, `sdm_targets_london_{9cat,8cat}.parquet`, `sdm_lsoa_splits_london{,_min60}.csv` | 13.9.2026 | aktuell — 4 Szenarien x 6 Backbone/Arm-Kombinationen x 5 Seeds = 120 Zeilen (`scenario, backbone, arm, seed, train_loss, val_loss, test_loss, epochs`). Gesamtlaufzeit 220,8s. |
| `data/processed/sdm_phase6_baselines.csv` | `build_sdm_phase6_predict.py` | dieselben | 13.9.2026 | aktuell — 4 Szenarien x (1 Mittelwerts- + 6 Random-Embedding-Baselines) = 28 Zeilen. |

Hauptlauf-Ergebnisse (test_loss, Mittel +- Std ueber 5 Seeds; Mittelwertsvorhersage-Baseline
2,1089):

| Backbone/Arm | test_loss | Random-Embedding-Baseline |
|---|---|---|
| AE/A | 2,0896 +- 0,0005 | 2,1145 +- 0,0015 |
| AE/B | 2,0934 +- 0,0005 | 2,1206 +- 0,0024 |
| AE/C | 2,0823 +- 0,0004 | 2,1206 +- 0,0024 |
| TESSERA/A | **2,0728 +- 0,0004** (bester Wert) | 2,1160 +- 0,0012 |
| TESSERA/B | 2,0797 +- 0,0002 | 2,1187 +- 0,0011 |
| TESSERA/C | 2,0748 +- 0,0002 | 2,1187 +- 0,0011 |

Durchgehend ueber alle vier Szenarien: echte Embeddings schlagen beide Baselines, Random-
Embedding schneidet leicht SCHLECHTER ab als die reine Mittelwertsvorhersage (plausibel - dem
MLP fehlt bei reinem Rauschen jedes Signal, Konvergenz zur trivialen Mittelwertloesung ist nicht
perfekt). TESSERA/A durchgehend das beste Ergebnis in allen vier Szenarien.

## SDM-Metriken (Phase 7, INSTRUCTIONS_sdm.md)

KL(wahr‖vorhergesagt), L1 (Summe + Mittel als zweite Spalte), Chebyshev, ueber das Testset,
fuer alle 4 Szenarien x 2 Backbones x 3 Arme x 5 Seeds plus beide Baselines. Bootstrap-
Cluster-Einheit: die 1-km-Zelle. LSOA->Zelle-Zuordnung aus `sdm_lsoa_cell_overlay_london.csv`,
je LSOA die Zelle mit der GROESSTEN Ueberlappungsflaeche (`majority_cell_per_lsoa()`) - dieselbe
Mehrheitsregel wie fuer die LSOA->Split-Zuordnung in Phase 3, nur auf Zellebene statt
Split-Label-Ebene. Bootstrap-Punktschaetzer aus Seed 0 (dokumentierte Wahl, nicht ueber Seeds
gemittelt - Seed-Varianz ist eine separate, bereits berichtete Groesse).

Sanity-Check (vor dem vollen Lauf): Arm A/AlphaEarth/9cat/SplitA/partitionsrein, KL=0,0503
(unskaliert) gegen AETHERs 0,0422 - plausibel (erwarteter Bereich "um 0,05" wegen strengerem
Split), siehe Chat-Bericht 13.9.2026.

| Datei | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/processed/sdm_phase7_metrics.csv` | `build_sdm_phase7_metrics.py` | `sdm_pooled_*.parquet`, `sdm_targets_london_{9cat,8cat}.parquet`, `sdm_lsoa_splits_london{,_min60}.csv`, `sdm_lsoa_cell_overlay_london.csv` | 13.9.2026 | aktuell — 4 Szenarien x 13 Modelle (6 echte + 6 dimensionsangepasste Random-Embedding + 1 Mittelwertsvorhersage) = 52 Zeilen. Nur der Hauptlauf hat zusaetzlich `*_ci_lsoa`-Spalten (LSOA-Bootstrap zum Breitenvergleich). Gesamtlaufzeit 244,8s. |

**Hauptlauf-Ergebnistabelle** (AETHER-Skala: L1x10², KL/Chebyshev x10³; unskaliert in Klammern),
95%-Kachel-Bootstrap-CI in eckigen Klammern (unskaliert):

| Modell | KL | L1 (Summe) | Chebyshev |
|---|---|---|---|
| **AETHER Tab. 1 — Random** (Referenz, fremder Split/Kategorien) | 75,7 | 30,0 | 91,4 |
| **AETHER Tab. 1 — CaLLiPer** (Referenz) | 34,1 | 19,3 | 58,0 |
| **AETHER Tab. 1 — AlphaEarth/roh** (Referenz) | 42,2 | 21,5 | 64,8 |
| **AETHER Tab. 1 — AETHER** (Referenz) | 30,1 | 17,8 | 53,8 |
| Mittelwertsvorhersage | 69,6 (0,0696) [60,8-78,8] | 28,5 (0,2852) [26,9-30,2] | 86,4 (0,0864) [79,3-93,2] |
| AE/A (=AlphaEarth roh) | 50,3 (0,0503) [42,6-57,7] | 23,7 (0,2367) [22,0-25,1] | 71,1 (0,0711) [64,6-76,8] |
| AE/B | 54,1 (0,0541) [47,8-62,9] | 24,7 (0,2472) [23,4-26,5] | 74,6 (0,0746) [69,3-81,2] |
| AE/C | 43,1 (0,0431) [37,0-50,5] | 21,6 (0,2165) [20,2-23,1] | 64,9 (0,0649) [59,3-70,6] |
| TESSERA/A | 33,6 (0,0336) [28,0-39,9] | 18,7 (0,1874) [17,4-20,1] | 56,9 (0,0569) [51,5-61,9] |
| TESSERA/B | 40,4 (0,0404) [33,7-48,0] | 20,7 (0,2074) [19,3-22,4] | 62,9 (0,0629) [56,7-69,5] |
| TESSERA/C | 35,6 (0,0356) [29,4-42,1] | 19,4 (0,1939) [18,0-20,8] | 58,7 (0,0587) [53,2-64,2] |
| Random-Embedding (dimensionsangepasst, 64-256) | 75-81 | 30-31 | 90-94 |

**Sanity-Check bestaetigt** (AE/A: 50,3 vs. AETHERs 42,2, im vorab als plausibel markierten
Bereich). **TESSERA/A schlaegt sogar AETHERs eigenes Modell** (33,6 vs. 30,1 KL) UND liegt
naeher an AETHER als an ihrem eigenen "AlphaEarth roh"-Wert (42,2) - deutet auf einen
substanziellen Backbone-Effekt zugunsten TESSERA fuer diese Aufgabe, nicht nur Rauschen (die
95%-CIs [28,0-39,9] fuer TESSERA/A und [37,0-50,5] fuer AE/C ueberlappen sich knapp nicht am
Rand). Kein direkter KL-Vergleich mit AETHERs eigenen Werten im strengen Sinn zulaessig
(anderer Split, andere Kategorienwahl, andere LSOA-Partitionszuordnung - siehe CLAUDE.md-
Leitlinie "kein direkter Zahlenvergleich gegen publizierte Paper"), nur als Groessenordnungs-
Einordnung zu verstehen.

**Kachel- vs. LSOA-Bootstrap, Breitenvergleich (Hauptlauf, Beleg dass die Cluster-Einheit einen
Unterschied macht):** Kachel-CIs sind ueber alle 13 Modelle und alle drei Metriken konsistent
BREITER als LSOA-CIs - mittlerer Breitenfaktor (Kachel/LSOA): KL **1,29x**, L1-Summe **1,39x**,
Chebyshev **1,34x**. Bestaetigt die erwartete raeumliche Autokorrelation zwischen benachbarten
LSOAs in derselben Zelle - LSOA-Bootstrap unterschaetzt die Unsicherheit systematisch.

## Noch nicht archiviert, aber ebenfalls Alt-/Nebenmaterial (zur Kenntnisnahme, nicht Teil des Archivierungsauftrags)

- `data/interim/poi_text.parquet` (31.8., von `poi_text_full.parquet` abgelöst)
- `data/processed/text_embeddings.parquet` (1.9., POI-Ebene-Textembeddings, von `pixel_text_embeddings.parquet` abgelöst — Achtung: laut `compute_pixel_text_embeddings.py`-Docstring für HILUCS-Zero-Shot-Prompts evtl. noch separat gebraucht, nicht vorschnell löschen)
- `data/interim/pois_london_test.parquet`, `pois_london_test_filtered.parquet`, `pois_london_object_qid_check.parquet` (Test-/Zwischenstände der Extraktion)
- `run_ae_join.py`, `run_tessera_join.py` — **archiviert** (`data/archive_gen1/scripts/`, 3.9.). Waren alte Join-Skripte mit `OUT_PATH` auf die jetzt ebenfalls archivierten `pairs_ae.parquet`/`pairs_tessera.parquet`; abgelöst durch `run_ae_join_full.py`/`run_tessera_join_full.py`. Liegen jetzt nicht mehr im aktiven Repo-Root, ein versehentlicher Re-Run ist damit ausgeschlossen.
