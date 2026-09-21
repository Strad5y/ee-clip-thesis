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

## Text-Query-Evaluation, Phase 1 (INSTRUCTIONS_text_query.md)

**Entscheidung (Jakob, 2026-09-17):** Phase 1 rechnet KEINE neuen projizierten 256-d-Felder.
Wiederverwendet werden unveraendert die bereits bestehenden Arm-C-Felder aus der SDM-Phase 4
(s.o., `build_sdm_projected_field.py`) - selber Checkpoint, selbe Normalisierung, bereits AOI-
zugeschnitten. Neu fuer diese Aufgabe: `src/eval/text_query/tiles.py` (Loader, ordnet jede
Rohkachel ihrem Arm-C-Feld zu, nativ pro Kachel-CRS) und
`src/eval/text_query/verify_projected_fields.py` (die drei geforderten Pruefungen).

| Datei/Ordner | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/interim/sdm_projected/{ae,tessera}_armc/*_proj256.tif` | `build_sdm_projected_field.py` (SDM Phase 4, s.o., **unveraendert wiederverwendet**) | — | 8.9.2026 | aktuell fuer Text-Query Phase 1 — `checkpoints/ae_v1/best.pt` (Epoche 22), `checkpoints/tessera_v1/best.pt` (Epoche 26) |

**Pruefung 1 (L2-Norm, Stichprobe):** TESSERA 8 Kacheln/4.000 Pixel, AE 2 Kacheln/1.000 Pixel,
je max. Abweichung von 1,0: **1,19e-07** (Float32-Maschinenepsilon) — bestaetigt, keine neue
Pruefung noetig (deckt sich mit dem bereits in Phase 4/SDM dokumentierten Wert).

**Pruefung 2 (Reprojektions-Kosinus, 1.000 Pixel/Backbone):** min_cosine TESSERA/AE je
**0,999999880...**, mean 1,0, 0 von 1.000 unter der Schwelle 0,9999 — **bestanden**, beide
Backbones. Erster Lauf schlug fehl (min_cosine bis -0,29, 260/1000 unter Schwelle bzw. Abbruch
bei AE) — Ursache war ein Bug im eigenen Pruefskript (Roh-Pixel wurden am selben (row,col) wie
im AOI-zugeschnittenen Arm-C-Feld gelesen, ohne den Fenster-Offset aus `project_tile()` zu
beruecksichtigen), kein Befund ueber die Arm-C-Daten selbst. Fix: Positionen laufen jetzt ueber
Weltkoordinaten (Arm-C-Transform -> x,y -> Roh-Tile-Transform).

**Pruefung 3 (raeumliche Kachel-Ueberlappung):** empirisch per Punktstichprobe verifiziert, ob
sich geometrisch ueberlappende Kachel-Footprints auch in ECHTER (nicht-NoData) Datendopplung
niederschlagen:

- **TESSERA:** 133 von 1.225 moeglichen Kachelpaaren ueberlappen geometrisch, 283,43 km² Gesamt-
  flaeche. Davon 13 Paare (26,22 km²) **zonenuebergreifend** (32630/32631 — die bereits in
  `spatial_join.py` dokumentierte, bisher ungeloeste Leakage-Quelle) und **120 Paare (257,21 km²,
  bisher nirgends dokumentiert) innerhalb derselben Zone** (Nachbarkacheln mit eingebautem
  Rand-Overlap). Stichprobencheck (1.001 bzw. 500 Punkte) in je einem Beispiel-Paar: **100%**
  der Punkte haben in BEIDEN Kacheln echte, gueltige Werte — die Ueberlappung ist real, nicht nur
  ein Bounding-Box-Artefakt.
  `spatial_join.py::assign_utm_zone()`/`join_pois_to_tiles()` verhindert Dopplung **nur**
  zonenuebergreifend, und **nur** fuer den POI-Pixel-Join (`pairs_*.parquet` -> `pixels_*.parquet`)
  — nicht fuer Nachbarkacheln derselben Zone (dafuer existiert ueberhaupt kein Dedup-Mechanismus)
  und nicht fuer die rohen Arm-C-Felder dieser Aufgabe (`build_sdm_projected_field.py` verarbeitet
  jede Kachel unabhaengig, ohne Zonenfilter). `build_splits_pixels.py` weist Splits ueber reale
  Weltkoordinaten (`cell_row`/`cell_col`) zu — zwei Dopplungs-Pixel derselben Position landen daher
  im selben Split (kein Train/Test-Leck ueber den Split selbst), aber beide Kopien bleiben als
  separate `(tile,row,col)`-Zeilen im Datensatz bzw. als zwei potenziell abweichende Werte in den
  Arm-C-Feldern bestehen.
- **AlphaEarth:** nur 1 Kachelpaar, aber geometrisch riesige Ueberlappung (5.940,41 km² von
  6.710,89 km² Kachelflaeche, ≈88,5%) — die beiden 8192×8192-Rohkacheln (30N/31N) decken in
  ihrer vollen Bounding Box fast dieselbe Flaeche ab. Stichprobencheck (2.823 Punkte im
  Ueberlappungspolygon): nur **0,1%** (2/2.823) haben in BEIDEN Kacheln gueltige (Nicht-NoData)
  Werte — jede Kachel deckt real nur ihre eigene UTM-Zonen-Haelfte Greater Londons ab, der Rest
  ist NoData (-128). Die grosse geometrische Bounding-Box-Ueberlappung ist damit fuer AlphaEarth
  **praktisch folgenlos**.

**Wichtig — betrifft auch die bestehenden Trainingspaare:** die oben beschriebene Ueberlappung
(283,43 km² TESSERA-Kachelflaeche, davon 257,21 km² innerhalb derselben UTM-Zone; AlphaEarths
riesige Bounding-Box-Ueberlappung ist dagegen praktisch folgenlos) steckt strukturell auch in
`pairs_tessera_full.parquet`/`pixels_tessera.parquet` und den daraus abgeleiteten
Trainingspaaren — `spatial_join.py::assign_utm_zone()` dedupliziert dort NUR den
zonenuebergreifenden Anteil (26,22 km²), die 257,21 km² innerhalb derselben Zone sind
ungefiltert enthalten. Diese Dopplung wird hier NUR dokumentiert, NICHT korrigiert (keine
Aenderung an `pairs_*`/`pixels_*`/Checkpoints) — Entscheidung, ob/wie das fuer das Training
relevant ist, liegt bei Jakob/Kai, nicht Teil dieser Aufgabe.

**Entscheidung (Jakob, 2026-09-17) fuer die Text-Query-Auswertung:** eigene Eindeutigkeits- und
AOI-Maske pro Kachel, siehe Abschnitt "Text-Query-Masken (Phase 2)" unten — die Arm-C-Embeddings
selbst bleiben unveraendert, nur eine zusaetzliche Maskenschicht wird ergaenzt.

## Text-Query-Masken, Phase 2 (INSTRUCTIONS_text_query.md)

**Eindeutigkeitsregel:** ein gueltiges Pixel gehoert der Kachel, deren Kachelmittelpunkt
(Bounding-Box-Zentrum der ROHEN Kachel, in EPSG:32630 verglichen) seinem Pixelzentrum am
naechsten liegt; bei exaktem Gleichstand gewinnt die Kachel mit dem lexikographisch kleineren
Dateinamen. **AOI-Maskierung:** zusaetzlich muss das Pixelzentrum innerhalb des Greater-London-
Polygons (`data/aoi/london.gpkg`) liegen — verifiziert, dass die Arm-C-Felder nur auf die
Bounding Box des AOI-Polygons zugeschnitten sind, nicht auf das Polygon selbst (Beispiel-Kachel
`grid_-0.05_51.25_2025.tiff`: 135.044 laut Arm-C "gueltige" Pixel, aber nur 1.931 davon liegen
tatsaechlich innerhalb des Polygons — 133.113 sind Bounding-Box-Eckflaechen ausserhalb Greater
Londons).

| Datei/Ordner | Erzeugendes Skript | Direkte Inputs | Erzeugungsdatum | Generation |
|---|---|---|---|---|
| `data/interim/text_query_masks/{tessera,ae}/*_mask.tif` | `src/eval/text_query/masks.py` | Arm-C-Felder (Phase 1), `data/aoi/london.gpkg`, Roh-Kachel-Bounds (fuer Kachelmittelpunkte) | 17.9.2026 | aktuell — 1 Byte-GeoTIFF (0/1) pro Kachel, identisches Raster wie das zugehoerige Arm-C-Feld |

**Ergebnis pro Backbone** (gueltige Pixel vor/nach Maskierung, Flaeche bei 100 m²/Pixel):

| Backbone | gueltig vorher | Fläche vorher | gueltig nachher | Fläche nachher |
|---|---|---|---|---|
| TESSERA | 28.831.211 | 2.883,12 km² | 15.954.158 | 1.595,42 km² |
| AlphaEarth | 26.581.324 | 2.658,13 km² | 15.954.033 | 1.595,40 km² |

Beide Backbones landen nach Maskierung auf praktisch identische Flaeche (Differenz 125 Pixel,
0,00078%) — plausibel, da beide auf dasselbe AOI-Polygon maskiert werden; die minimale
Restdifferenz kommt aus der unterschiedlichen nativen Rasterausrichtung/Kachelgrenzen der
beiden Backbones. Referenzwert Greater London (offizielle Flaeche): ≈1.572 km² — die 1.595 km²
hier sind der AOI-**Polygon**-Flaeche (OSM-Verwaltungsgrenze `admin_level=5`, s. Phase 3 SDM)
zuzurechnen, keine neue Abweichung.

**Pruefungen:**
- **Flaechenpruefung** (unabhaengige Berechnung: Vereinigung aller Kachel-Footprints ∩
  AOI-Polygon, in EPSG:32630): TESSERA 1.595,49 km² vs. maskierte Pixelflaeche 1.595,42 km²
  (Verhaeltnis 0,99996); AlphaEarth 1.595,49 km² vs. 1.595,40 km² (Verhaeltnis 0,99995) — beide
  bestanden.
- **Stichprobe gegen Doppelvergabe:** AlphaEarth 500 Punkte, 0 doppelt vergeben. TESSERA 370
  Punkte, **2 scheinbar doppelt vergeben** — untersucht: beide Faelle liegen exakt auf der
  mathematischen Grenzlinie zwischen zwei Kacheln, deren 10-m-Pixelgitter **nicht phasengleich**
  sind (unabhaengige Downloads, kein gemeinsamer Rasterursprung). Der Punkt aus Kachel A und der
  von der Pruefung in Kachel B gefundene "naechste" Pixel sind dadurch zwei **verschiedene**
  reale Orte, nur ca. 5 m auseinander (weniger als eine Pixelbreite) — beide sind, einzeln
  nachgerechnet, korrekt ihrer jeweils naeheren Kachel zugeordnet. Kein Fehler in der
  Zuordnungsregel, sondern ein inhaerenter Diskretisierungseffekt am Kachelrand im Bereich von
  weniger als einer Pixelbreite, quantitativ vernachlaessigbar (bestaetigt durch die
  Flaechenpruefung oben, die auf 0,004%–0,005% genau aufgeht).

## TESSERA-Vollständige-Dedup: Ursache, Regel, Artefakte (Schritt 1, Hauptaufgabe 2026-09-18)

**Ursache:** TESSERA-Rohkacheln überlappen sich flächenmäßig (Phase 2 der Text-Query-Aufgabe,
siehe Abschnitt "Text-Query-Masken, Phase 2" oben) — insgesamt ≈283 km², davon ≈257 km²
**innerhalb derselben UTM-Zone** (bis dahin unbekannt/undokumentiert, weil die Kachel-Footprint-
Prüfung dort nur zonenübergreifende Überlappung [30N/31N, ≈26 km²] betrachtet hatte) sowie die
bereits bekannten ≈26 km² zonenübergreifend. Grund: unabhängig heruntergeladene, nicht
phasengleiche 10-m-Rohkachel-Grids. Governing-Prinzip (Jakob): "Jeder Weltort existiert in
jedem TESSERA-Datensatz genau einmal" — betrifft alle drei zentralen TESSERA-Pixel-/Paar-
Tabellen (`pairs_tessera_full`, `pixels_tessera`, `landuse_pairs_tessera`), nicht nur die
Text-Query-Arm-C-Felder, für die die Maske ursprünglich gebaut wurde.

**Eindeutigkeitsregel** (identisch zu Phase 2 oben, hier auf Pixel-/Paar-Tabellenebene
angewendet statt auf Rasterfelder): ein Pixel gehört der Kachel, deren Rohkachel-Bounding-Box-
Mittelpunkt (in EPSG:32630) seinem Pixelzentrum am nächsten liegt; Tie-Break über den
lexikographisch kleineren Dateinamen. Zusätzlich muss das Pixelzentrum innerhalb des Greater-
London-AOI-Polygons liegen (nicht nur dessen Bounding Box).

**Restauflösung über Weltkoordinaten** (`tessera_dedup_utils.py::resolve_residual_duplicates()`):
die beiden Masken werden JE KACHEL unabhängig berechnet — an der exakten mathematischen
Kachelgrenze können deshalb zwei verschiedene reale Orte (nur wenige Meter auseinander, unter
einer Pixelbreite, wegen nicht-phasengleicher Grids) BEIDE als "gültig" markiert werden, weil
jede Maske nur ihre eigene Kachel sieht (identischer Diskretisierungseffekt wie in Phase 2,
Zeile 440–449 oben, dort mit 2 von 370 Stichprobenpunkten quantifiziert). Fix: iterativer
zweiter Schritt, der `find_duplicate_rows()` (siehe `check_tile_overlap_side_effects.py`) direkt
auf den bereits primär gemaskten Zeilen paarweise über Weltkoordinaten erneut anwendet und die
laut Eindeutigkeitsregel unterlegene Zeile entfernt, bis 0 Dopplungen verbleiben (konvergierte
in allen drei Fällen nach 2 Runden).

**Betroffene Artefakte (Zeilen vorher → nachher):**

| Datei | Vorher | Nachher | Entfernt | Anteil |
|---|---|---|---|---|
| `pairs_tessera_full_dedup.parquet` | 15.093.377 | 13.553.817 | 1.539.560 | 10,20% |
| `pixels_tessera_dedup.parquet` | 8.257.024 | 7.268.807 | 988.217 | 11,97% |
| `landuse_pairs_tessera_dedup.parquet` | 4.417.428 | 4.359.468 | 57.960 | 1,31% (netto — die primäre Maske entfernt VOR der bestehenden Max-Coverage-Sampling-Deckelung [max 100 Pixel/Polygon], die ohnehin viele der betroffenen Zeilen entfernt hätte; brutto vor Sampling: 1.547.169 primär + 5.122 Restauflösung von 21.016.428 Pixel-Polygon-Zeilen) |

Alle drei Dateien: **0 verbleibende Dopplungs-Weltorte** nach Verifikation. `pairs_tessera_full`
und `landuse_pairs_tessera` durchliefen je einen eigenen, 2-Runden-Restauflösungsschritt
(3.258 bzw. 5.122 zusätzlich entfernte Zeilen); `pixels_tessera_dedup` wird direkt aus dem
bereits dedupliierten `pairs_tessera_full_dedup` aggregiert (`aggregate_pixel_pairs.py`,
unverändert) und brauchte keinen eigenen Restauflösungsschritt.

**Nebenbefund (`diag_removal_rate_breakdown.py`): 5,14% der ursprünglichen,
undeduplizierten `pixels_tessera.parquet`-Zeilen lagen außerhalb des Greater-London-AOI-
Polygons.** Zerlegung der 988.217 aus `pixels_tessera` entfernten Zeilen:

| Ursache | Zeilen | Anteil an allen | Anteil an Entfernten |
|---|---|---|---|
| Eindeutigkeitsmaske (innerhalb AOI, aber nicht nächste Kachel) | 561.759 | 6,80% | 56,8% |
| **AOI-Polygon-Maske (außerhalb Greater London)** | **424.795** | **5,14%** | **43,0%** |
| Restauflösung (Grenzlinien-Rest) | 1.663 | 0,02% | 0,2% |

Die 424.795 Zeilen liegen unabhängig von jeder Eindeutigkeitsfrage außerhalb des Polygons —
das heißt, das **alte, undeduplizierte** `pixels_tessera.parquet` enthielt bereits vor jeglicher
Kachel-Überlappungs-Korrektur Trainingspaare außerhalb der eigentlichen Studienregion.
Wahrscheinliche Ursache: die POI-Extraktion nutzte vermutlich die Geofabrik-PBF-
Bounding-Box statt der präzisen `admin_level=5`-Verwaltungsgrenze (dieselbe Grenze, die auch
als AOI-Polygon in dieser Aufgabe verwendet wird) — nicht unabhängig re-verifiziert, plausibelste
Erklärung angesichts des Musters (viele Zeilen knapp aber eindeutig außerhalb eines rechteckigen
Bereichs um die eigentliche Polygongrenze).

**Splits-Rebuild** (`check_splits_rebuild_dedup.py`, `SEED=0`, unverändert gegenüber
`build_splits.py`): Vereinigung aus `pixels_tessera_dedup` (1.723 Zellen) und unverändertem
`pixels_ae` (1.818 Zellen) ergibt dieselben 1.818 distinkten 1-km-Zellen wie das bestehende
`data/splits/splits_london.csv` — **0 Zellen nur in einer der beiden Mengen, 0/1.818 Zellen mit
abweichendem Split**. Erwartbar: Dedup entfernt Pixel innerhalb bestehender Zellen, eliminiert
aber keine ganze Zelle — die Zellmenge (und damit die Seed-0-Permutation) bleibt unverändert.

**`verify_provenance.py`** blieb nach dem Dedup unverändert grün (prüft die alten, unangetasteten
Dateien). Zusätzlich manuell `verify_pair()` gegen die neuen `_dedup`-Dateien aufgerufen:
konsistent (1.916.851 gemeinsame Pixel).

**[Stand 2026-09-21: AlphaEarth-Dedup-Daten existieren inzwischen, Training/Auswertung stehen noch aus — siehe Abschnitt "AlphaEarth-Vollständige-Dedup" weiter unten; dieser Absatz beschreibt den Stand 2026-09-18/19.]** AlphaEarth ist von diesem gesamten Schritt 1 (und auch Schritt 2/3) NICHT betroffen — es
bleibt bei den bisherigen, undeduplizierten `pixels_ae`/`pairs_ae_full`/`landuse_pairs_ae`.
AlphaEarths eigene Kachel-Überlappung (≈5.940 km² Bounding-Box-Overlap, aber nur ≈0,1% echter
Dual-valid-Data-Overlap, siehe frühere Diagnose) wurde in dieser Aufgabe **nicht** bearbeitet.
**Alle TESSERA-vs-AlphaEarth-Backbone-Vergleiche in Schritt 3 (Retrieval, Land-Use, SDM) sind
deshalb weiterhin nicht direkt vergleichbar** — TESSERA-Zahlen beruhen auf deduplizierten,
AlphaEarth-Zahlen auf undeduplizierten Daten. Ein direkter Backbone-Vergleich TESSERA-dedup vs.
AlphaEarth ist erst nach einer analogen AlphaEarth-Dedup-Runde methodisch sauber möglich.

## TESSERA-Vollständige-Dedup: Auswertungen mit neuem Checkpoint (Schritt 3, Hauptaufgabe 2026-09-19)

Vollständige Zahlen (Retrieval, Land-Use-Probe inkl. Zellebenen-Bootstrap, Class-Retrieval inkl.
Duplikat-Befund, SDM Arm-C-Neuauswertung inkl. gepaartem Bootstrap), jeweils mit Baseline,
Konfidenzintervall und Quelldatei: **`results/tessera_dedup_summary.md`** (nicht hier dupliziert,
um Divergenz zwischen zwei Kopien derselben Zahlen zu vermeiden). Zugrundeliegende Skripte:
`run_retrieval_baselines_tessera_v1dedup.py`, `build_landuse_pixel_cells_tessera_dedup.py`,
`run_landuse_probe_tessera_v1dedup.py`, `check_class_retrieval_duplicates.py`,
`run_class_retrieval_tessera_v1dedup.py`, `build_sdm_arm_c_tessera_v1dedup.py`,
`build_sdm_phase7_metrics_tessera_v1dedup.py` (alle committet). Der Land-Use-Zellebenen-Bootstrap
nutzt einen Fix (`run_landuse_probe.py::test_bootstrap_by_cell()`), der bereits als uncommittete
Vorarbeit vorlag und in dieser Aufgabe committet und für den neuen Checkpoint angewendet wurde.

## TESSERA-Vollständige-Dedup: Training (Schritt 2, Hauptaufgabe 2026-09-18/19)

**Der Alt-Neu-Vergleich `checkpoints/tessera_v1` (0,4581) vs. `tessera_v1_dedup` (0,4712) war
NICHT belastbar** (Jakob, 2026-09-19, korrekt erkannt): der Dedup-Lauf hatte MEHR
Trainingszeilen (5.797.503) als der ursprünglich als Vergleichsbasis herangezogene alte
Log-Wert (4.105.455), obwohl Dedup nur entfernen kann. Deshalb ein Kontrolllauf
(`tessera_v1_predup_control`): identische Config/identischer `torch.manual_seed(0)` wie
`tessera_v1_dedup`, aber auf der **unveränderten, undeduplizierten** `pixels_tessera.parquet`.
Kontroll- und Dedup-Lauf unterscheiden sich dadurch **ausschließlich** in der Deduplizierung.

**Der für die Arbeit geltende Vergleich ist Kontrolllauf vs. Dedup-Lauf, NICHT alter
Checkpoint vs. Dedup-Lauf.** Der alte Checkpoint dient nur noch der Dokumentation.

| | Alter Checkpoint (`tessera_v1`) | Kontrolllauf (`tessera_v1_predup_control`) | Dedup-Lauf (`tessera_v1_dedup`) |
|---|---|---|---|
| Trainingszeilen | 4.105.455 (siehe Klärung unten) | 6.583.295 | 5.797.503 |
| Val-Zeilen | 480.817 | 843.989 | 742.439 |
| Beste Epoche | 26 | 32 | 48 |
| recall@10_diag (Val, beste Epoche) | 0,4581 | **0,4602** | **0,4712** |
| Early Stopping nach Epoche | 36 | 42 | 58 |
| Finale Temperatur (bei Stop) | 27,51 | 27,71 | 28,95 |
| Seed | **keiner** (siehe unten) | `torch.manual_seed(0)` | `torch.manual_seed(0)` |

**Gültiger Dedup-Effekt (Kontrolllauf → Dedup-Lauf, gleicher Seed, gleiche Config):**
recall@10_diag (Val, beste Epoche) 0,4602 → 0,4712 (**+0,0110**). Rein deskriptiv berichtet,
keine Signifikanzaussage (Einzellauf je Konfiguration, kein Seed-Sweep).

### Seed-Status je Lauf (train.py setzt im echten Trainingspfad KEINEN eigenen Seed)

`train.py::train()` ruft `torch.manual_seed()` an keiner Stelle auf — nur die Smoke-Tests
und `overfit_sanity_check()` tun das. Jeder reale Trainingslauf ist deshalb nur dann geseedet,
wenn der AUFRUFENDE Code (Wrapper-Skript) das vorher explizit setzt:

| Lauf | Geseedet? | Wie |
|---|---|---|
| `checkpoints/tessera_v1/best.pt` (alt, 4.9.) | **Nein** | direkter `python train.py --config ...`-Aufruf, kein Seed gesetzt |
| `checkpoints/ae_v1/best.pt` | **Nein** (nicht erneut geprüft, gleiches Muster zu erwarten) | direkter `python train.py --config ...`-Aufruf |
| `checkpoints/tessera_v1_predup_control/best.pt` | **Ja**, `torch.manual_seed(0)` | `train_tessera_v1_predup_control.py` |
| `checkpoints/tessera_v1_dedup/best.pt` | **Ja**, `torch.manual_seed(0)` | `train_tessera_v1_dedup.py` |

`train.py` selbst wurde nicht geändert (kein Seed nachgerüstet) — die beiden neuen Läufe
seeden nur über ihre eigenen Wrapper-Skripte.

### Klärung: welche Daten hat der alte Checkpoint (`checkpoints/tessera_v1/best.pt`, Epoche 26) gesehen?

**Nicht abschließend geklärt — hier dokumentiert, nicht plausibel gemacht.** Geprüfte Belege,
die sich teilweise widersprechen:

1. **Eingebettete Checkpoint-Config** (`ckpt['cfg']`, von `train.py` beim Speichern selbst
   serialisiert): zeigt Gen-2-Pfade — `pixels_path: data/processed/pixels_tessera.parquet`,
   `text_embeddings_path: data/processed/pixel_text_embeddings.parquet`,
   `norm_stats_path: data/processed/norm_stats_tessera_pixels.json`. Kein `splits_path`/
   `pairs_path`/`poi_text_path` (Gen-1-Config-Schema) vorhanden.
2. **Trainings-Log** (`logs/train_tessera_v1_20260904_220621.log`, Start 4.9. 22:06:21):
   druckt "4105455 Trainings-/480817 Val-Zeilen" — passt NICHT zum heutigen
   `pixels_tessera.parquet` (6.583.295 Train-/843.989 Val-Zeilen, Datei unveraendert seit
   3.9. 14:52:42 laut mtime/Checksum, also bereits VOR dem Trainingsstart in ihrem heutigen
   Zustand).
3. **Git-Historie:** `configs/tessera_v1.yaml`/`train.py`/`dataset.py` zeigen am letzten
   Commit VOR dem Trainingsdatum (`7486012`, 1.9. 21:41) noch das Gen-1-Schema
   (`pairs_path: data/processed/pairs_tessera.parquet`, `poi_text_path`, `splits_path`).
   Der naechste Commit, der das Gen-2-Schema einfuehrt (`eaee511 "feat: gen-2 pixel-level
   pipeline"`), liegt aber erst am 8.9. — **zwischen `29b7e19` (1.9. 21:41:57) und `525ad4d`
   (8.9. 16:49:18) liegt eine 7-taegige Luecke ohne jeden Commit** (`git reflog`
   verifiziert). Der tatsaechliche Arbeitsstand am 4.9. (Trainingszeitpunkt) ist damit aus
   der Git-Historie NICHT rekonstruierbar — er kann Gen-2-Code bereits lokal (uncommitted)
   enthalten haben, ohne dass das irgendwo sichtbar waere.
4. **Starkes Gegenindiz zu Punkt 2** (Kernbefund dieser Klaerung): die AKTIVE
   `data/processed/norm_stats_tessera_pixels.json` hat mtime **4.9. 22:06:53** — also
   Sekunden NACH dem Start des fraglichen Trainingslaufs (`apply_normalization()` fittet und
   speichert die Statistik unmittelbar bei Trainingsbeginn). Diese Datei ist **bit-identisch**
   (max. Differenz 0,0 ueber alle 128 Kanaele, siehe `diag_norm_stats_consistency.py`) mit
   einer frischen Z-Score-Anpassung auf dem TRAIN-Split des HEUTIGEN
   `pixels_tessera.parquet`. Das ist ein sehr starkes Indiz (keine acht Nachkommastellen ueber
   128 Kanaele treffen zufaellig exakt zu), dass der Trainings-Split dieses Laufs tatsaechlich
   der heutige 6.583.295-Zeilen-Train-Split war — die gedruckte Zeilenzahl im Log waere dann
   ein Anzeige-/Logging-Fehler im damaligen `train.py`, keine echte Abweichung der
   tatsaechlich verwendeten Daten.
5. Ein archiviertes, ANDERES, AELTERES tessera_v1 (`data/archive_gen1/checkpoints/tessera_v1/`,
   1.9. 16:19/16:36 — bereits vor dieser Session als Gen-1 dokumentiert) trainierte
   nachweislich auf dem separat archivierten `data/archive_gen1/processed/pairs_tessera.parquet`
   (5.287.941 POI-Pixel-Zeilen, POI-Ebene) — das ist ein ANDERER, frueherer Checkpoint als der
   hier untersuchte (`checkpoints/tessera_v1/best.pt`, Epoche 26, 4.9.), nicht dessen
   Datenquelle.

**Einordnung:** Punkt 4 spricht dafuer, dass der alte Checkpoint trotz der irrefuehrenden
Log-Zeile wahrscheinlich doch auf der heutigen Datenmenge (Gen-2, undedupliziert) trainiert
wurde. Sicher ist das nicht - die genaue Ursache der falschen Log-Zeilenzahl bleibt ungeklaert.
Unabhaengig davon bleibt der Kontrolllauf-vs-Dedup-Vergleich der methodisch sauberere Weg
(gleicher Seed, keine Restunsicherheit ueber die Datenmenge), weil der alte Checkpoint ohnehin
unseeded war (siehe Seed-Tabelle oben) und deshalb fuer einen fairen Punktvergleich nicht
geeignet ist.

## AlphaEarth-Vollständige-Dedup: Datenbasis & Split-Entscheidung (Schritt 1, Hauptaufgabe 2026-09-21)

Dieselbe Prozedur wie bei TESSERA (Eindeutigkeitsmaske + AOI-Polygon-Maske + Restauflösung über
Weltkoordinaten; Logik unverändert aus `tessera_dedup_utils.py`, `backbone="ae"`; Skripte
`build_pairs_ae_dedup.py`, `build_pixels_ae_dedup.py`, `build_landuse_pairs_ae_dedup.py`,
`ae_dedup_utils.py`). Alte Dateien unverändert; neue Dateien tragen den Suffix `_dedup`.

| Datei | Vorher | Nachher | Entfernt | Davon Eindeutigkeit / AOI-Polygon / Restauflösung |
|---|---|---|---|---|
| `pairs_ae_full_dedup.parquet` | 14.008.309 | 13.552.617 | 455.692 (3,25 %) | 6.720 (1,5 %) / 448.867 (98,5 %) / 105 (0,0 %) |
| `pixels_ae_dedup.parquet` | 7.664.167 | 7.268.678 | 395.489 (5,16 %) | 5.605 (1,4 %) / 389.831 (98,6 %) / 53 (0,0 %) |
| `landuse_pairs_ae_dedup.parquet` | 4.363.718 | 4.359.867 | 3.851 (0,09 % netto) | vor Coverage-Dedup/Sampling, von 19.508.778 Pixel-Polygon-Zeilen: 22.071 (56,2 %) / 17.090 (43,5 %) / 101 (0,3 %) |

Bei AlphaEarth dominiert (anders als bei TESSERA) fast vollständig die AOI-Maske: die
Rohkacheln (30N/31N, 8192×8192) überlappen als Bounding Box zu ≈88 %, echte Datenüberlappung ist
aber nur ≈0,1 % (siehe Text-Query-Phase-2). Beim Landuse-Artefakt ist das Bild umgekehrt
(Eindeutigkeit 56 %), weil Urban Atlas bereits vor dem Join auf die AOI geclippt wird und die
AOI-Maske nur noch Randpixel trifft. 0 verbleibende Dopplungs-Weltorte in allen drei Dateien
(im Original `pixels_ae`: 1.432 Dopplungs-Weltorte). `verify_pair()` für `pixels_ae_dedup` ↔
`landuse_pairs_ae_dedup`: konsistent (1.915.221 gemeinsame Pixel); `verify_provenance.py` für die
alten Paare unverändert grün. `pixels_ae_dedup` und `pixels_tessera_dedup` decken praktisch
dieselben Pixel ab (7.268.678 vs. 7.268.807 Zeilen, Differenz 129) und dieselben 1.723 Zellen.

**Entscheidung Splits (Jakob, 2026-09-21): `data/splits/splits_london.csv` bleibt unverändert,
KEIN Neubau.** `check_splits_rebuild_dedup_ae.py` (SEED=0, `assign_splits()` importiert) baut
die Zellen auf dem Pixelbestand NACH beiden Deduplizierungen neu: nur noch **1.723** statt 1.818
Zellen (die 95 fehlenden Zellen liegen komplett außerhalb des AOI-Polygons und haben in keinem
der beiden Backbones mehr Pixel; Split der 95 in `splits_london.csv`: 70 train / 11 val /
14 test). Ein Neubau würde wegen der geänderten Zellmenge die globale Seed-0-Permutation
verändern und **574 von 1.723 Zellen (33,3 %) umsortieren** (Kreuztabelle alt→neu:
train→test 141, train→val 134, val→train 139, test→train 130, …). Das würde die
TESSERA-dedup-Ergebnisse (Training, Auswertung) ungültig machen — sie wurden auf den bestehenden
Splits gemessen — und ebenso den auf der Zellzuordnung beruhenden SDM-LSOA-Split
(`sdm_lsoa_splits_london.csv`, `sdm_lsoa_cell_overlay_london.csv`). Der räumliche Split ist eine
feste Größe der Arbeit; die 95 Zellen bleiben in `splits_london.csv` stehen und sind schlicht
leer. Dokumentiert in `data/processed/diag_splits_rebuild_dedup_ae_comparison.csv` (nur
Dokumentation, ersetzt `splits_london.csv` nicht).

**Split-Größen (Pixel, Anteil an allen Pixeln des Artefakts):**

| Datei | Gesamt | train | val | test |
|---|---|---|---|---|
| `pixels_ae` (alt) | 7.664.167 | 6.095.537 (79,53 %) | 791.022 (10,32 %) | 777.608 (10,15 %) |
| `pixels_ae_dedup` | 7.268.678 | 5.797.540 (79,76 %) | 742.550 (10,22 %) | 728.588 (10,02 %) |
| `pixels_tessera` (alt) | 8.257.024 | 6.583.295 (79,73 %) | 843.989 (10,22 %) | 829.740 (10,05 %) |
| `pixels_tessera_dedup` | 7.268.807 | 5.797.503 (79,76 %) | 742.439 (10,21 %) | 728.865 (10,03 %) |

Zellen (1-km, `splits_london.csv`): 1.818 (1.454 / 182 / 182); davon mit Pixeln in den
`_dedup`-Beständen: 1.723 (1.384 / 171 / 168). `landuse_pairs_ae_dedup`: 273 Pixel mit
`split='missing'` (Original: 317; UA-Pixel in Zellen ohne POI-Pixel, markiert, nicht verworfen).

**Nebenbefund: die Eindeutigkeitsmaske entfernt teils Orte ganz, statt zu deduplizieren**
(`diag_ae_uniqueness_loss.py --backbone {ae,tessera}`, Ergebnisse
`results/diag_uniqueness_loss_{ae,tessera}.csv`; nur dokumentiert, NICHT korrigiert, die
Prozedur bleibt für beide Backbones identisch). Bei manchen durch die Eindeutigkeitsmaske
entfernten `pixels_*`-Zeilen existiert in der Gewinner-Kachel keine Zeile (die Maske spricht
den Ort der Gewinner-Kachel zu, der POI war aber nur der Verlierer-Kachel zugewiesen):

| | AlphaEarth | TESSERA |
|---|---|---|
| durch Eindeutigkeitsmaske entfernte Zeilen (innerhalb AOI) | 5.605 | 561.759 |
| A) echte Dopplung (Rival hat Zeile im Original) | 1.407 | 531.439 |
| B) Ort verschwindet (Rival gültig, aber keine Zeile) | 4.197 (0,055 % von `pixels_ae`) | 30.320 (0,37 % von `pixels_tessera`) |
| — B, Rivale in ANDERER UTM-Zone (Zonenzuweisung `assign_utm_zone`) | 4.197 | 10.914 |
| — B, Rivale in GLEICHER UTM-Zone | 0 | 19.406 |
| — B ohne jede Zeile in der 3×3-Nachbarschaft des Rivalen-Pixels, Rivale andere Zone | 3.372 | 9.109 |
| — B ohne Nachbarzeile, Rivale gleiche Zone | 0 | 0 |

Deutung: Nur der zonenübergreifende Teil kann auf `assign_utm_zone` zurückgehen (dort existiert
der Zonenfilter; er weist jeden POI genau einer Kachel zu). Belastbar als „Orte, die durch die
Zonenzuweisung ganz verschwinden" sind die **B-Fälle mit anderer Zone und ohne Nachbarzeile:
TESSERA 9.109, AlphaEarth 3.372** (Obergrenze ohne Nachbarprüfung: 10.914 bzw. 4.197). Die
19.406 TESSERA-Fälle mit Rivalen in der GLEICHEN Zone (dort gibt es keinen Zonenfilter) haben
ausnahmslos eine Zeile in der 3×3-Nachbarschaft des Rivalen-Pixels: das ist ein Effekt der nicht
phasengleichen 10-m-Gitter (das POI-Polygon deckt das Pixel der einen Kachel ab, das
nächstliegende Pixel der anderen aber nicht), keine Zonenzuweisung — der Ort ist dort in der
Nachbarschaft weiterhin durch eine Zeile vertreten. Die Nachbarprüfung ist eine Näherung (die
Nachbarzeile muss nicht denselben POI-Ort meinen). Restgröße: `check_location_absent_in_dedup_result`
(kein Rivalen-Pixel im Dedup-Ergebnis) TESSERA 33.521, AE 4.259 — inkl. 3.201 bzw. 61 A-Fälle, deren
Rivalen-Zeile selbst ebenfalls entfernt wurde. Größenordnung insgesamt klein
(≤ 0,4 % der Original-Pixel), verändert die Aussage „0 verbleibende Dopplungen" nicht.

## Noch nicht archiviert, aber ebenfalls Alt-/Nebenmaterial (zur Kenntnisnahme, nicht Teil des Archivierungsauftrags)

- `data/interim/poi_text.parquet` (31.8., von `poi_text_full.parquet` abgelöst)
- `data/processed/text_embeddings.parquet` (1.9., POI-Ebene-Textembeddings, von `pixel_text_embeddings.parquet` abgelöst — Achtung: laut `compute_pixel_text_embeddings.py`-Docstring für HILUCS-Zero-Shot-Prompts evtl. noch separat gebraucht, nicht vorschnell löschen)
- `data/interim/pois_london_test.parquet`, `pois_london_test_filtered.parquet`, `pois_london_object_qid_check.parquet` (Test-/Zwischenstände der Extraktion)
- `run_ae_join.py`, `run_tessera_join.py` — **archiviert** (`data/archive_gen1/scripts/`, 3.9.). Waren alte Join-Skripte mit `OUT_PATH` auf die jetzt ebenfalls archivierten `pairs_ae.parquet`/`pairs_tessera.parquet`; abgelöst durch `run_ae_join_full.py`/`run_tessera_join_full.py`. Liegen jetzt nicht mehr im aktiven Repo-Root, ein versehentlicher Re-Run ist damit ausgeschlossen.
