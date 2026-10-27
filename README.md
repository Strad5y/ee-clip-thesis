# Semantic Enrichment of Earth Embeddings through Contrastive Learning with Text Labels

Code zur Bachelorarbeit von Jakob Leander Strade (Leuphana Universität Lüneburg, 2026).
Erstgutachter: Prof. Dr. Ricardo Usbeck, Zweitgutachter: Kai Moltzen.

Gefrorene Earth-Embeddings (TESSERA, 128-dim; AlphaEarth, 64-dim) werden für Greater London
pixelweise mit Texten aus OpenStreetMap und Wikidata gepaart. Trainiert werden nur zwei kleine
Projektionsköpfe mit SigLIP-Loss; Backbones und Text-Encoder (Qwen3-Embedding-4B) bleiben
eingefroren. Ausgewertet wird über Text-Query-Karten, Land-Use-Klassifikation (Urban Atlas) und
Socioeconomic Distribution Mapping (NS-SeC, Census 2021), jeweils mit drei Armen:
roh (A), zufällig projiziert (B), trainiert projiziert (C).

Das Repo enthält nur die Skripte, Konfigurationen, Checkpoints, Ergebnisdateien und Logs, aus
denen die Zahlen und Abbildungen der Arbeit stammen.

## Aufbau des Repos

Die Skripte liegen flach im Wurzelverzeichnis und importieren sich gegenseitig über den
Dateinamen. Deshalb gibt es keine Unterordner pro Kapitel. Die Tabellen unten ordnen jedes
Skript dem passenden Abschnitt der Arbeit zu, in der Reihenfolge, in der die Pipeline läuft.

| Ordner | Inhalt |
|---|---|
| `*.py`, `src/` | Pipeline-Skripte (siehe unten) |
| `src/eval/text_query/` | Text-Query-Karten (Abschnitt 4.3.1, 6.3) |
| `configs/` | Trainingskonfigurationen der beiden finalen Läufe |
| `checkpoints/` | Trainierte Projektionsköpfe (`best.pt` = in der Arbeit verwendet) |
| `results/` | Ergebnis-CSVs der Tabellen 6.1/6.2 und LaTeX-Tabelle 6.1 |
| `outputs/text_query/` | Karten aus Abbildung 6.1 (`maps/`) und Anhang 9.1 (`maps_untrained/`) |
| `logs/` | Protokolle der Läufe, aus denen die Ergebnisse stammen |
| `data/` | Kleine Hilfsdaten: AOI London, AlphaEarth-Kachelindex, räumliche Splits |
| `queries/text_queries.md` | Text-Queries aus Tabelle 4.1 |
| `wikidata tags/osm_wikidata_keys.xlsx` | Klassentabelle OSM-Tag → Wikidata-QID (Anhang) |

Gemeinsam genutzte Module: `config.py` (Pfade), `dequantize.py`, `normalize.py` (z-Score),
`spatial_join.py` (POI × Raster mit exactextract), `chunked_join_runner.py` (blockweise
Verarbeitung, Abschnitt 5.1), `class_table.py`, `poi_classes.py`.

## Pipeline nach Kapiteln der Arbeit

### 3.4 Daten beschaffen

| Schritt | Skript |
|---|---|
| OSM-Extrakt Greater London (Geofabrik) | `download_osm_extracts.py` |
| Studiengebiet aus OSM-Grenzrelation | `src/extract_aoi.py` → `data/aoi/london.gpkg` |
| AlphaEarth-Kacheln: Header per HTTP lesen, Index mit AOI schneiden (UTM 30N/31N) | `build_ae_index.py --region london` |
| AlphaEarth-Kacheln laden | `download_ae_tiles.py --region london` |
| AlphaEarth dequantisieren und mosaikieren | `build_mosaic.py` |
| TESSERA über geotessera laden | `make_tessera_tif.py --region london` |

### 3.4.2–3.4.5 POIs, Tag-Filter, Wikidata

| Schritt | Skript |
|---|---|
| POI-Extraktion (13 Keys + Objekte mit eigener QID), Wertefilter | `src/extract_pois.py --region london` |
| Wikidata-Beschreibungen der Klassen-QIDs | `src/fetch_class_qid_descriptions.py` |
| Wikidata-Beschreibungen objekteigener QIDs | `src/fetch_object_qid_descriptions.py` |

### 3.5 Texte (Templates, Multi-Tag, Sonderfälle)

| Schritt | Skript |
|---|---|
| Templates, Key-Frames (Tabelle 3.2), Fälle A–C | `build_poi_text.py` |

### 3.2 / 5.1 Pixel-Paare

| Schritt | Skript |
|---|---|
| Spatial Join POI × Pixel (blockweise, 20.000 POIs pro Block) | `run_tessera_join_full.py`, `run_ae_join_full.py` |
| Pixelweise Aggregation, Sortierung nach Coverage (3.5.3) | `src/aggregate_pixel_pairs.py` |
| Deduplizierung (Kachelüberlappungen, AOI-Maske) | `build_pixels_tessera_dedup.py`, `build_pixels_ae_dedup.py`, `build_pairs_tessera_dedup.py`, `build_pairs_ae_dedup.py` (Hilfsmodule: `tessera_dedup_utils.py`, `ae_dedup_utils.py`, `diag_removal_rate_breakdown.py`, `check_tile_overlap_side_effects.py`) |
| Validierung der Paardateien | `validate_pairs.py` |

### 3.5.4 Text-Encoder

| Schritt | Skript |
|---|---|
| Qwen3-Embedding-4B, Last-Token-Pooling, MRL 2560 → 1024, L2-Norm, Cache | `compute_pixel_text_embeddings.py` (nutzt `compute_text_embeddings.py`, `compute_poi_text_embeddings.py`) |

### 3.7 Räumlich geblockte Splits

| Schritt | Skript |
|---|---|
| 1-km-Gitter, 80/10/10 auf Zellebene | `build_splits_pixels.py` (nutzt `build_splits.py`) → `data/splits/splits_london.csv` |

### 3.6 / 5.2 Training

| Schritt | Datei |
|---|---|
| Projektionskopf Linear–GELU–Dropout–Linear + L2 | `model.py` |
| SigLIP-Loss, lernbare Temperatur/Bias, Maske für identische Texte (3.6.2) | `losses.py` |
| Dataset/Batches | `dataset.py` |
| Trainingsschleife, Early Stopping auf Val-Recall@10 | `train.py` |
| Finale Läufe (Tabelle 5.1) | `train_tessera_v1_dedup.py`, `train_ae_v1_dedup.py` mit `configs/{tessera,ae}_v1_dedup.yaml` |
| Test-Recall@10 (0,506 / 0,547) | `run_retrieval_baselines_tessera_v1dedup.py`, `run_retrieval_baselines_ae_v1dedup.py` → `results/retrieval_baselines_*_v1_dedup.csv` |

Die LR-Kalibrierung (5.2) und der Vergleich mit/ohne Maske (3.6.2) sind in
`logs/train_*_lr*_5ep_*.log`, `logs/train_ae_v1_nomask_*.log` und
`logs/train_ae_v1_20260901_112113.log` protokolliert.

### 4.2 / 6.1 Land-Use-Klassifikation (Tabelle 6.1)

| Schritt | Skript |
|---|---|
| Urban Atlas auf das Pixelraster, Klasse per maximaler Überlappung | `build_landuse_pairs.py`, `build_landuse_pairs_tessera_dedup.py`, `build_landuse_pairs_ae_dedup.py` |
| Gitterzelle je Pixel (für den Zell-Bootstrap) | `build_landuse_pixel_cells_tessera_dedup.py`, `build_landuse_pixel_cells_ae_dedup.py` |
| Gemeinsame Funktionen (Arme A/B/C, Zell-Bootstrap 4.3.2) | `run_landuse_probe.py`, `eval_landuse.py`, `run_landuse_cell_bootstrap.py` |
| Drei Arme, LogReg | `run_landuse_probe_tessera_v1dedup.py`, `run_landuse_probe_ae_v1dedup.py` |
| LogReg (full), z-standardisiert, Macro-P/R/F1 auf fester Klassenmenge | `run_landuse_probe_std.py --backbone {tessera,ae}` (nutzt `run_landuse_probe_prec_rec.py`) |
| LogReg (subsample), CatBoost, MLP | `run_landuse_probe_heads_std.py --backbone {tessera,ae}` (nutzt `eval_landuse_heads.py`, `run_landuse_probe_heads.py`) |
| Bootstrap-CIs für den Random-Arm | `run_landuse_probe_heads_std_random_ci.py --backbone {tessera,ae}` |
| Mehrheitsklassen-Baseline (4.1) | `run_landuse_majority_baseline_tessera_v1dedup.py`, `run_landuse_majority_baseline_ae_v1dedup.py` |

Ergebnisse: `results/landuse_probe_prec_rec_*_std.csv` (LogReg full),
`results/landuse_probe_heads_*_std*.csv` (übrige Köpfe), `results/luc_std_table.tex` (Tabelle 6.1).

### 4.3 / 6.2 Socioeconomic Distribution Mapping (Tabelle 6.2)

| Schritt | Skript |
|---|---|
| NS-SeC-Zielverteilungen je LSOA (Census 2021, TS062) | `build_sdm_targets.py` |
| LSOAs den Gitterzellen/Splits zuordnen | `build_sdm_lsoa_splits.py` → `data/splits/sdm_lsoa_*.csv` |
| Projizierte Embedding-Felder (Arme B/C) | `build_sdm_projected_field.py`, `retile_sdm_projected.py` |
| Mittelung je LSOA | `build_sdm_pooled_embeddings.py`, `build_sdm_pooled_embeddings_dedup.py`, aufgerufen von `build_sdm_arm_ab_tessera_dedupnorm.py`, `build_sdm_arm_c_tessera_v1dedup.py`, `build_sdm_ae_v1dedup.py` |
| Vorhersagekopf, KL/L1/Chebyshev, 5 Seeds, Zell-Bootstrap | `build_sdm_phase7_metrics_seedmean_std.py` (nutzt `build_sdm_phase6_predict.py`, `build_sdm_phase7_metrics.py`, `build_sdm_phase7_metrics_seedmean.py`, `build_sdm_random_embedding_seedmean.py`) |

Ergebnisse: `results/sdm_phase7_metrics_{tessera,ae}_seedmean_std.csv`.

### 4.3.1 / 6.3 Text-Query-Karten (Abbildung 6.1, Anhang 9.1)

| Schritt | Skript |
|---|---|
| Green-Cover-2024-Layer reparieren (Referenz) | `build_greencover_valid.py` |
| Kachelmasken (Eindeutigkeit, AOI) | `src/eval/text_query/masks.py`, `tiles.py` |
| Queries encodieren (Tabelle 4.1) | `src/eval/text_query/encode.py` |
| Kosinus-Similarity je Pixel | `src/eval/text_query/similarity.py` |
| Referenzkarten Grünflächen / Flughäfen | `src/eval/text_query/reference.py` |
| Karten rendern | `src/eval/text_query/maps.py` → `outputs/text_query/maps/` |
| Karten auf untrainierten Embeddings (Anhang) | `src/eval/text_query/similarity_untrained.py` → `outputs/text_query/maps_untrained/` |
| Kontrolle der projizierten Felder | `src/eval/text_query/verify_projected_fields.py` |

## Umgebung

```bash
conda env create -f environment.yml   # Umgebung "eeclip"
conda activate eeclip
```

Gerechnet auf einer NVIDIA DGX Spark (GB10, aarch64). Die Skripte werden aus dem
Wurzelverzeichnis gestartet, z. B. `python train_tessera_v1_dedup.py`.

## Nicht enthaltene Daten

Rohdaten und abgeleitete Tabellen (`*.parquet`, `*.tif`) sind nicht im Repo. Sie sind zu groß,
und die aus OpenStreetMap abgeleiteten Tabellen fallen unter die ODbL (Abschnitt 8.1).
`config.py` legt fest, wo die Pipeline sie erwartet. Quellen:

- OpenStreetMap, Geofabrik-Extrakt Greater London (© OpenStreetMap contributors, ODbL)
- AlphaEarth Foundations Satellite Embedding, annual 2025 (Google / Google DeepMind, CC BY 4.0)
- TESSERA Embeddings 2025 (University of Cambridge, CC0)
- Wikidata (CC0)
- Copernicus Urban Atlas 2021 (© European Union, Copernicus Land Monitoring Service)
- ONS Census 2021 TS062 (NS-SeC) und LSOA-Grenzen 2021 (OGL v3.0)
- GLA Green Cover 2024 (London Datastore)
- Qwen3-Embedding-4B (Apache 2.0)

## Hinweis zu den Kommentaren im Code

Docstrings und Kommentare sind deutschsprachige Arbeitsnotizen aus der Entwicklung. Manche
verweisen auf interne Protokolle (`MANIFEST.md`, `INSTRUCTIONS_*.md`) oder auf überholte
Zwischenstände (z. B. vor der Deduplizierung). Diese Dateien gehören nicht zum Repo. Maßgeblich
sind die oben genannten Skripte und Ergebnisdateien.
