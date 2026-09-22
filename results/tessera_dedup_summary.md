# TESSERA-Vollständige-Dedup: Schritt-3-Ergebnisse (Zusammenfassung)

Stand: 2026-09-19. Checkpoint: `checkpoints/tessera_v1_dedup/best.pt` (Epoche 48), trainiert auf
deduplizierten TESSERA-Daten (siehe `MANIFEST.md`, Abschnitt "Schritt 1"/"Schritt 2"). Alle
"Neu"-Zahlen unten beruhen auf diesem Checkpoint + den `*_dedup`-Datendateien. Alle "Alt"-Zahlen
beruhen auf dem alten `checkpoints/tessera_v1/best.pt` (Epoche 26, Provenienz laut MANIFEST.md
nicht abschließend geklärt) + undeduplizierten Daten.

**Geltungshinweise (für alle Tabellen unten):**
- Jede Datei ist unter ihrem Namen sowohl in `results/` (committet) als auch — Ursprungsquelle —
  unter `data/processed/` (nicht committet, per `.gitignore`) vorhanden; MD5 in
  `results/checksums.txt`.
- Alt↔Neu vermengt bei Retrieval, Land-Use-Probe und Class-Retrieval **immer zwei Effekte**
  (Checkpoint-Wechsel UND Dedup der Auswertedaten) — es gibt für diese drei Komponenten keine
  Zwischenstufe, die beide Effekte trennt. Nur bei SDM existiert eine dritte, isolierende Stufe
  (siehe dortige Tabelle).
- AlphaEarth ist von alldem nicht betroffen (undedupliziert) — kein direkter Backbone-Vergleich
  TESSERA-dedup vs. AlphaEarth.
- Keine Signifikanzangabe ohne ausgewiesenes Konfidenzintervall — wo unten kein CI steht, ist die
  Zahl eine Punktschätzung aus einem Einzellauf (kein Seed-Sweep).

---

## 1. Retrieval (Test-Split)

Quelldatei Neu: `results/retrieval_baselines_tessera_v1_dedup.csv`.
Quelldatei Alt: nicht separat gesichert in `results/` (Referenzdatei aus früherer Sitzung,
`data/processed/retrieval_baselines_tessera_v1.csv`, MD5 in `results/checksums.txt`).
Kein Bootstrap/CI in dieser Auswertung (Punktschätzung über den vollen Test-Split, kein Resampling).

| Bucket | Metrik | Alt | Neu |
|---|---|---|---|
| all (n=829.740 / 728.865) | recall@1_diag | 0,1514 | 0,1564 |
| all | recall@10_diag | 0,4957 | 0,5058 |
| all | ceiling@1_diag | 0,5909 | 0,6078 |
| all | ceiling@10_diag | 0,9446 | 0,9512 |
| all | exploitation@1_diag | 0,2563 | 0,2573 |
| all | exploitation@10_diag | 0,5248 | 0,5317 |
| single (n=563.779 / 480.183) | recall@10_diag | 0,5225 | 0,5310 |
| single | ceiling@10_diag | 0,9221 | 0,9305 |
| single | exploitation@10_diag | 0,5666 | 0,5706 |
| multi (n=265.961 / 248.682) | recall@10_diag | 0,4389 | 0,4570 |
| multi | ceiling@10_diag | 0,9923 | 0,9912 |
| multi | exploitation@10_diag | 0,4423 | 0,4611 |

---

## 2. Land-Use-Probe

Quelldateien Neu (alle in `results/`): `landuse_probe_results_tessera_dedup_ckpt.csv`
(Punktschätzung), `landuse_probe_cell_bootstrap_tessera_dedup_ckpt.csv` (Zellebenen-Bootstrap,
1000×, 167 Test-Zellen), `landuse_probe_paired_diff_cell_tessera_dedup_ckpt.csv` (gepaarter
Zellebenen-Diff trainiert−roh).
Quelldatei Alt: `data/processed/landuse_probe_results_tessera_dedup.csv` (Referenzdatei, MD5 in
`results/checksums.txt`, Provenienz verifiziert: `checkpoints/tessera_v1/best.pt` Epoche 26,
undeduplizierte `landuse_pairs_tessera.parquet`, Log
`logs/run_landuse_probe_tessera_detached_20260905_165433.log`).

**KORRIGIERT 2026-09-22 (Jakob):** Die vorherige Aussage „Der Alt-Lauf hat KEINEN
Zellebenen-Bootstrap" war falsch. Ein Alt-Zellebenen-Bootstrap existiert
(`run_landuse_cell_bootstrap.py`, 14.9., 5000×, derselbe alte Checkpoint/dieselben undeduplizierten
Daten wie oben — Punktschätzungen stimmen exakt mit der Alt-Spalte unten überein, verifiziert),
Quelldateien `data/processed/landuse_probe_bootstrap_comparison.csv` und
`landuse_probe_paired_diff_cell_bootstrap.csv` (MD5 in `results/checksums.txt`, ergänzt im Zuge
der AlphaEarth-Dedup-Aufgabe). Die Alt-Zeile unten ist damit nachträglich um ein CI ergänzt.

**Punktschätzung (bestes val_macro_f1 je Konfiguration):**

| Konfiguration | C (Alt) | Alt test_oa | Alt test_macro_f1 | C (Neu) | Neu test_oa | Neu test_macro_f1 |
|---|---|---|---|---|---|---|
| roh | 10,0 | 0,6024 | 0,2478 | 10,0 | 0,6042 | 0,2511 |
| zufällig projiziert | 1,0 | 0,5975 | 0,2375 | 10,0 | 0,5993 | 0,2397 |
| trainiert projiziert | 10,0 | 0,6119 | 0,2498 | 10,0 | 0,6158 | 0,2532 |

**Fußnote "zufällig projiziert":** Alt und Neu wählen hier unterschiedliche C-Werte (C=1,0 vs.
C=10,0) — das ist keine Inkonsistenz, sondern das Ergebnis derselben C-Grid-Suche
(bestes val_macro_f1 aus {0,1; 1,0; 10,0}) auf zwei verschiedenen Datenständen. Die zufällige
Projektion hat keinen gelernten Zusammenhang mit der Zielgröße, daher ist die C-Wahl hier
empfindlicher gegenüber Rauschen im jeweiligen Val-Split als bei "roh"/"trainiert" (beide Male
C=10,0 in Alt UND Neu).

**Zellebenen-Bootstrap, Alt (5000×, 14.9., korrigiert s.o.) vs. Neu (1000×, 167 Test-Zellen):**

| Konfiguration | Alt test_oa | Alt 95%-CI | Neu test_oa | Neu 95%-CI | Alt macro_f1 | Alt 95%-CI | Neu macro_f1 | Neu 95%-CI |
|---|---|---|---|---|---|---|---|---|
| roh | 0,6020 | [0,5744; 0,6278] | 0,6035 | [0,5803; 0,6287] | 0,2452 | [0,2195; 0,2654] | 0,2479 | [0,2226; 0,2723] |
| zufällig projiziert | 0,5972 | [0,5692; 0,6233] | 0,5986 | [0,5749; 0,6249] | 0,2357 | [0,2117; 0,2559] | 0,2371 | [0,2157; 0,2607] |
| trainiert projiziert | 0,6116 | [0,5846; 0,6369] | 0,6151 | [0,5921; 0,6402] | 0,2484 | [0,2258; 0,2710] | 0,2501 | [0,2294; 0,2681] |

**Gepaarter Zellebenen-Bootstrap (trainiert − roh), Alt vs. Neu:**

| Metrik | Alt Diff | Alt 95%-CI | Alt signifikant | Neu Diff | Neu 95%-CI | Neu signifikant |
|---|---|---|---|---|---|---|
| oa | +0,0096 | [0,0070; 0,0122] | **ja** | +0,0116 | [0,0087; 0,0145] | **ja** |
| macro_f1 | +0,0033 | [−0,0039; +0,0151] | nein | +0,0023 | [−0,0105; +0,0121] | nein |

Beide Zellebenen-Bootstraps (Alt und Neu) zeigen dasselbe Muster: overall accuracy verbessert
sich signifikant von roh zu trainiert projiziert, macro-F1 nicht signifikant.

---

## 3. Class-Retrieval

Quelldatei Neu: `results/class_retrieval_tessera_v1_dedup.csv` (Zeile `MACRO_MEAN`).
Quelldatei Alt: `data/processed/class_retrieval_tessera.csv` (Referenzdatei, MD5 in
`results/checksums.txt`).
Kein Bootstrap/CI (Makro-Mittel über 186 Klassen mit ≥100 relevanten Test-Pixeln je Seite,
keine Resampling-Prozedur implementiert).

| Metrik | Alt (n_gallery=829.740) | Neu (n_gallery=728.865) |
|---|---|---|
| random_baseline | 0,00741 | 0,00758 |
| p@10 | 0,07407 | 0,07366 |
| p@50 | 0,07429 | 0,07495 |
| p@100 | 0,06989 | 0,07462 |
| r_precision | 0,04712 | 0,04942 |
| average_precision | 0,03426 | 0,03534 |

**Duplikat-Befund** (Vorab-Check für `run_class_retrieval.py`, keine eigene Datei — direkt auf
`pixels_tessera.parquet` bzw. `pixels_tessera_dedup.parquet` berechnet, reproduzierbar über
`check_class_retrieval_duplicates.py`, committet):
- Alt-Galerie (Test-Split `pixels_tessera.parquet`): **49.221 Dopplungs-Weltort-Paare, 98.442
  betroffene Zeilen von 829.740 (11,86%)**.
- Neu-Galerie (Test-Split `pixels_tessera_dedup.parquet`): **0 verbleibende Dopplungs-Weltorte**
  (verifiziert mit derselben Erkennungsfunktion `find_duplicate_rows()`).

---

## 4. SDM (nur Arm C für TESSERA neu; Arm A dedupliziert, unverändert seit Schritt 1)

Dreistufiger Vergleich isoliert Mask-Effekt (Original → Schritt-1-Dedup) von Checkpoint-Effekt
(Schritt-1-Dedup → Schritt-3 neuer Checkpoint):

| Stufe | Quelldatei | tessera/a KL | tessera/c KL |
|---|---|---|---|
| Original (vor jeglichem Dedup) | `data/processed/sdm_phase7_metrics.csv` (Referenz, MD5 in checksums.txt) | 0,03356 | 0,03556 |
| Schritt-1-Dedup (A dedup, C = alter Checkpoint) | `results/sdm_phase7_metrics_dedup.csv` | 0,03375 | 0,03550 |
| **Schritt-3 (A dedup, C = neuer Checkpoint)** | `results/sdm_phase7_metrics_tessera_v1dedup.csv` | 0,03375 | **0,03419** |

Konfidenzintervalle (Zellebenen-Bootstrap, Schritt-3-Zeile, aus `results/sdm_phase7_metrics_tessera_v1dedup.csv`):
- `tessera/a`: KL=0,0337±0,0008, 95%-CI [0,02868; 0,04094]
- `tessera/c`: KL=0,0342±0,0001, 95%-CI [0,02861; 0,04032]

**Gepaarter Zell-Bootstrap C (neuer Checkpoint) − A** (Quelldatei:
`results/sdm_paired_bootstrap_tessera_v1dedup_c_minus_a.csv`, 417 gemeinsame Test-LSOAs):

| Metrik | Punktdiff (C−A) | 95%-CI | Anteil Resamples>0 | signifikant |
|---|---|---|---|---|
| kl | −0,00033 | [−0,00274; +0,00217] | 0,387 | nein |
| l1_sum | +0,00062 | [−0,00537; +0,00725] | 0,579 | nein |
| l1_mean | +0,00007 | [−0,00060; +0,00081] | 0,579 | nein |
| chebyshev | −0,00002 | [−0,00195; +0,00214] | 0,470 | nein |

Kein Unterschied zwischen Arm C (neuer Checkpoint) und Arm A ist statistisch signifikant.

---

## Referenzierte Dateien (alle mit MD5 in `results/checksums.txt`)

`retrieval_baselines_tessera_v1_dedup.csv`, `retrieval_baselines_tessera_v1.csv` (Alt, nicht in
`results/`), `landuse_probe_results_tessera_dedup_ckpt.csv`,
`landuse_probe_cell_bootstrap_tessera_dedup_ckpt.csv`,
`landuse_probe_paired_diff_cell_tessera_dedup_ckpt.csv`,
`landuse_probe_results_tessera_dedup.csv` (Alt, nicht in `results/`),
`class_retrieval_tessera_v1_dedup.csv`, `class_retrieval_tessera.csv` (Alt, nicht in `results/`),
`sdm_phase7_metrics.csv` (Original, Alt, nicht in `results/`), `sdm_phase7_metrics_dedup.csv`,
`sdm_phase7_metrics_tessera_v1dedup.csv`, `sdm_paired_bootstrap_tessera_v1dedup_c_minus_a.csv`.
