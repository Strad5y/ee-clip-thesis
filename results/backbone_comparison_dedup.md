# Backbone-Vergleich TESSERA-dedup vs. AlphaEarth-dedup (Schritt 4)

Stand: 2026-09-22 (Abschnitt 4/SDM aktualisiert nach der TESSERA-Normalisierungs-Nachziehung —
TESSERA Arm A/B nutzen jetzt dieselbe z-score-Statistik wie Arm C, Arm B neu ergänzt; siehe
Änderungsvermerke im Abschnitt selbst). Checkpoints: `checkpoints/tessera_v1_dedup/best.pt` (Epoche 48) und
`checkpoints/ae_v1_dedup/best.pt` (Epoche 32), jeweils trainiert auf der deduplizierten
Datenbasis des jeweiligen Backbones (Eindeutigkeits- + AOI-Polygon-Maske, siehe `MANIFEST.md`).
Beide Backbones sind jetzt nach derselben Prozedur dedupliziert — dies ist der erste direkte
TESSERA-vs-AlphaEarth-Backbone-Vergleich der Arbeit (vorher methodisch nicht zulässig, siehe
`MANIFEST.md`-Korrektur). Kein Vergleich gegen publizierte Paper (CLAUDE.md-Leitlinie) — nur
Ablation innerhalb der eigenen Pipeline, hier zusätzlich Backbone gegen Backbone.

**Geltungshinweise (für alle Tabellen unten):**
- **Gültiger Dedup-Effekt = Kontrolllauf → Dedup-Lauf** (`{backbone}_v1_predup_control` →
  `{backbone}_v1_dedup`, gleicher Seed, gleiche Config, siehe `MANIFEST.md`/Schritt-2-Berichte).
  Der alte, unseeded Checkpoint (`tessera_v1`/`ae_v1`) dient nur der Dokumentation, nicht als
  Vergleichsbasis für den Dedup-Effekt.
- Alt↔Neu vermengt bei Retrieval, Land-Use-Probe und Class-Retrieval immer zwei Effekte
  (Checkpoint-Wechsel UND Dedup der Auswertedaten). Nur bei SDM existiert eine isolierende
  Zwischenstufe.
- **Der Class-Retrieval-Rückgang von `ae_v1` (alt) zu `ae_v1_dedup` (neu) ist NICHT als
  Dedup-Effekt zu lesen** — siehe Abschnitt 3, Diagnose.
- Retrieval und Class-Retrieval haben in dieser Pipeline **keine Bootstrap-/CI-Maschinerie**
  (Punktschätzung über den vollen Test-Split). Für den Backbone-Vergleich gibt es dort deshalb
  keinen gepaarten Zellebenen-Bootstrap — nur bei Land-Use und SDM (Abschnitte 2 und 4).
- Keine Signifikanzangabe ohne ausgewiesenes Konfidenzintervall.

---

## 1. Retrieval (Test-Split, In-Batch-Diagnose)

Quellen: `results/retrieval_baselines_tessera_v1_dedup.csv`, `results/retrieval_baselines_ae_v1_dedup.csv`.
Punktschätzung, kein CI (kein Resampling implementiert).

| Bucket | Metrik | TESSERA-dedup | AE-dedup |
|---|---|---|---|
| all (n=728.865 / 728.588) | recall@1_diag | 0,1564 | 0,1942 |
| all | recall@10_diag | 0,5058 | 0,5465 |
| all | ceiling@10_diag | 0,9512 | 0,9511 |
| all | exploitation@10_diag | 0,5317 | 0,5746 |
| single (n=480.183 / 479.834) | recall@10_diag | 0,5310 | 0,5690 |
| single | exploitation@10_diag | 0,5706 | 0,6115 |
| multi (n=248.682 / 248.754) | recall@10_diag | 0,4570 | 0,5031 |
| multi | exploitation@10_diag | 0,4611 | 0,5078 |
| Zufalls-Baseline recall@10_diag | | 0,0391 | 0,0391 |

AE-dedup liegt in allen Buckets über TESSERA-dedup. Ohne CI ist das eine Punktaussage, keine
statistisch abgesicherte — siehe Geltungshinweis oben.

---

## 2. Land-Use-Probe

Quellen: `results/landuse_probe_results_{tessera,ae}_dedup_ckpt.csv` (Punktschätzung),
`results/landuse_probe_cell_bootstrap_{tessera,ae}_dedup_ckpt.csv` (Zellebenen-Bootstrap, 1000×,
167 Test-Zellen je Backbone), `results/landuse_probe_paired_diff_cell_{tessera,ae}_dedup_ckpt.csv`
(gepaarter Zellebenen-Diff trainiert−roh je Backbone), `data/processed/landuse_probe_majority_
baseline_ae_dedup_ckpt.csv` (Häufigkeitsniveau; für TESSERA-dedup nicht separat berechnet, nur
für AE auf explizite Nutzeranfrage in dieser Aufgabe).

**Punktschätzung (bestes val_macro_f1 je Konfiguration):**

| Konfiguration | C (TESSERA) | TESSERA test_oa | TESSERA macro_f1 | C (AE) | AE test_oa | AE macro_f1 |
|---|---|---|---|---|---|---|
| roh | 10,0 | 0,6042 | 0,2511 | 1,0 | 0,6352 | 0,2811 |
| zufällig projiziert | 10,0 | 0,5993 | 0,2397 | 1,0 | 0,6294 | 0,2624 |
| trainiert projiziert | 10,0 | 0,6158 | 0,2532 | 1,0 | 0,6433 | 0,2831 |
| Häufigkeitsniveau | | — | — | | 0,4565 | 0,0273 |

**Zellebenen-Bootstrap (1000×, 167 Test-Zellen je Backbone), 95-%-CI:**

| Konfiguration | TESSERA test_oa | 95%-CI | AE test_oa | 95%-CI |
|---|---|---|---|---|
| roh | 0,6035 | [0,5803; 0,6287] | 0,6347 | [0,6119; 0,6605] |
| zufällig projiziert | 0,5986 | [0,5749; 0,6249] | 0,6287 | [0,6052; 0,6553] |
| trainiert projiziert | 0,6151 | [0,5921; 0,6402] | 0,6427 | [0,6195; 0,6677] |

| Konfiguration | TESSERA macro_f1 | 95%-CI | AE macro_f1 | 95%-CI |
|---|---|---|---|---|
| roh | 0,2479 | [0,2226; 0,2723] | 0,2798 | [0,2542; 0,3111] |
| zufällig projiziert | 0,2371 | [0,2157; 0,2607] | 0,2625 | [0,2334; 0,3015] |
| trainiert projiziert | 0,2501 | [0,2294; 0,2681] | 0,2792 | [0,2559; 0,3032] |

**Gepaarter Zellebenen-Bootstrap je Backbone (trainiert − roh):**

| Backbone | Metrik | Diff | 95%-CI | signifikant |
|---|---|---|---|---|
| TESSERA | oa | +0,0116 | [0,0087; 0,0145] | **ja** |
| TESSERA | macro_f1 | +0,0023 | [−0,0105; +0,0121] | nein |
| AE | oa | +0,0080 | [0,0039; 0,0120] | **ja** |
| AE | macro_f1 | −0,0006 | [−0,0216; +0,0169] | nein |

Beide Backbones zeigen dasselbe Muster: der trainierte Kopf verbessert overall accuracy
signifikant gegenüber roh, macro-F1 nicht.

**Hinweis (2026-09-22, TESSERA-Normalisierungs-Nachziehung): dieser Abschnitt ist von der
Normalisierungs-Korrektur NICHT betroffen und wurde nicht neu gerechnet.** Der Land-Use-Wrapper
`run_landuse_probe_tessera_v1dedup.py` hat für TESSERA-dedup von Anfang an `configs/
tessera_v1_dedup.yaml` (die Dedup-Statistik) als Normalisierungsquelle für "roh"/"zufällig
projiziert" verwendet (`YAML_CFG_PATH`, siehe Skript) — anders als die SDM-Arm-A/B-Poolung, die
über `resolve_normalization(yaml_cfg, f"configs/{backbone}_v1.yaml")` bis zu diesem Nachtrag
die alte, undeduplizierte Statistik (`configs/tessera_v1.yaml` → `norm_stats_tessera_pixels.json`)
gelesen hat. Land-Use war also nie auf dem "alten Statistik"-Pfad — nichts zu korrigieren.

### Gepaarter Bootstrap auf Zellebene für die Backbone-Differenz (TESSERA − AE)

Datei: `diag_backbone_paired_bootstrap_landuse.py` → `results/diag_backbone_paired_bootstrap_landuse.csv`
(Diagnose, keine bestehende Ergebnisdatei überschrieben). Beide Backbones haben **exakt dieselben
167 Test-Zellen** (verifiziert: symmetrische Differenz 0). In jedem der 2000 Resamples wird
dieselbe gezogene Zellmenge auf beide Backbones angewendet (gepaart, nicht zwei unabhängige CIs
verglichen). Refit bei den bereits gewählten besten C-Werten (TESSERA C=10,0, AE C=1,0).

| Konfiguration | Metrik | TESSERA | AE | Diff (TESSERA − AE) | 95%-CI | signifikant |
|---|---|---|---|---|---|---|
| roh | oa | 0,6042 | 0,6352 | −0,0310 | [−0,0392; −0,0227] | **ja** |
| roh | macro_f1 | 0,2511 | 0,2811 | −0,0301 | [−0,0510; −0,0166] | **ja** |
| trainiert projiziert | oa | 0,6158 | 0,6433 | −0,0275 | [−0,0351; −0,0192] | **ja** |
| trainiert projiziert | macro_f1 | 0,2532 | 0,2831 | −0,0299 | [−0,0478; −0,0117] | **ja** |

**AE schlägt TESSERA bei der Land-Use-Klassifikation signifikant**, sowohl roh als auch trainiert
projiziert, sowohl overall accuracy als auch macro-F1. Der Effekt bleibt nach der Projektion
praktisch unverändert (roh −0,0310/−0,0301 vs. trainiert −0,0275/−0,0299) — die Projektionsköpfe
verschieben den Backbone-Unterschied nicht wesentlich.

---

## 3. Class-Retrieval

Quellen: `results/class_retrieval_tessera_v1_dedup.csv`, `results/class_retrieval_ae_v1_dedup.csv`
(Zeile `MACRO_MEAN`). Makro-Mittel über je 185/186 Query-Klassen mit ≥100 relevanten Test-Pixeln,
kein CI (keine Resampling-Prozedur implementiert — deshalb kein gepaarter Zellebenen-Bootstrap
für die Backbone-Differenz an dieser Stelle, siehe Geltungshinweis).

| Metrik | TESSERA-dedup (n_gallery=728.865, 186 Klassen) | AE-dedup (n_gallery=728.588, 185 Klassen) |
|---|---|---|
| random_baseline | 0,00758 | 0,00762 |
| p@10 | 0,0737 | 0,0649 |
| p@50 | 0,0749 | 0,0665 |
| p@100 | 0,0746 | 0,0656 |
| r_precision | 0,0494 | 0,0523 |
| average_precision | 0,0353 | 0,0408 |

**Duplikat-Beleg** (`check_class_retrieval_duplicates.py`/`check_class_retrieval_duplicates_ae.py`):
TESSERA-Galerie 0 verbleibende Dopplungen (Alt-Galerie 11,86 %), AE-Galerie 0 verbleibende
Dopplungen (Alt-Galerie 0,008 % — bei AE macht Dedup dort ohnehin kaum etwas, siehe Schritt 1).

**Hinweis zur Seed-Sensitivität des Class-Retrievals** (freigegebene Diagnose, 2026-09-21/22,
NICHT Teil der obigen Ergebnisdatei-Zahlen — `diag_class_retrieval_ae_control.py`, Ergebnisse
`results/diag_class_retrieval_ae_control_{summary,per_query,gallery_relevance}.csv`): Der
Makro-AP-Unterschied zwischen zwei AE-Checkpoints auf **identischen** Daten und Config
(`ae_v1` vs. `ae_v1_predup_control`, beide auf der alten, undeduplizierten Galerie) beträgt
**0,0041 (≈9 % relativ zur Makro-AP von `ae_v1`, 0,0456)**, bei stabiler Klassen-Rangfolge über
die 185 Klassen (Spearman ρ = 0,954 zwischen den AP-Werten beider Läufe). Der beobachtete
Rückgang `ae_v1` (alt) → `ae_v1_dedup` (neu) von −0,0048 Makro-AP zerfällt in Galerie-Effekt
(−0,0002 bis −0,0003), Checkpoint-Lauf-zu-Lauf-Effekt (−0,0041, s.o.) und reinen Dedup-Effekt
(gleicher Seed, `predup_control`/alt → `ae_v1_dedup`/neu: −0,0006, ≈1,5 %). Diese Metrik ist also
in dieser Pipeline stark seed-sensitiv und für Aussagen über einzelne Checkpoint-Vergleiche mit
Vorsicht zu lesen — ein TESSERA-Äquivalent dieser Diagnose wurde nicht gerechnet (nicht
angefordert), die Seed-Sensitivität ist aber ein Eigenschaft des Verfahrens (wenige Testpixel je
Klasse dominieren die Makro-AP), nicht spezifisch für AE.

---

## 4. SDM

Quellen: `results/sdm_phase7_metrics_tessera_dedupnorm.csv` (**aktualisiert 2026-09-22** — TESSERA
Arm A/B jetzt mit derselben z-score-Statistik wie Arm C, siehe MANIFEST.md/
`build_sdm_arm_ab_tessera_dedupnorm.py`; alte Datei `sdm_phase7_metrics_tessera_v1dedup.csv`
unverändert erhalten, hatte nur Arm A/C mit unterschiedlichen Statistiken), `results/
sdm_phase7_metrics_ae_v1dedup.csv` (Hauptlauf 9cat/A/pure, KL/L1/Chebyshev, Zellebenen-
Bootstrap-CI aus Seed 0), `results/sdm_paired_bootstrap_tessera_dedupnorm_c_minus_a.csv`,
`results/sdm_paired_bootstrap_ae_v1dedup_c_minus_a.csv` (gepaarter Zell-Bootstrap C−A je
Backbone). Die frühere Asymmetrie (TESSERA/B fehlte, weil in der TESSERA-Dedup-Aufgabe nur
Arm C neu gerechnet wurde) ist mit der TESSERA-Normalisierungs-Nachziehung behoben — jetzt
haben beide Backbones alle drei Arme unter konsistenter Normalisierung.

| Modell | KL | 95%-CI (Zelle) | L1 (Summe) | 95%-CI | L1 (Mittel) | Chebyshev | 95%-CI |
|---|---|---|---|---|---|---|---|
| Mittelwertsvorhersage | 0,0696 | [0,0608; 0,0788] | 0,2852 | [0,2689; 0,3024] | 0,0317 | 0,0864 | [0,0793; 0,0932] |
| TESSERA/A (roh) | 0,0338 | [0,0286; 0,0409] | 0,1873 | [0,1744; 0,2016] | 0,0208 | 0,0571 | [0,0524; 0,0631] |
| TESSERA/B (zufällig) | 0,0403 | [0,0337; 0,0482] | 0,2068 | [0,1929; 0,2238] | 0,0230 | 0,0629 | [0,0570; 0,0699] |
| TESSERA/C (trainiert, dedup-Ckpt) | 0,0342 | [0,0286; 0,0403] | 0,1902 | [0,1772; 0,2029] | 0,0211 | 0,0578 | [0,0524; 0,0631] |
| AE/A (roh) | 0,0503 | [0,0425; 0,0577] | 0,2366 | [0,2201; 0,2511] | 0,0263 | 0,0710 | [0,0646; 0,0768] |
| AE/B (zufällig) | 0,0544 | [0,0468; 0,0618] | 0,2480 | [0,2318; 0,2633] | 0,0276 | 0,0748 | [0,0684; 0,0806] |
| AE/C (trainiert, dedup-Ckpt) | 0,0426 | [0,0362; 0,0494] | 0,2154 | [0,2011; 0,2301] | 0,0239 | 0,0643 | [0,0586; 0,0702] |

Die Normalisierungs-Korrektur ändert die TESSERA-Werte nur minimal (Δ KL Arm A: +0,000005,
Arm B war vorher unter der Dedup-Statistik gar nicht berechnet; gegen die dedup-maskierte,
aber alt-normierte Referenz `sdm_phase7_metrics_dedup.csv` — Arm A KL 0,033749→0,033754, Arm B
KL 0,040406→0,040264 — Delta jeweils zwei bis drei Größenordnungen kleiner als die CI-Breite).

**Gepaarter Zell-Bootstrap C − A je Backbone:**

| Backbone | Metrik | Diff (C−A) | 95%-CI | signifikant |
|---|---|---|---|---|
| TESSERA | kl | −0,00020 | [−0,00315; +0,00271] | nein |
| TESSERA | l1_sum | +0,00214 | [−0,00440; +0,00953] | nein |
| TESSERA | l1_mean | +0,00024 | [−0,00049; +0,00106] | nein |
| TESSERA | chebyshev | −0,00009 | [−0,00228; +0,00222] | nein |
| AE | kl | −0,00726 | [−0,01019; −0,00459] | **ja** |
| AE | l1_sum | −0,02012 | [−0,02740; −0,01370] | **ja** |
| AE | l1_mean | −0,00224 | [−0,00304; −0,00152] | **ja** |
| AE | chebyshev | −0,00635 | [−0,00882; −0,00412] | **ja** |

Bei TESSERA verbessert der trainierte Kopf die SDM-Vorhersage gegenüber dem rohen Backbone
**nicht** signifikant (alle vier Metriken, unverändert gegenüber der alten Normalisierung — die
CIs überlappen praktisch vollständig mit den vorherigen). Bei AE verbessert der trainierte Kopf
signifikant, in derselben Richtung, für alle vier Metriken.

### Gepaarter Bootstrap auf Zellebene für die Backbone-Differenz (TESSERA − AE)

Datei: `diag_backbone_paired_bootstrap_sdm.py` → `results/diag_backbone_paired_bootstrap_sdm.csv`
(Diagnose, **aktualisiert 2026-09-22**: TESSERA/A zeigt jetzt auf die dedupnorm-Poolung, TESSERA/B
neu ergänzt — beide Backbones haben jetzt Arm A/B/C im Vergleich). Beide Backbones haben **exakt
dieselben 417 Test-LSOAs** (symmetrische Differenz 0, in allen drei Armen geprüft). Zellzuordnung
über `bsp7.LSOA_TO_CELL` (Mehrheitsregel, wie im bestehenden Arm-A-vs-C-Vergleich), 1000
Resamples, dieselbe gezogene Zellmenge für beide Backbones je Resample, Seed 0 (Referenz-Seed,
wie überall sonst in Phase 7 — Punktwerte hier sind deshalb Einzelseed-Werte, nicht der
5-Seed-Mittelwert aus der Tabelle oben, wie schon in der vorherigen Fassung dieses Abschnitts).

| Arm | Metrik | TESSERA | AE | Diff (TESSERA − AE) | 95%-CI | signifikant |
|---|---|---|---|---|---|---|
| A (roh) | kl | 0,0344 | 0,0497 | −0,01532 | [−0,02057; −0,01021] | **ja** |
| A (roh) | l1_sum | 0,1880 | 0,2353 | −0,04726 | [−0,05994; −0,03492] | **ja** |
| A (roh) | l1_mean | 0,0209 | 0,0261 | −0,00525 | [−0,00666; −0,00388] | **ja** |
| A (roh) | chebyshev | 0,0578 | 0,0707 | −0,01292 | [−0,01719; −0,00856] | **ja** |
| B (zufällig) | kl | 0,0407 | 0,0539 | −0,01326 | [−0,01731; −0,00925] | **ja** |
| B (zufällig) | l1_sum | 0,2079 | 0,2470 | −0,03904 | [−0,04981; −0,02777] | **ja** |
| B (zufällig) | l1_mean | 0,0231 | 0,0274 | −0,00434 | [−0,00553; −0,00309] | **ja** |
| B (zufällig) | chebyshev | 0,0631 | 0,0744 | −0,01126 | [−0,01459; −0,00768] | **ja** |
| C (trainiert) | kl | 0,0342 | 0,0424 | −0,00826 | [−0,01131; −0,00525] | **ja** |
| C (trainiert) | l1_sum | 0,1901 | 0,2151 | −0,02500 | [−0,03319; −0,01688] | **ja** |
| C (trainiert) | l1_mean | 0,0211 | 0,0239 | −0,00278 | [−0,00369; −0,00188] | **ja** |
| C (trainiert) | chebyshev | 0,0577 | 0,0643 | −0,00666 | [−0,00957; −0,00372] | **ja** |

**TESSERA schlägt AE bei der SDM-Aufgabe signifikant** (niedrigeres KL/L1/Chebyshev = näher an
der wahren NS-SeC-Verteilung), jetzt in **allen drei** Armen und allen vier Metriken (Arm B war
vor der Normalisierungs-Nachziehung noch nicht Teil dieses Vergleichs). **Das ist das Gegenteil
der Land-Use-Richtung** (dort schlägt AE TESSERA) — die beiden EE-Backbones sind für die beiden
Downstream-Aufgaben unterschiedlich gut geeignet, kein Backbone dominiert durchgehend. Die
Normalisierungs-Korrektur ändert an dieser Aussage nichts (Arm A vorher −0,01519, jetzt −0,01532
KL-Differenz, praktisch identisch).

---

## 5. Hauptbefunde alter vs. neuer AE-Checkpoint — Konsistenzprüfung

Beide unabhängigen "Alignment"-Vergleiche (SDM Arm C−A, Land-Use trainiert−roh) zeigen mit dem
alten (`ae_v1`, unseeded) und dem neuen Checkpoint (`ae_v1_dedup`, Seed 0, deduplizierte Daten)
**dasselbe qualitative Muster** — der Dedup-Schritt und der Checkpoint-Wechsel verändern also
nicht, welche Aussage die Arbeit über AlphaEarth trifft, nur die Punktschätzungen leicht.

| Vergleich | Metrik | Alter Checkpoint (`ae_v1`) | 95%-CI | Neuer Checkpoint (`ae_v1_dedup`) | 95%-CI |
|---|---|---|---|---|---|
| SDM C−A | kl | −0,00641 | [−0,00983; −0,00361] | −0,00726 | [−0,01019; −0,00459] |
| SDM C−A | l1_sum | −0,01859 | [−0,02666; −0,01182] | −0,02012 | [−0,02740; −0,01370] |
| SDM C−A | l1_mean | −0,00207 | [−0,00296; −0,00131] | −0,00224 | [−0,00304; −0,00152] |
| SDM C−A | chebyshev | −0,00578 | [−0,00843; −0,00348] | −0,00635 | [−0,00882; −0,00412] |
| Land-Use trainiert−roh | oa | +0,0075 | [0,0041; 0,0110] | +0,0080 | [0,0039; 0,0120] |
| Land-Use trainiert−roh | macro_f1 | −0,0025 | [−0,0268; +0,0171] | −0,0006 | [−0,0216; +0,0169] |

Beide Male: SDM C−A signifikant negativ (Verbesserung, niedrigeres KL ist besser) in allen vier
Metriken; Land-Use oa signifikant positiv, macro_f1 nicht signifikant. Quellen: SDM alt
`results/diag_sdm_paired_bootstrap_arm_diff.csv` (Zeile `backbone=ae`), SDM neu `results/
sdm_paired_bootstrap_ae_v1dedup_c_minus_a.csv`; Land-Use alt `data/processed/landuse_probe_
paired_diff_cell_bootstrap.csv` (Zeile `backbone=ae`, 14.9., 5000×, MD5 in `results/checksums.txt`),
Land-Use neu `results/landuse_probe_paired_diff_cell_ae_dedup_ckpt.csv`.

---

## Offene Punkte

- **main.tex:** auf dieser Maschine nicht vorhanden (bereits im Schritt-4-Bericht der
  TESSERA-Dedup-Aufgabe vermerkt) — die aktualisierte Liste der zu ersetzenden Stellen (jetzt für
  beide Backbones) kann erst erstellt werden, wenn die Datei bereitgestellt wird.
- **Retrieval/Class-Retrieval ohne CI:** für einen statistisch abgesicherten Backbone-Vergleich
  bei diesen beiden Aufgaben müsste erst eine Bootstrap-Prozedur für sie implementiert werden
  (nicht Teil dieser Aufgabe).
- ~~TESSERA/Arm B fehlt im SDM-Vergleich~~ **erledigt 2026-09-22** — Arm B mit der Dedup-Statistik
  neu gerechnet und in den Vergleich aufgenommen (s.o.).
- ~~TESSERA-Normalisierung (Arm A/B mit alten Statistiken)~~ **erledigt 2026-09-22** — Arm A/B
  neu gepoolt mit derselben Statistik wie Arm C (`sdm_pooled_tessera_arm{a,b}_dedupnorm.parquet`,
  `build_sdm_arm_ab_tessera_dedupnorm.py`). Effekt auf alle SDM-Zahlen und die Backbone-Differenz
  vernachlässigbar (s.o.) — die Kernaussagen der Arbeit ändern sich dadurch nicht.
