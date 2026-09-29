# REPORT — Recall@Top-k% nach AETHER (Nachtlauf 2026-09-30)

Branch `recall-topk` (nicht nach main gemergt). Checkpoints: `ae_v1_dedup` (Epoche 32), `tessera_v1_dedup`
(Epoche 48). Kein Training, keine Hyperparameter-Suche, keine Ablation. Vollständiges Protokoll: `RUN_LOG.md`.

## 1. Was gelaufen ist

| Phase | Inhalt | Status | Commit |
|---|---|---|---|
| 0 | Bestandsaufnahme: Dedup-Arm-C-Felder + Text-Query-Masken wiederverwendet (keine Neuprojektion); kein Prototyp-Code vorhanden → neu implementiert | ok | – |
| 1 | Klassenzuordnung `queries/aether_categories.yaml` (vor jeder Auszählung) + `feasibility.md` | ok | 87dc23a |
| 2 | `eval/recall_topk.py` + synthetischer Selbsttest (Zufall ≈ k, Orakel = min(1, K/\|GT\|)): PASSED | ok | fd7fd83 |
| 3 | AlphaEarth, dann TESSERA; Gate `verify_provenance.py` vor jedem Lauf ok; Preflight-Reprojektion (Kosinus ≥ 0,9999) ok; Selbsttest auf Echtdaten PASS; CSVs, Tabelle, 10 Karten | ok | bbdd384 |
| 4 | Dieser Bericht | ok | (dieser Commit) |

Keine Kategorie und kein Backbone ist „failed“. Zwei Kategorien sind `no_gt` (siehe 3.).

## 2. Probleme und Entscheidungen

| Problem | Entscheidung | Verworfene Alternative | Auswirkung auf Zahlen |
|---|---|---|---|
| Kein `eval/`-Verzeichnis im Repo (Eval-Code sonst unter `src/eval/`) | `eval/recall_topk.py` wie vorgegeben angelegt | `src/eval/…` (nicht erlaubter Schreibpfad) | keine |
| `verify_provenance.py` prüft die Dedup-Tabellen nicht | Gate trotzdem vor jedem Lauf; zusätzlich harte Prüfung im Skript: rekonstruierte Zelle → Split muss die `split`-Spalte jedes GT-Pixels exakt reproduzieren (bestanden) | `verify_provenance.py` ändern (nicht erlaubt) | keine |
| Prototyp-Code fehlt (Text-Query Phase 7 gesperrt) | Neuimplementierung nach dieser Instruktion (Mittel der rohen Train-GT-Pixel) | Hälften-Split aus dem Phase-7-Vorschlag | definiert Arm P |
| GT-Quelle | GT = Pixel aus `pixels_{bb}_dedup.parquet`, deren `poi_ids` einen POI mit Tag aus `osm_classes` enthalten (exakt die Paarbildungsregel, kein Puffer; POIs ohne Text, „Fall D“, sind damit wie im Training nicht enthalten) | Neurasterung per exactextract bzw. `gt.py`-Weg mit 25-m-Puffer | definiert GT; 0 GT-Pixel außerhalb des Kandidatenraums |
| `n_pois` mehrdeutig | CSV: POIs mit ≥ 1 GT-Pixel; `feasibility.md` zeigt zusätzlich die Tabellenzahl | nur Tabellenzahl | nur Beschriftung |
| Split `val` | nur in „all“ enthalten | eigener Block | keine auf test/train |
| Bootstrap-Einheit | Resampling über Zellen mit ≥ 1 GT-Pixel der Kategorie im Split, B = 1000, Seed 0, gleiche Gewichte für T, P und alle k (gepaart) | alle Zellen des Splits inkl. GT-freier Zellen | beeinflusst nur die CIs |
| GT-Darstellung auf 50-m-Kartengitter | `Resampling.max` (Zelle grün bei ≥ 1 GT-Pixel), nur Darstellung | nearest (Punkt-POIs würden verschwinden) | keine (Kennzahlen auf nativem Raster) |
| Test-GT von art_antiques konzentriert: 73 % (AE) / 74 % (TESSERA) in einer 1-km-Zelle, Top-3-Zellen 87 % / 89 %, 16 Testzellen | Status bleibt nach Regel `ok`, Befund markiert, nichts entfernt | Umdeklaration auf `thin` (nicht regelgedeckt) | Test-Recall dieser Kategorie spiegelt überwiegend einen Ort; CIs breit und schief |

Weitere Einträge: keine Permission-Fehler, keine fehlenden Pakete, keine Installationen, freier Speicher 2,3 TB.

## 3. Zuordnung

| Kategorie | osm_classes (GT) | uncertain_classes (nicht in der GT) | Status |
|---|---|---|---|
| Art and Antiques | shop=art, shop=antiques | tourism=gallery, shop=collector, shop=frame, shop=craft, shop=pottery | ok (siehe Konzentrationsbefund) |
| Religious Organisations | amenity=place_of_worship, amenity=monastery, office=religion | building=church/chapel/cathedral/mosque/synagogue/temple/shrine/religious/kingdom_hall/monastery/presbytery, historic=church/monastery/mosque/temple, shop=religion | ok |
| Medical Equipment, Supplies and Pharmaceuticals | shop=medical_supply, amenity=pharmacy, healthcare=pharmacy | shop=chemist, shop=hearing_aids, shop=optician, shop=nutrition_supplements, shop=herbalist | ok |
| Sports Clubs and Associations | — (club=* nicht in der Klassentabelle) | leisure=sports_centre/pitch/stadium/sports_hall/golf_course, office=association, sport=* | **no_gt** |
| Import and Export Services | — | office=logistics, shop=wholesale, office=courier | **no_gt** |

Zu prüfen (Jakob): die unsicheren Klassen, v. a. die building=*-Bauformen bei Religious Organisations und
tourism=gallery bei Art and Antiques. Eine Änderung verschiebt die GT und erfordert einen Neulauf von Phase 3.

## 4. Ergebnisse (Test-Split, Recall in %, 95-%-Zellebenen-Bootstrap-CI)

Kandidatenraum N = 15.954.033 (AE) / 15.954.158 (TESSERA) Pixel. Zufall = K/N.

**AlphaEarth**

| k | Arm | Art and Antiques (351 px / 16 Zellen) | Religious Org. (3269 / 105) | Medical (218 / 63) | Sports Clubs | Import/Export |
|---|---|---|---|---|---|---|
| 1 % | Zufall | 1,0 | 1,0 | 1,0 | 1,0 | 1,0 |
| 1 % | Prototyp | 10,5 [0,0–48,2] | 2,0 [1,1–3,1] | 8,7 [3,0–15,4] | no_gt | no_gt |
| 1 % | Trainiert | 21,4 [10,1–37,0] | 7,4 [3,4–12,2] | 8,3 [4,3–13,7] | no_gt | no_gt |
| 5 % | Zufall | 5,0 | 5,0 | 5,0 | 5,0 | 5,0 |
| 5 % | Prototyp | 48,7 [30,0–74,1] | 9,7 [7,0–12,7] | 34,4 [23,2–45,1] | no_gt | no_gt |
| 5 % | Trainiert | 80,9 [44,9–87,4] | 24,0 [15,5–34,3] | 39,0 [25,1–53,7] | no_gt | no_gt |
| 10 % | Zufall | 10,0 | 10,0 | 10,0 | 10,0 | 10,0 |
| 10 % | Prototyp | 69,8 [61,0–89,6] | 20,0 [15,9–24,6] | 51,4 [37,0–65,1] | no_gt | no_gt |
| 10 % | Trainiert | 94,3 [56,7–98,9] | 36,3 [27,4–45,7] | 62,8 [48,9–77,7] | no_gt | no_gt |

**TESSERA**

| k | Arm | Art and Antiques (358 / 16) | Religious Org. (3210 / 105) | Medical (221 / 62) | Sports Clubs | Import/Export |
|---|---|---|---|---|---|---|
| 1 % | Zufall | 1,0 | 1,0 | 1,0 | 1,0 | 1,0 |
| 1 % | Prototyp | 60,1 [8,5–70,5] | 2,6 [1,7–3,5] | 14,0 [5,7–22,2] | no_gt | no_gt |
| 1 % | Trainiert | 44,4 [3,8–53,0] | 7,0 [4,8–9,4] | 4,5 [1,5–8,3] | no_gt | no_gt |
| 5 % | Zufall | 5,0 | 5,0 | 5,0 | 5,0 | 5,0 |
| 5 % | Prototyp | 81,0 [23,6–92,2] | 13,5 [11,1–15,9] | 41,2 [31,0–49,6] | no_gt | no_gt |
| 5 % | Trainiert | 81,6 [30,9–91,7] | 23,0 [18,0–28,4] | 26,2 [17,8–35,8] | no_gt | no_gt |
| 10 % | Zufall | 10,0 | 10,0 | 10,0 | 10,0 | 10,0 |
| 10 % | Prototyp | 88,8 [48,6–96,6] | 26,2 [22,1–30,2] | 68,8 [58,4–77,7] | no_gt | no_gt |
| 10 % | Trainiert | 92,7 [67,6–97,7] | 35,7 [29,8–41,8] | 58,4 [49,0–67,3] | no_gt | no_gt |

**Gepaarte Differenz T − P (Test, Prozentpunkte, 95-%-CI)**

| Backbone | Kategorie | k = 1 % | k = 5 % | k = 10 % |
|---|---|---|---|---|
| AE | Art and Antiques | +10,8 [−36,2; +34,4] | +32,2 [−20,9; +44,0] | +24,5 [−16,9; +32,1] |
| AE | Religious Org. | +5,4 [+1,0; +10,5] | +14,3 [+5,0; +25,2] | +16,3 [+6,2; +27,1] |
| AE | Medical | −0,5 [−8,8; +7,6] | +4,6 [−11,4; +22,2] | +11,5 [−4,9; +28,8] |
| TESSERA | Art and Antiques | −15,6 [−18,9; 0,0] | +0,6 [−1,2; +12,5] | +3,9 [+0,3; +23,2] |
| TESSERA | Religious Org. | +4,4 [+2,0; +6,9] | +9,5 [+4,5; +14,5] | +9,5 [+4,1; +14,5] |
| TESSERA | Medical | −9,5 [−19,3; +0,5] | −14,9 [−27,5; −0,9] | −10,4 [−20,7; 0,0] |

Train- und all-Blöcke: `table_aether_style.md`. Der Prototyp nutzt die Train-GT-Pixel als Stützpixel, seine
train- und all-Werte sind daher nicht unabhängig. Karten: `maps/recall_topk_{backbone}_{category}.png`.
Die Zahlen sind nicht mit AETHER Tab. 3 vergleichbar (andere Region, GT-Quelle, Kandidatenraum und Encoder).

## 5. Neustart von Phase 3 (z. B. nach Änderung der YAML)

```bash
cd ~/PycharmProjects/ee-clip-thesis && git checkout recall-topk
python -W ignore::DeprecationWarning eval/recall_topk.py feasibility          # optional: Report neu
python -W ignore::DeprecationWarning eval/recall_topk.py run --backbone ae       > outputs/recall_topk/run_ae.log 2>&1
python -W ignore::DeprecationWarning eval/recall_topk.py run --backbone tessera  > outputs/recall_topk/run_tessera.log 2>&1
python -W ignore::DeprecationWarning eval/recall_topk.py finalize
```

Laufzeit etwa 3–4 min je Backbone plus etwa 20 s für `finalize` (GPU nur für den Query-Encoder).

Dabei überschrieben (alle unter `outputs/recall_topk/`):
- `feasibility.md`, `feasibility.csv` (nur bei `feasibility`)
- `recall_topk_{ae,tessera}.csv`, `diffs_recall_topk_{ae,tessera}.csv`, `run_meta_{ae,tessera}.json`,
  `selftest_real_{ae,tessera}.txt`, `run_{ae,tessera}.log`
- `fields/{ae,tessera}/<category>/*_simT.tif`, `*_gt.tif`, `fields/scoresT_{ae,tessera}.npy` (nicht in Git).
  Wird eine Kategorie-ID umbenannt, bleiben die alten Unterordner liegen.
- `recall_topk.csv`, `diffs_recall_topk.csv`, `table_aether_style.md`, `map_scale_limits.csv`, `maps/*.png` (`finalize`)

Nicht überschrieben: `RUN_LOG.md`, `REPORT.md`, `selftest_synthetic.txt` (nur durch `eval/recall_topk.py selftest`),
sämtliche Eingabedaten, Felder und Masken außerhalb von `outputs/recall_topk/`.
