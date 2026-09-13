# Land-Use-Probe: Bootstrap auf Kachelebene

Der Test-Bootstrap in `run_landuse_probe.py` resampelt bisher einzelne
Testzeilen (Pixel). Benachbarte Pixel sind räumlich autokorreliert, deshalb
unterschätzt das die Unsicherheit systematisch — die Konfidenzintervalle sind
zu eng. Die korrekte Cluster-Einheit ist dieselbe wie die Split-Einheit: die
1-km-Zelle.

Der Effekt ist in SDM bereits quantifiziert: Kachel-CIs sind dort je nach
Metrik 1,29× bis 1,39× breiter als LSOA-CIs.

**Arbeitsweise:** Nach jeder Phase anhalten und berichten. Nichts umtrainieren
— das ist eine reine Nachbereitung der vorhandenen Testvorhersagen. Keine
Pfade oder Werte annehmen, die nicht verifiziert sind.

---

## Phase 1 — Bestandsaufnahme

Nichts verändern. Ermitteln und berichten:

1. In `run_landuse_probe.py`: wo genau wird der Test-Bootstrap gezogen?
   Codestelle, Zahl der Ziehungen, was resampelt wird, wie der gepaarte
   Test (trainiert − roh) die Indizes teilt.
2. Werden die Testvorhersagen persistiert, und mit welchen Spalten? Konkret:
   liegt pro Testzeile eine Zell-ID (`cell_row`, `cell_col`) vor, oder muss
   sie erst hergestellt werden?
3. `diag_landuse_distance_accuracy.py` hat `pred_test` mit Pixelkoordinaten
   verknüpft (aus den Kachel-Affintransformationen, EPSG:32630). Lässt sich
   dieser Mechanismus wiederverwenden, oder gibt es einen direkteren Weg —
   etwa weil die Pixel in `pixels_{ae,tessera}.parquet` bereits eine
   Zellzuordnung tragen?
4. Wie viele eindeutige Zellen enthält das Land-Use-Testset? Zum Vergleich:
   die 182 Testzellen aus `splits_london.csv` sind die Obergrenze.
5. Wie ist der zweistufige Aufbau dokumentiert — Trainings-Bootstrap
   (5 Seeds, Refit) für die Verfahrensstreuung, Test-Bootstrap (1000×, kein
   Refit) für die Präzision? Nur der zweite ist hier betroffen; bestätigen,
   dass der erste unberührt bleibt.

**Stopp. Bericht abwarten.**

---

## Phase 2 — Umstellung

Der Test-Bootstrap zieht künftig Zellen statt Zeilen. Eine gezogene Zelle
bringt alle ihre Testpixel mit; bei Mehrfachziehung mehrfach. Analog zum
Vorgehen in `build_sdm_phase7_metrics.py`.

Anforderungen:

- Der **gepaarte** Bootstrap für die Differenz trainiert − roh muss weiterhin
  dieselben Resample-Indizes je Backbone verwenden — jetzt auf Zellebene.
  Sonst ist die Differenz nicht mehr gepaart und das Intervall falsch.
- Der Trainings-Bootstrap (5 Seeds mit Refit) bleibt **unverändert**. Er misst
  Verfahrensstreuung, nicht Präzision, und ist von der Clusterung nicht
  berührt.
- Zahl der Ziehungen wie bisher (1000, falls Phase 1 das bestätigt).
- Die alte zeilenweise Variante **nicht löschen**, sondern als Option
  behalten. Beide Ergebnisse werden gebraucht (siehe Phase 3).

Berichten: wie die Zellzuordnung hergestellt wurde, ob Testpixel ohne
Zellzuordnung auftreten und wie viele.

**Stopp.**

---

## Phase 3 — Beide Varianten rechnen und gegenüberstellen

Für beide Backbones und alle drei Arme, sowie für die gepaarte Differenz
trainiert − roh:

| | Punktschätzung | CI (Zeilen-Bootstrap) | CI (Kachel-Bootstrap) | Breitenfaktor |

Metriken: Overall Accuracy und Macro-F1.

Die Punktschätzungen müssen zwischen beiden Varianten **identisch** sein —
der Bootstrap ändert nur die Intervalle, nicht die Schätzung. Falls sie
abweichen, ist etwas falsch; das dann melden statt wegzurunden.

Ausdrücklich berichten:

- Schließt das Kachel-CI der Differenz trainiert − roh die Null noch aus?
  Für AE lag die Differenz bei +0,0075 mit Zeilen-CI [0,0066; 0,0084], für
  TESSERA bei +0,0096 mit [0,0087; 0,0105]. Bei einem Breitenfaktor um 1,3
  sollte die Null draußen bleiben — aber das ist zu rechnen, nicht zu
  schätzen.
  \NUM{Diese Werte stammen aus Notizen. Erst gegen den tatsächlichen
  Probe-Output prüfen, dann verwenden.}
- Wie der Breitenfaktor sich zu den 1,29–1,39 aus SDM verhält.
- Ob sich die Rangfolge der Arme oder Backbones durch die Umstellung
  irgendwo ändert (sollte nicht, da die Punktschätzungen gleich bleiben —
  aber Signifikanzaussagen können kippen).

**Stopp.**

---

## Phase 4 — Provenienz und Commit

- `MANIFEST.md`-Eintrag: geänderte Bootstrap-Ebene, Begründung (räumliche
  Autokorrelation, Cluster-Einheit = Split-Einheit), Breitenfaktor, Datum.
- Die alten Ergebnisse mit Zeilen-Bootstrap **nicht überschreiben**, sondern
  als Vergleichsartefakt behalten. Sie sind der Beleg dafür, dass die
  Umstellung nötig war.
- In logischen Einheiten committen.

---

## Was in diesem Vorhaben NICHT passiert

- kein Retraining der Projektionsköpfe
- keine Änderung am Probe-Fit, an der Normalisierung oder am C-Grid
- keine Änderung am Trainings-Bootstrap
- keine Änderung an den Splits
- keine Änderung an den SDM-Artefakten

---

## Hinweis

Falls die Zellzuordnung für die Testpixel nur über den Umweg der
Pixelkoordinaten herstellbar ist (wie in der Distanzdiagnose), lohnt es sich,
sie **einmal** zu berechnen und zu persistieren — sie wird für jede weitere
räumlich geclusterte Auswertung wieder gebraucht.
