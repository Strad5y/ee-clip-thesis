# SDM — Socioeconomic Distribution Mapping (London)

Neue Evaluationsaufgabe, analog zu AETHER (arXiv 2510.09894, Tabelle 1). Ziel:
pro LSOA eine normierte NS-SeC-Verteilung vorhersagen, aus gepoolten
EO-Embeddings, in den drei Armen A/B/C.

**Arbeitsweise:** Nach jeder Phase anhalten und berichten. Nicht eigenständig
weiterlaufen. Keine Pfade, Spaltennamen oder Konfigurationswerte annehmen —
was nicht verifiziert ist, wird berichtet, nicht geraten. Lange Downloads und
Läufe über `nohup`/`setsid`, kein Auto-Abbruch nach 2 h.

---

## Datenquellen — was genau heruntergeladen wird

| # | Was | Quelle | Datei / Produkt | Lizenz |
|---|-----|--------|-----------------|--------|
| 1 | NS-SeC-Zielvariable | Nomis Census 2021 Bulk<br>`https://www.nomisweb.co.uk/sources/census_2021_bulk` | `census2021-ts062.zip` → LSOA-CSV daraus | OGL v3.0 |
| 2 | LSOA-Grenzen 2021 | ONS Open Geography Portal<br>`https://geoportal.statistics.gov.uk/` | „Lower layer Super Output Areas (December 2021) Boundaries EW **BFC**" (full resolution, clipped to coastline), aktuelle Version | OGL v3.0, enthält OS- und ONS-IPR |
| 3 | LSOA → Local Authority | ONS Open Geography Portal | „LSOA (2011) to LSOA (2021) to Local Authority District (2022) Best Fit Lookup for EW", Felder `LSOA11CD, LSOA11NM, LSOA21CD, LSOA21NM, LAD22CD, LAD22NM` | OGL v3.0 |

Zu #2: **BFC**, nicht BGC. Die generalisierte Variante (20 m) verschiebt Grenzen
um bis zu zwei Pixelbreiten, was bei 10-m-Pixeln und flächengewichteter
Zuordnung relevant ist. Falls BFC nicht verfügbar: BFE (full extent, ungeclippt)
nehmen und berichten — dann liegen Themse-Flächen mit im Polygon.

Falls der Portal-Download hakt, sind die Boundaries auch über die
ArcGIS-REST-API abrufbar, Endpoint-Muster
`https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10/FeatureServer/0/query`
mit `outFields=*`, `f=geoJSON`, paginiert über `resultOffset`. Die
Versionsnummer im Servicenamen kann sich geändert haben — vor Verwendung
prüfen, nicht blind übernehmen.

Zu #3: Nur für den London-Filter nötig. Falls die TS062-Bulk-CSV bereits eine
LA-Spalte mitbringt, entfällt der Lookup — in Phase 1 prüfen.

**Alles unter OGL v3.0, kein Login, keine Registrierung, keine
Hochschullizenz.** Unterschied zu AETHER, die für London Ordnance Survey POI
und Verisk NLUD nutzen (Digimap, Educational Licence). Gehört in die Diskussion.

---

## Phase 0 — Repo-Zustand feststellen

Nichts verändern. Ermitteln und berichten:

1. Struktur von `data/`: Konvention für externe Rohdaten und verarbeitete
   Artefakte. Wo liegen Urban Atlas und die OSM-PBFs, nach welchem Schema?
2. Vollständiger Beispieleintrag aus `MANIFEST.md`.
3. `build_splits_pixels.py`: In welchem Format liegt das Split-Artefakt vor?
   Gibt es eine Zell-ID pro Pixel oder eine Zellgeometrie mit Partitionslabel?
   Lässt sich daraus eine Zellgeometrie rekonstruieren, ohne das Raster erneut
   zu durchlaufen?
4. `run_landuse_probe.py`: Wie sind die Arme A/B/C implementiert? Wo genau wird
   projiziert, wie ist der Probe-Kopf aufgebaut, wie läuft das Bootstrap?
5. Wie werden trainierte Projektionsköpfe gespeichert und geladen
   (`best.pt`-Struktur, Kopplung an die Config)?
6. `grep -ri "ns-sec\|nssec\|ts062\|lsoa\|census\|sdm" --include="*.py" --include="*.md"`

**Stopp. Bericht abwarten.**

---

## Phase 1 — Rohdaten beschaffen

1. Die drei Quellen herunterladen, ins Rohdatenverzeichnis aus Phase 0. Nicht
   ins Repo committen.
2. SHA-256 je Datei berechnen und notieren.
3. TS062-ZIP entpacken, Inhalt auflisten, LSOA-CSV identifizieren.
4. Für die TS062-LSOA-CSV **berichten, nicht interpretieren**: exakte
   Spaltennamen vollständig, Zeilenzahl, erste drei Zeilen, NaN pro Spalte,
   ob eine Local-Authority- oder Regionsspalte vorhanden ist.
5. Für die Boundaries: CRS, Feature-Zahl, Feldnamen, ob `LSOA21CD` enthalten ist.

**Stopp. Insbesondere die Spaltenliste abwarten — die Kategorienentscheidung in
Phase 2 wird nicht vorweggenommen.**

---

## Phase 2 — Zielvariable normieren

### 2a Kategorien festlegen

Im AETHER-Repo nachsehen: `https://github.com/inwind0212/AETHER`

Gesucht: wie viele NS-SeC-Kategorien in die Verteilung eingehen, und ob L15
(Vollzeitstudierende) und „Not classified" enthalten sind. Das Paper sagt nur
„normalized occupational distribution".

Keine Formalie: in studentendichten LSOAs dominiert L15 und überdeckt die
Berufsstruktur. **Falls sich das im Repo nicht eindeutig klären lässt, so
berichten und nichts annehmen.**

### 2b Normieren

- auf Greater London filtern (LA-Spalte oder Lookup aus Phase 1)
- gewählte Kategoriespalten auswählen
- pro Zeile durch die Zeilensumme der **gewählten** Spalten teilen, **nicht**
  durch die mitgelieferte Gesamtsumme — die enthält verworfene Kategorien
- prüfen: jede Zeile summiert sich auf 1,0 (Toleranz 1e-6)
- Zeilen mit Summe 0 vor der Division gesondert berichten, nicht stillschweigend
  als NaN durchreichen

Berichten: LSOA-Zahl nach London-Filter, Anzahl und Namen der Kategorien,
Zeilen mit Summe 0, je Kategorie Mittelwert/Median/Min/Max, häufigste und
seltenste drei Kategorien.

**Stopp.**

---

## Phase 3 — Geometrie und Split-Zuordnung

1. Boundaries auf Greater London clippen, nach **EPSG:32630** reprojizieren
   (in Phase 0 gegen das restliche Pipeline-CRS verifizieren).
2. Join TS062 ↔ Geometrie über `LSOA21CD`. **Niemals über den Namen.** Nicht
   gematchte Zeilen beidseitig zählen und berichten.
3. Für jedes LSOA den Flächenanteil in Train-, Val- und Testzellen des
   bestehenden 1-km-Rasters berechnen.
4. Jedes LSOA **vollständig** der Partition mit dem größten Flächenanteil
   zuweisen. Kein LSOA darf über Partitionen zerfallen.
5. Berichten:
   - LSOAs je Partition
   - wie viele überhaupt über mehr als eine Partition lagen
   - Verteilung des maximalen Flächenanteils (wie eindeutig war die Zuordnung?)
   - wie viele LSOAs mit maximalem Anteil unter 60 %
6. Zusätzlich **Variante B** erzeugen, in der alle mehrdeutigen LSOAs unterhalb
   eines zu berichtenden Schwellwerts verworfen werden. Beide Varianten als
   versionierte Artefakte speichern.

Hintergrund: Das 1-km-Raster wurde für Retrieval gebaut und kennt die
LSOA-Grenzen nicht. Ohne diese Zuordnung landen Pixel desselben LSOA in Train
und Test — dieselbe Leakage-Klasse wie bei den Urban-Atlas-Boundary-Pixeln.

**Stopp.**

---

## Phase 4 — Projiziertes Embedding-Feld erzeugen

Der Schritt, der bisher fehlt. Die Trainingspaare decken nur POI-Pixel ab; SDM
braucht **alle** Pixel im LSOA, auch Wohnbebauung ohne POI.

Je Backbone (AlphaEarth, TESSERA) und je Arm:

- **Arm A:** kein Projektionsfeld nötig, gepoolt wird direkt über das rohe
  Raster (64 bzw. 128 Kanäle).
- **Arm B:** Projektionskopf mit **untrainierten** Gewichten, gleiche
  Architektur und gleicher Seed-Umgang wie in `run_landuse_probe.py`.
- **Arm C:** trainierter Kopf aus dem jeweiligen `best.pt`.

Jeden Pixel einzeln durch den Kopf schicken, Ergebnis als 256-Kanal-Raster
persistieren. Das Feld wird mehrfach gebraucht (SDM, Land Use, klassenbasiertes
Retrieval) — einmal rechnen und cachen, nicht pro Auswertung neu.

**Kritisch, Reihenfolge:** erst projizieren, dann mitteln. Nicht umgekehrt. Der
Kopf enthält GELU und L2-Normalisierung, ist also nichtlinear; Mittelwert der
Ausgaben ≠ Ausgabe des Mittelwerts. Er wurde außerdem auf Einzelpixeln
trainiert — ein gemittelter Rohvektor liegt in einem Bereich des Raums, den er
nie gesehen hat. AETHER wendet den Projektor auf das gesamte Feld an und
aggregiert danach.

Berichten: Laufzeit, Speicherbedarf, Wertebereich je Arm, Stichprobencheck,
dass die projizierten Pixelvektoren L2-Norm 1 haben.

**Stopp.**

---

## Phase 5 — Pooling je LSOA

Mean-Pooling über alle Pixel im LSOA-Polygon, mit `exactextract`, konsistent
zum restlichen Vorgehen.

- Arm A poolt über das rohe Raster, Arm B und C über das jeweilige projizierte
  Feld aus Phase 4.
- Für alle drei Arme dieselbe Pixelmenge pro LSOA. Kein Filter auf POI-Pixel.
- LSOAs mit null überdeckten Pixeln identifizieren und berichten.

**Offene Entscheidung, einmal treffen und dokumentieren:** nach dem Mitteln
liegen die Regionsvektoren nicht mehr auf der Einheitskugel. Ob danach erneut
L2-normalisiert wird, sagt AETHER nicht. Beide Optionen vertretbar —
entscheidend ist, dass alle drei Arme gleich behandelt werden. Entscheidung
berichten, nicht implizit treffen.

**Stopp.**

---

## Phase 6 — Vorhersagekopf und Baselines

Zweischichtiger MLP, Embeddings eingefroren, nur der Kopf wird optimiert,
Softmax auf die C Kategorien.

- Loss: distributional cross-entropy
- Adam, Early Stopping auf Validation-Loss. **Hier ist Val-Loss korrekt** —
  anders als bei SigLIP gibt es keine lernbare Temperatur, die ihn verzerrt.
- Fünf Seeds, Mittelwert ± Standardabweichung.
- Alle drei Arme, beide Backbones, identisches Protokoll.

Zusätzlich **zwei Baselines**:

1. **Mittelwertsvorhersage:** durchschnittliche NS-SeC-Verteilung über alle
   Trainings-LSOAs, konstant für jedes Test-LSOA. Kein Modell, kein Training.
   Das ist die eigentliche Untergrenze bei Verteilungsaufgaben und fehlt in
   AETHERs Tabelle — ihr „Random" ist ein zufällig initialisiertes Embedding
   mit trainiertem Kopf, etwas anderes.
2. **Random-Embedding:** zufällig initialisierte Vektoren gleicher
   Dimensionalität plus trainierter Kopf, wie bei AETHER, zur Vergleichbarkeit.

**Stopp.**

---

## Phase 7 — Metriken und Konfidenzintervalle

Drei Metriken, alle niedriger-ist-besser, über das Testset:

- **KL-Divergenz.** Richtung explizit festlegen und dokumentieren —
  KL(wahr ‖ vorhergesagt) ≠ KL(vorhergesagt ‖ wahr). Umgang mit Nullen in der
  wahren Verteilung festlegen (Glättung, Epsilon, Ausschluss); ohne das wird KL
  an Nullstellen undefiniert, und Nullen kommen bei kleinen NS-SeC-Kategorien
  auf LSOA-Ebene vor. Erst im AETHER-Repo nachsehen, dann entscheiden.
- **L1-Distanz.** **Als Summe über die Komponenten implementieren, nicht als
  Mittelwert.** AETHER schreibt „mean absolute difference", aber ihre Zahlen
  widersprechen dem: L1 = 0,178 bei Chebyshev = 0,0538. Ein Mittelwert kann
  nicht über dem Maximum liegen. Als Summe über ~9 Kategorien passt es. Im Repo
  verifizieren und Ergebnis berichten.
- **Chebyshev-Distanz.** Maximale Absolutabweichung über die Komponenten.

**Bootstrap auf Kachelebene**, nicht auf LSOA- und nicht auf Pixelebene.
Benachbarte LSOAs sind räumlich autokorreliert; die 1-km-Zelle ist die
Split-Einheit und damit die korrekte Cluster-Einheit. Feinere Ebenen erzeugen
zu enge Intervalle.

### Sanity-Check vor allem anderen

**Arm A für AlphaEarth zuerst rechnen und gegen AETHER halten.** Für Arm A ist
das fast eine Replikation: rohe AE-Embeddings, LSOA-weise gemittelt, MLP-Kopf
auf NS-SeC. Es bleiben drei Unterschiede — geblockter statt zufälliger Split,
Kategorienauswahl, LSOA-Partitionszuordnung.

AETHERs Referenzwerte für London SDM, **skaliert wie in ihrer Tabelle**
(L1 × 10², KL und Chebyshev × 10³):

| Modell | KL | L1 | Chebyshev |
|---|---|---|---|
| Random | 75,7 | 30,0 | 91,4 |
| CaLLiPer | 34,1 | 19,3 | 58,0 |
| **AlphaEarth (= Arm A)** | **42,2** | **21,5** | **64,8** |
| AETHER | 30,1 | 17,8 | 53,8 |

Unskaliert ist AlphaEarths KL also **0,0422**, nicht 42,2.

Erwartung: Arm A sollte nahe 0,042 landen, tendenziell etwas schlechter wegen
des strengeren Splits. Ein Wert um 0,05 ist plausibel. Ein Wert um 0,15
bedeutet einen Pipeline-Fehler, mit hoher Wahrscheinlichkeit Normierung,
KL-Richtung oder L1-Definition.

**Diesen Check abschließen, bevor Arm C trainiert wird.**

**Stopp.**

---

## Was bewusst NICHT geändert wird

- kein Wechsel auf zufällige Splits, nur um mit AETHER vergleichbar zu sein.
  Die Vergleichbarkeit liegt in der Aufgabe (gleiche Zielvariable, gleiche
  Analyseeinheit, gleiche Metriken), nicht im Trainingsprotokoll.
- kein Wechsel der POI-Quelle, des Loss oder der Paargranularität
- keine Änderung an bestehenden Retrieval- oder Land-Use-Skripten
- keine Änderung am bestehenden 1-km-Split-Artefakt

---

## Provenienz

Je Quelle ein `MANIFEST.md`-Eintrag nach dem Schema aus Phase 0: Quelle,
Produktname und Version, Zugriffsdatum, URL, SHA-256, Lizenz (OGL v3.0),
Boundary-Jahrgang 2021.

Zusätzlich zu dokumentieren, weil es Entscheidungen sind und keine Fakten:
gewählte NS-SeC-Kategorien mit Begründung, KL-Richtung, Nullen-Behandlung,
L1-Definition, Re-Normalisierung nach dem Pooling ja/nein,
LSOA-Partitionsregel und Schwellwert für Variante B.

---

## Hinweise

- Boundary-Jahrgang ist **2021**. Falls später IMD dazukommt: IMD nutzt 2011er
  Grenzen, dafür wird der Lookup aus Quelle #3 gebraucht.
- Bei jeder Unsicherheit über Spaltenbedeutung, Metrikdefinition oder
  Kategorienzuschnitt: berichten statt annehmen. Eine falsch normierte
  Zielvariable oder eine falsch herum gerechnete KL-Divergenz ist im Ergebnis
  nicht von einem schlechten Modell zu unterscheiden.
