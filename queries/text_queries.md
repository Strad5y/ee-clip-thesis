# Query-Liste: Text-Query-Evaluation
Festgelegt am 2026-09-23, VOR Ansicht der Ergebnisse.
Zu committen vor dem ersten Lauf.

Die AETHER-Vergleichsqueries (green, airport, cbd) und die
Kontroll-Query (desert) sind bereits gelaufen und stehen in
INSTRUCTIONS_maps_aether.md. Diese Liste ergänzt sie.

---

## A. Grünflächen-Varianten
Zweck: Die AETHER-Query "Green Park and Natural Reserve." erzeugt das
stärkste Signal entlang der Themse statt über den Parks. Diese drei
Queries zerlegen den zusammengesetzten Wortlaut, um zu prüfen, welcher
Bestandteil das Muster treibt. Sie sind eine Diagnose eines bestehenden
Ergebnisses, keine eigenständige Fähigkeitsprüfung, und werden im Text
auch so eingeordnet.

### green_park
Query: "park"
Erwartung: Höchste Werte über den grossen Parks (Hyde Park, Richmond
Park, Hampstead Heath) und kleineren Grünflächen in der Stadt. Kein
Themse-Schwerpunkt.
OSM-Zuordnung: leisure=park, leisure=garden
Auswertung: quantitativ (AP, Prototyp) und visuell

### green_space
Query: "green space"
Erwartung: Breiter als "park", zusätzlich Wald, Grasland und Gebüsch.
Vergleichbar mit der Green-Cover-Referenz.
OSM-Zuordnung: leisure=park, leisure=garden, natural=wood,
natural=scrub, natural=grassland
Auswertung: quantitativ (AP gegen OSM-Klassen und zusätzlich gegen die
Green-Cover-Referenz) und visuell

### green_reserve
Query: "nature reserve"
Erwartung: Wenn der Themse-Schwerpunkt von diesem Bestandteil kommt,
sollte diese Query das Muster der AETHER-Query am stärksten
reproduzieren. Londons Naturschutzgebiete liegen überwiegend an
Fluss- und Feuchtgebieten.
OSM-Zuordnung: leisure=nature_reserve
Auswertung: quantitativ (AP, Prototyp) und visuell

Gemeinsame Auswertung der Gruppe: Rangkorrelation (Spearman) jeder
Variante mit der AETHER-Query "Green Park and Natural Reserve.", je
Backbone. Die Variante mit der höchsten Korrelation ist der
dominierende Bestandteil.

---

## B. Tätigkeits- und Atmosphären-Queries
Zweck: Open-Vocabulary-Prüfung. Keiner dieser Wortlaute kommt in den
Trainingstexten vor. Die Begriffe beschreiben Funktion und Nutzung
statt Aussehen und prüfen damit die zentrale These der Arbeit.

### qual_tourist
Query: "a place that is a tourist attraction"
Erwartung: Höchste Werte an bekannten Sehenswürdigkeiten und im
Zentrum (Westminster, South Bank, Tower, Museumsviertel South
Kensington). Einzelne Punkte in den Aussenbezirken möglich.
OSM-Zuordnung: tourism=attraction, tourism=museum, tourism=gallery,
tourism=viewpoint, historic=monument, historic=memorial,
historic=castle
Auswertung: quantitativ (AP, Prototyp) und visuell

### qual_busy
Query: "a place that is busy"
Erwartung: Zentrum, Hauptverkehrsachsen, grosse Bahnhöfe,
Einkaufsstrassen. Niedrige Werte in Wohngebieten am Stadtrand, auf
Grünflächen und Wasser.
OSM-Zuordnung: keine
Auswertung: nur visuell, gemeinsam mit qual_quiet

### qual_quiet
Query: "a place that is quiet"
Erwartung: Grünflächen, Friedhöfe, Wohngebiete am Stadtrand,
Wasserflächen. Niedrige Werte im Zentrum.
OSM-Zuordnung: keine
Auswertung: nur visuell, als Gegenstück zu qual_busy. Die beiden
Karten sollten sich deutlich unterscheiden. Tun sie es nicht, ist das
ein Befund über die Grenzen des Alignments. Zusätzlich: Spearman
zwischen beiden Karten, je Backbone. Ein stark negativer Wert wäre das
erwartete Ergebnis.

### qual_learn
Query: "a place to learn"
Erwartung: Bildungseinrichtungen (Schulen, Universitäten,
Bibliotheken). Museen und Zoos plausibel, da "learn" breiter ist als
"study". Verteilte Einzelorte, keine räumliche Konzentration.
OSM-Zuordnung: amenity=school, amenity=university, amenity=college,
amenity=library, amenity=kindergarten, tourism=museum
Auswertung: quantitativ (AP, Prototyp) und visuell

---

## Gemeinsame Festlegungen
- Kontroll-Query "desert" gilt für alle Queries dieser Liste. Für jede
  Query wird die Spearman-Korrelation mit der Kontrollkarte berichtet.
- Kommentiert werden die fünf höchsten Pixel mit mindestens 500 m
  Abstand zueinander, je Query und Backbone. Keine manuelle Auswahl.
- Skalengrenzen je Query: 1./99. Perzentil über beide Backbones,
  identisch für beide, wie bei den AETHER-Queries.
- Wo eine OSM-Zuordnung angegeben ist, gilt ein Pixel als relevant,
  wenn mindestens eine der genannten Klassen in seiner Klassenmenge
  liegt (gleiche Relevanzdefinition wie im Klassen-Retrieval).
- Kennzahlen werden getrennt für Trainings- und Test-Zellen berichtet,
  Karten weiterhin über ganz London.
- Die Prototyp-Baseline bildet den Mittelwert über die Trainingspixel
  aller zugeordneten Klassen gemeinsam.