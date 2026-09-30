# Machbarkeit Recall@Top-k% — AETHER-Kategorien

Erzeugt von `eval/recall_topk.py feasibility` am 2026-09-30 21:26. Zuordnung: `queries/aether_categories.yaml` (vor dieser Auszählung festgelegt).

Status: `ok` / `thin` (< 50 Test-GT-Pixel oder < 3 Testzellen) / `no_gt` (keine GT-Pixel). GT-Pixel = Pixel der Dedup-Trainingspixeltabelle, die ein POI einer zugeordneten Klasse schneidet (kein Puffer), eingeschränkt auf den Kandidatenraum (AOI-Polygon + Eindeutigkeitsmaske + gültiges Arm-C-Feld). Zellen = 1-km-Zellen aus `data/splits/splits_london.csv`.

## Klassen

| Kategorie | Klasse | POIs in London (Tabelle) |
|---|---|---|
| Art and Antiques | `shop=art` | 330 |
| Art and Antiques | `shop=antiques` | 249 |
| Religious Organisations | `office=religion` | 25 |
| Medical Equipment, Supplies and Pharmaceuticals | `shop=medical_supply` | 13 |
| Sports Clubs and Associations | `club=sport` (außerhalb der Klassentabelle, direkt aus dem PBF) | 328 |
| Import and Export Services | `office=logistics` | 17 |

Unsichere Klassen (nicht in der GT, nur zur Prüfung): siehe YAML `uncertain_classes`.

Extra-Tag `club=sport`: 328 Objekte, Geometrietypen {'Polygon': 236, 'Point': 92}, 0 per make_valid repariert. Gerastert mit spatial_join.join_pois_to_tiles (Punkte → Pixel, Polygone → exactextract Coverage > 0, Linien übersprungen).

## ae (N = 15.954.033 Kandidatenpixel)

| Kategorie | POIs (Tabelle / mit GT-Pixel) | GT-Pixel train | Zellen train | GT-Pixel test | Zellen test | GT-Pixel val | GT-Pixel gesamt | außerhalb Kandidatenraum verworfen | Status |
|---|---|---|---|---|---|---|---|---|---|
| art_antiques | 579 / 579 | 808 | 131 | 351 | 16 | 100 | 1259 | 0 | **ok** |
| religious_orgs | 25 / 25 | 88 | 17 | 3 | 2 | 2 | 93 | 0 | **thin** |
| medical_supplies | 13 / 13 | 15 | 9 | 2 | 2 | 4 | 21 | 0 | **thin** |
| sports_clubs | 328 / 322 | 12659 | 214 | 1667 | 29 | 1718 | 16044 | 1266 | **ok** |
| import_export | 17 / 17 | 31 | 15 | 0 | 0 | 1 | 32 | 0 | **thin** |

## tessera (N = 15.954.158 Kandidatenpixel)

| Kategorie | POIs (Tabelle / mit GT-Pixel) | GT-Pixel train | Zellen train | GT-Pixel test | Zellen test | GT-Pixel val | GT-Pixel gesamt | außerhalb Kandidatenraum verworfen | Status |
|---|---|---|---|---|---|---|---|---|---|
| art_antiques | 579 / 579 | 787 | 133 | 358 | 16 | 104 | 1249 | 0 | **ok** |
| religious_orgs | 25 / 25 | 84 | 17 | 1 | 1 | 2 | 87 | 0 | **thin** |
| medical_supplies | 13 / 13 | 14 | 9 | 2 | 2 | 6 | 22 | 0 | **thin** |
| sports_clubs | 328 / 322 | 12683 | 215 | 1662 | 28 | 1714 | 16059 | 2317 | **ok** |
| import_export | 17 / 17 | 28 | 15 | 0 | 0 | 1 | 29 | 0 | **thin** |
