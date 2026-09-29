# Machbarkeit Recall@Top-k% — AETHER-Kategorien

Erzeugt von `eval/recall_topk.py feasibility` am 2026-09-30 00:07. Zuordnung: `queries/aether_categories.yaml` (vor dieser Auszählung festgelegt).

Status: `ok` / `thin` (< 50 Test-GT-Pixel oder < 3 Testzellen) / `no_gt` (keine GT-Pixel). GT-Pixel = Pixel der Dedup-Trainingspixeltabelle, die ein POI einer zugeordneten Klasse schneidet (kein Puffer), eingeschränkt auf den Kandidatenraum (AOI-Polygon + Eindeutigkeitsmaske + gültiges Arm-C-Feld). Zellen = 1-km-Zellen aus `data/splits/splits_london.csv`.

## Klassen

| Kategorie | Klasse | POIs in London (Tabelle) |
|---|---|---|
| Art and Antiques | `shop=art` | 330 |
| Art and Antiques | `shop=antiques` | 249 |
| Religious Organisations | `amenity=place_of_worship` | 3299 |
| Religious Organisations | `amenity=monastery` | 15 |
| Religious Organisations | `office=religion` | 25 |
| Medical Equipment, Supplies and Pharmaceuticals | `shop=medical_supply` | 13 |
| Medical Equipment, Supplies and Pharmaceuticals | `amenity=pharmacy` | 1277 |
| Medical Equipment, Supplies and Pharmaceuticals | `healthcare=pharmacy` | 1138 |
| Sports Clubs and Associations | — (keine Klasse) | — |
| Import and Export Services | — (keine Klasse) | — |

Unsichere Klassen (nicht in der GT, nur zur Prüfung): siehe YAML `uncertain_classes`.

## ae (N = 15.954.033 Kandidatenpixel)

| Kategorie | POIs (Tabelle / mit GT-Pixel) | GT-Pixel train | Zellen train | GT-Pixel test | Zellen test | GT-Pixel val | GT-Pixel gesamt | außerhalb Kandidatenraum verworfen | Status |
|---|---|---|---|---|---|---|---|---|---|
| art_antiques | 579 / 579 | 808 | 131 | 351 | 16 | 100 | 1259 | 0 | **ok** |
| religious_orgs | 3339 / 3339 | 30949 | 854 | 3269 | 105 | 3891 | 38109 | 0 | **ok** |
| medical_supplies | 1299 / 1298 | 2637 | 528 | 218 | 63 | 365 | 3220 | 0 | **ok** |
| sports_clubs | 0 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **no_gt** |
| import_export | 0 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **no_gt** |

## tessera (N = 15.954.158 Kandidatenpixel)

| Kategorie | POIs (Tabelle / mit GT-Pixel) | GT-Pixel train | Zellen train | GT-Pixel test | Zellen test | GT-Pixel val | GT-Pixel gesamt | außerhalb Kandidatenraum verworfen | Status |
|---|---|---|---|---|---|---|---|---|---|
| art_antiques | 579 / 579 | 787 | 133 | 358 | 16 | 104 | 1249 | 0 | **ok** |
| religious_orgs | 3339 / 3339 | 31030 | 855 | 3210 | 105 | 3903 | 38143 | 0 | **ok** |
| medical_supplies | 1299 / 1299 | 2601 | 529 | 221 | 62 | 358 | 3180 | 0 | **ok** |
| sports_clubs | 0 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **no_gt** |
| import_export | 0 / 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **no_gt** |
