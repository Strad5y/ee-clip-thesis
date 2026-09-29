# Recall@Top-k% (%) — AETHER-Kategorien, Stil Tab. 3

Werte in Prozent; Klammer = 95-%-Zellebenen-Bootstrap-CI (1000 Resamples, Seed 0). Zufall = K/N analytisch. Schwelle/Top-Menge immer über ganz London (alle Kandidatenpixel), Recall über die GT-Pixel des jeweiligen Splits. Status: `thin` = < 50 Test-GT-Pixel oder < 3 Testzellen, `no_gt` = keine GT-Pixel (Recall undefiniert). Hinweis: Der Prototyp wird aus den GT-Pixeln der Train-Zellen gebildet; seine train/all-Werte enthalten daher die eigenen Stützpixel.

## Split: test (Hauptblock)

### ae

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **ok** | **ok** | **no_gt** | **no_gt** |
| | GT-Pixel / Zellen | 351 / 16 | 3269 / 105 | 218 / 63 | 0 / 0 | 0 / 0 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 10.5 [0.0–48.2] | 2.0 [1.1–3.1] | 8.7 [3.0–15.4] | — | — |
| 1% | Trainiert | 21.4 [10.1–37.0] | 7.4 [3.4–12.2] | 8.3 [4.3–13.7] | — | — |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 48.7 [30.0–74.1] | 9.7 [7.0–12.7] | 34.4 [23.2–45.1] | — | — |
| 5% | Trainiert | 80.9 [44.9–87.4] | 24.0 [15.5–34.3] | 39.0 [25.1–53.7] | — | — |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 69.8 [61.0–89.6] | 20.0 [15.9–24.6] | 51.4 [37.0–65.1] | — | — |
| 10% | Trainiert | 94.3 [56.7–98.9] | 36.3 [27.4–45.7] | 62.8 [48.9–77.7] | — | — |

### tessera

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **ok** | **ok** | **no_gt** | **no_gt** |
| | GT-Pixel / Zellen | 358 / 16 | 3210 / 105 | 221 / 62 | 0 / 0 | 0 / 0 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 60.1 [8.5–70.5] | 2.6 [1.7–3.5] | 14.0 [5.7–22.2] | — | — |
| 1% | Trainiert | 44.4 [3.8–53.0] | 7.0 [4.8–9.4] | 4.5 [1.5–8.3] | — | — |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 81.0 [23.6–92.2] | 13.5 [11.1–15.9] | 41.2 [31.0–49.6] | — | — |
| 5% | Trainiert | 81.6 [30.9–91.7] | 23.0 [18.0–28.4] | 26.2 [17.8–35.8] | — | — |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 88.8 [48.6–96.6] | 26.2 [22.1–30.2] | 68.8 [58.4–77.7] | — | — |
| 10% | Trainiert | 92.7 [67.6–97.7] | 35.7 [29.8–41.8] | 58.4 [49.0–67.3] | — | — |

## Split: train

### ae

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **ok** | **ok** | **no_gt** | **no_gt** |
| | GT-Pixel / Zellen | 808 / 131 | 30949 / 854 | 2637 / 528 | 0 / 0 | 0 / 0 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 23.3 [14.5–33.9] | 2.5 [2.1–3.0] | 13.8 [10.8–17.1] | — | — |
| 1% | Trainiert | 27.8 [20.7–35.9] | 5.9 [4.8–7.1] | 12.6 [9.5–15.8] | — | — |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 54.1 [43.7–63.6] | 12.3 [11.2–13.7] | 41.4 [37.0–46.0] | — | — |
| 5% | Trainiert | 77.2 [69.4–83.3] | 25.7 [22.8–28.5] | 45.8 [40.8–51.0] | — | — |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 71.0 [61.9–78.3] | 23.0 [21.3–24.9] | 59.3 [54.5–63.9] | — | — |
| 10% | Trainiert | 91.3 [86.6–95.0] | 41.4 [38.1–44.6] | 69.9 [65.8–73.7] | — | — |

### tessera

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **ok** | **ok** | **no_gt** | **no_gt** |
| | GT-Pixel / Zellen | 787 / 133 | 31030 / 855 | 2601 / 529 | 0 / 0 | 0 / 0 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 25.0 [16.1–34.5] | 3.3 [2.9–3.6] | 14.0 [11.4–16.8] | — | — |
| 1% | Trainiert | 23.4 [17.2–29.4] | 7.6 [6.8–8.5] | 5.3 [4.0–6.9] | — | — |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 60.9 [51.3–68.5] | 15.2 [14.0–16.3] | 46.3 [42.4–50.2] | — | — |
| 5% | Trainiert | 64.9 [56.1–72.1] | 24.9 [22.9–26.9] | 28.1 [25.2–31.2] | — | — |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 75.2 [68.0–80.9] | 27.7 [26.0–29.5] | 68.2 [64.7–71.5] | — | — |
| 10% | Trainiert | 84.2 [77.6–89.1] | 39.6 [37.0–41.8] | 53.7 [50.1–57.6] | — | — |

## Split: all

### ae

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **ok** | **ok** | **no_gt** | **no_gt** |
| | GT-Pixel / Zellen | 1259 / 164 | 38109 / 1053 | 3220 / 653 | 0 / 0 | 0 / 0 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 18.8 [11.2–28.3] | 2.4 [2.0–2.8] | 13.4 [10.8–16.1] | — | — |
| 1% | Trainiert | 24.9 [20.1–31.3] | 5.9 [4.8–7.1] | 11.9 [9.5–14.6] | — | — |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 51.1 [44.1–58.6] | 11.9 [10.8–13.0] | 40.2 [36.2–44.4] | — | — |
| 5% | Trainiert | 77.4 [69.2–82.9] | 25.1 [22.4–27.7] | 44.8 [40.2–49.5] | — | — |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 69.8 [63.7–75.9] | 22.3 [20.8–23.8] | 57.3 [53.0–61.7] | — | — |
| 10% | Trainiert | 91.7 [86.6–95.2] | 40.5 [37.5–43.2] | 68.7 [64.9–72.2] | — | — |

### tessera

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **ok** | **ok** | **no_gt** | **no_gt** |
| | GT-Pixel / Zellen | 1249 / 167 | 38143 / 1055 | 3180 / 651 | 0 / 0 | 0 / 0 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 34.3 [17.9–48.4] | 3.2 [2.9–3.5] | 14.1 [11.8–16.5] | — | — |
| 1% | Trainiert | 29.0 [18.0–37.8] | 7.5 [6.7–8.3] | 5.2 [3.9–6.6] | — | — |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 65.2 [50.2–76.8] | 15.0 [14.0–15.9] | 46.6 [43.1–49.7] | — | — |
| 5% | Trainiert | 68.4 [55.3–78.2] | 24.7 [23.0–26.5] | 27.5 [24.6–30.5] | — | — |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 77.7 [67.6–85.7] | 27.6 [26.1–29.0] | 69.0 [66.0–71.8] | — | — |
| 10% | Trainiert | 85.6 [77.4–90.8] | 39.3 [37.2–41.3] | 54.0 [50.6–57.0] | — | — |
