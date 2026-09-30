# Recall@Top-k% (%) — AETHER-Kategorien, Stil Tab. 3

Werte in Prozent; Klammer = 95-%-Zellebenen-Bootstrap-CI (1000 Resamples, Seed 0). Zufall = K/N analytisch. Schwelle/Top-Menge immer über ganz London (alle Kandidatenpixel), Recall über die GT-Pixel des jeweiligen Splits. Status: `thin` = < 50 Test-GT-Pixel oder < 3 Testzellen, `no_gt` = keine GT-Pixel (Recall undefiniert). Hinweis: Der Prototyp wird aus den GT-Pixeln der Train-Zellen gebildet; seine train/all-Werte enthalten daher die eigenen Stützpixel.

## Split: test (Hauptblock)

### ae

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **thin** | **thin** | **ok** | **thin** |
| | GT-Pixel / Zellen | 351 / 16 | 3 / 2 | 2 / 2 | 1667 / 29 | 0 / 0 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 10.5 [0.0–48.2] | 0.0 [0.0–0.0] | 50.0 [0.0–100.0] | 4.0 [1.2–5.8] | — |
| 1% | Trainiert | 21.4 [10.1–37.0] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 4.0 [0.8–8.0] | — |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 48.7 [30.0–74.1] | 0.0 [0.0–0.0] | 50.0 [0.0–100.0] | 19.7 [12.8–27.0] | — |
| 5% | Trainiert | 80.9 [44.9–87.4] | 0.0 [0.0–0.0] | 100.0 [100.0–100.0] | 19.8 [12.7–25.7] | — |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 69.8 [61.0–89.6] | 66.7 [0.0–100.0] | 50.0 [0.0–100.0] | 32.1 [22.7–41.6] | — |
| 10% | Trainiert | 94.3 [56.7–98.9] | 100.0 [100.0–100.0] | 100.0 [100.0–100.0] | 35.6 [25.2–44.4] | — |

### tessera

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **thin** | **thin** | **ok** | **thin** |
| | GT-Pixel / Zellen | 358 / 16 | 1 / 1 | 2 / 2 | 1662 / 28 | 0 / 0 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 60.1 [8.5–70.5] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 16.5 [3.9–23.4] | — |
| 1% | Trainiert | 44.4 [3.8–53.0] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 6.9 [3.0–12.5] | — |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 81.0 [23.6–92.2] | 0.0 [0.0–0.0] | 50.0 [0.0–100.0] | 32.1 [11.7–40.4] | — |
| 5% | Trainiert | 81.6 [30.9–91.7] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 21.0 [13.5–30.8] | — |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 88.8 [48.6–96.6] | 0.0 [0.0–0.0] | 50.0 [0.0–100.0] | 41.2 [20.1–48.8] | — |
| 10% | Trainiert | 92.7 [67.6–97.7] | 0.0 [0.0–0.0] | 50.0 [0.0–100.0] | 31.0 [21.7–44.9] | — |

## Split: train

### ae

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **thin** | **thin** | **ok** | **thin** |
| | GT-Pixel / Zellen | 808 / 131 | 88 / 17 | 15 / 9 | 12659 / 214 | 31 / 15 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 23.3 [14.5–33.9] | 8.0 [0.0–15.4] | 46.7 [0.0–66.7] | 4.7 [3.3–6.0] | 38.7 [5.0–67.9] |
| 1% | Trainiert | 27.8 [20.7–35.9] | 4.5 [0.9–12.8] | 13.3 [0.0–30.0] | 4.0 [2.4–5.8] | 48.4 [0.0–73.2] |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 54.1 [43.7–63.6] | 37.5 [8.5–62.1] | 73.3 [22.2–93.8] | 18.6 [14.7–22.4] | 74.2 [40.0–89.7] |
| 5% | Trainiert | 77.2 [69.4–83.3] | 27.3 [11.4–41.5] | 66.7 [11.1–89.5] | 12.2 [8.6–16.3] | 67.7 [38.5–89.2] |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 71.0 [61.9–78.3] | 56.8 [24.6–82.7] | 73.3 [22.2–93.8] | 30.6 [25.6–35.6] | 77.4 [45.0–91.9] |
| 10% | Trainiert | 91.3 [86.6–95.0] | 67.0 [39.4–84.2] | 86.7 [55.6–100.0] | 22.1 [16.4–27.8] | 77.4 [50.0–97.1] |

### tessera

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **thin** | **thin** | **ok** | **thin** |
| | GT-Pixel / Zellen | 787 / 133 | 84 / 17 | 14 / 9 | 12683 / 215 | 28 / 15 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 25.0 [16.1–34.5] | 21.4 [0.0–47.4] | 57.1 [22.2–77.8] | 11.0 [7.1–15.1] | 32.1 [10.0–54.6] |
| 1% | Trainiert | 23.4 [17.2–29.4] | 8.3 [3.2–13.0] | 0.0 [0.0–0.0] | 3.4 [2.6–4.3] | 17.9 [0.0–37.0] |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 60.9 [51.3–68.5] | 42.9 [17.6–71.1] | 78.6 [55.6–100.0] | 23.2 [16.8–28.9] | 60.7 [30.0–77.8] |
| 5% | Trainiert | 64.9 [56.1–72.1] | 31.0 [14.9–42.4] | 7.1 [0.0–33.3] | 13.8 [11.5–16.4] | 53.6 [32.4–81.0] |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 75.2 [68.0–80.9] | 58.3 [35.2–79.2] | 100.0 [100.0–100.0] | 31.2 [24.2–37.5] | 78.6 [52.9–92.7] |
| 10% | Trainiert | 84.2 [77.6–89.1] | 47.6 [27.4–61.7] | 50.0 [11.1–68.4] | 23.1 [19.6–26.8] | 78.6 [60.9–93.1] |

## Split: all

### ae

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **thin** | **thin** | **ok** | **thin** |
| | GT-Pixel / Zellen | 1259 / 164 | 93 / 21 | 21 / 13 | 16044 / 261 | 32 / 16 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 18.8 [11.2–28.3] | 7.5 [0.0–14.7] | 42.9 [13.3–64.0] | 4.5 [3.4–5.7] | 37.5 [4.2–65.9] |
| 1% | Trainiert | 24.9 [20.1–31.3] | 4.3 [0.9–12.0] | 9.5 [0.0–21.4] | 4.9 [3.2–6.9] | 46.9 [0.0–70.3] |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 51.1 [44.1–58.6] | 35.5 [6.5–58.5] | 66.7 [35.0–88.0] | 19.5 [16.4–23.2] | 71.9 [40.9–88.0] |
| 5% | Trainiert | 77.4 [69.2–82.9] | 26.9 [11.1–40.9] | 76.2 [46.2–92.6] | 15.8 [11.7–20.2] | 68.8 [39.3–89.7] |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 69.8 [63.7–75.9] | 55.9 [23.3–81.0] | 71.4 [40.0–88.9] | 31.7 [27.5–36.3] | 78.1 [52.4–92.5] |
| 10% | Trainiert | 91.7 [86.6–95.2] | 68.8 [43.1–85.7] | 90.5 [71.4–100.0] | 26.8 [20.7–33.2] | 78.1 [50.0–96.2] |

### tessera

| k | Arm | Art and Antiques | Religious Organisations | Medical Equipment, Supplies and Pharmaceuticals | Sports Clubs and Associations | Import and Export Services |
|---|---|---|---|---|---|---|
| | Status | **ok** | **thin** | **thin** | **ok** | **thin** |
| | GT-Pixel / Zellen | 1249 / 167 | 87 / 20 | 22 / 13 | 16059 / 261 | 29 / 16 |
| 1% | Zufall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1% | Prototyp | 34.3 [17.9–48.4] | 20.7 [0.0–47.8] | 36.4 [11.8–60.9] | 11.0 [7.3–14.3] | 31.0 [7.4–52.9] |
| 1% | Trainiert | 29.0 [18.0–37.8] | 8.0 [3.0–12.8] | 0.0 [0.0–0.0] | 3.9 [3.0–4.9] | 17.2 [0.0–34.6] |
| 5% | Zufall | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| 5% | Prototyp | 65.2 [50.2–76.8] | 42.5 [18.4–71.1] | 63.6 [42.9–83.3] | 24.0 [18.4–28.8] | 62.1 [36.4–78.6] |
| 5% | Trainiert | 68.4 [55.3–78.2] | 31.0 [15.4–43.1] | 4.5 [0.0–16.7] | 14.6 [12.4–17.1] | 51.7 [30.3–75.9] |
| 10% | Zufall | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 10% | Prototyp | 77.7 [67.6–85.7] | 58.6 [36.4–79.5] | 90.9 [79.9–100.0] | 32.6 [26.4–37.6] | 79.3 [57.1–93.8] |
| 10% | Trainiert | 85.6 [77.4–90.8] | 47.1 [26.9–62.5] | 50.0 [23.1–64.7] | 23.9 [20.8–27.4] | 79.3 [61.9–93.1] |
