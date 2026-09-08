"""
build_sdm_targets.py

Phase 2 (INSTRUCTIONS_sdm.md): normiert die TS062-NS-SeC-Rohdaten (data/SDM/London_SDM.csv,
manuell von Nomis geladen, siehe MANIFEST.md) zu einer normierten Zielverteilung pro LSOA
fuer die spaetere Socioeconomic-Distribution-Mapping-Auswertung (Phase 6/7).

Nomis-Bulk-Format (Phase-1-Bericht): echte Kopfzeile in Zeile 8 (1-indiziert), Daten ab
Zeile 10, Footer (Disclosure-Control-Hinweis) ab Zeile 5004. Join-Key ist 'mnemonic'
(LSOA21CD) - NICHT der Name, der ist nicht als stabiler Schluessel gedacht (siehe
INSTRUCTIONS_sdm.md Phase 3: "Join ... NIEMALS ueber den Namen").

Kategorienentscheidung (Nutzervorgabe, unabhaengig vom AETHER-Repo-Befund - das oeffentliche
Repo enthaelt keinen SDM/NS-SeC-Auswertungscode, siehe Bericht): alle neun NS-SeC-Kategorien
werden verwendet, inklusive L15 (Vollzeitstudierende). Die Total-Spalte geht NICHT als
Kategorie ein, dient ausschliesslich der Validierung (Summe der neun Kategorien vs. Total).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

SDM_DIR = cfg.DATA / "raw" / "sdm"  # umgezogen von data/SDM/ (ausserhalb dieser Session) - Pfad hier nachgezogen
TS062_PATH = SDM_DIR / "London_SDM.csv"
OUT_PATH = cfg.DATA / "processed" / "sdm_targets_london.parquet"

HEADER_ROW = 7        # 0-indiziert -> Nomis-Kopfzeile in Zeile 8 (1-indiziert)
N_DATA_ROWS = 4994     # Datenzeilen 10..5003 (1-indiziert), vor dem Footer ab Zeile 5004

NAME_COL = "2021 super output area - lower layer"
CODE_COL = "mnemonic"
TOTAL_COL = "Total: All usual residents aged 16 years and over"

# Kurzcodes fuer die neun NS-SeC-Kategorien (Reihenfolge wie in der CSV) -> volle Nomis-
# Spaltennamen. Kurzcodes werden fuer die Ausgabetabelle verwendet, volle Namen bleiben hier
# als Dokumentation der Zuordnung erhalten.
CATEGORY_COLS = {
    "l1_l3": "L1, L2 and L3 Higher managerial, administrative and professional occupations",
    "l4_l6": "L4, L5 and L6 Lower managerial, administrative and professional occupations",
    "l7": "L7 Intermediate occupations",
    "l8_l9": "L8 and L9 Small employers and own account workers",
    "l10_l11": "L10 and L11 Lower supervisory and technical occupations",
    "l12": "L12 Semi-routine occupations",
    "l13": "L13 Routine occupations",
    "l14": "L14.1 and L14.2 Never worked and long-term unemployed",
    "l15": "L15 Full-time students",
}
CAT_KEYS = list(CATEGORY_COLS.keys())


def load_ts062(path: Path = TS062_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=HEADER_ROW, nrows=N_DATA_ROWS)
    assert len(df) == N_DATA_ROWS, (
        f"{len(df)} Zeilen statt erwarteter {N_DATA_ROWS} - Header-/Footer-Grenzen "
        f"(HEADER_ROW/N_DATA_ROWS) gegen die aktuelle CSV pruefen."
    )
    missing = [c for c in [NAME_COL, CODE_COL, TOTAL_COL, *CATEGORY_COLS.values()] if c not in df.columns]
    assert not missing, f"Spalten fehlen in {path}: {missing}"
    return df


def main():
    df = load_ts062()
    print(f"Eingelesen: {len(df)} Zeilen aus {TS062_PATH}")

    raw = df[[CODE_COL, NAME_COL, TOTAL_COL] + [CATEGORY_COLS[k] for k in CAT_KEYS]].copy()
    raw = raw.rename(columns={CODE_COL: "lsoa_code", NAME_COL: "lsoa_name", TOTAL_COL: "total"})
    raw = raw.rename(columns={CATEGORY_COLS[k]: k for k in CAT_KEYS})

    n_dup = int(raw["lsoa_code"].duplicated().sum())
    print(f"Doppelte lsoa_code-Werte (sollte 0 sein): {n_dup}")

    raw["category_sum"] = raw[CAT_KEYS].sum(axis=1)

    mismatch = raw["category_sum"] != raw["total"]
    n_mismatch = int(mismatch.sum())
    print(f"\nValidierung vor Normierung:")
    print(f"  Zeilen mit Summe(9 Kategorien) != Total: {n_mismatch} von {len(raw)}")
    if n_mismatch > 0:
        diffs = raw.loc[mismatch, "category_sum"] - raw.loc[mismatch, "total"]
        print(f"    Differenz (Summe-Total): min={diffs.min()} max={diffs.max()} "
              f"mean={diffs.mean():.4f}")
        print(f"    Beispielzeilen (bis zu 5): "
              f"{raw.loc[mismatch, ['lsoa_code', 'total', 'category_sum']].head(5).to_dict('records')}")

    zero_sum = raw["category_sum"] == 0
    n_zero = int(zero_sum.sum())
    print(f"  Zeilen mit Summe(9 Kategorien) == 0: {n_zero}")
    if n_zero > 0:
        print(f"    Betroffene lsoa_code: {raw.loc[zero_sum, 'lsoa_code'].tolist()}")

    prop_cols = [f"p_{k}" for k in CAT_KEYS]
    props = raw[CAT_KEYS].div(raw["category_sum"].replace(0, np.nan), axis=0)
    props.columns = prop_cols
    out = pd.concat([raw, props], axis=1)

    row_sum = out[prop_cols].sum(axis=1)
    valid = ~zero_sum
    max_dev = (row_sum[valid] - 1.0).abs().max()
    print(f"\nNormierung: max. Abweichung der Zeilensumme von 1.0 (gueltige Zeilen, "
          f"n={int(valid.sum())}): {max_dev:.2e}")
    assert max_dev < 1e-6, f"Normierung fehlerhaft: max. Abweichung {max_dev:.2e} > 1e-6"
    if n_zero > 0:
        assert out.loc[zero_sum, prop_cols].isna().all(axis=None), (
            "Summe-0-Zeilen sollten NaN-Anteile haben (0/0), nicht stillschweigend 0."
        )

    print("\nAnteile je Kategorie (Mittelwert/Median/Min/Max, ueber gueltige Zeilen):")
    stats = out.loc[valid, prop_cols].agg(["mean", "median", "min", "max"]).T
    stats = stats.sort_values("mean", ascending=False)
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nDrei haeufigste Kategorien (nach Mittelwert):")
    print(stats.head(3).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nDrei seltenste Kategorien (nach Mittelwert):")
    print(stats.tail(3).to_string(float_format=lambda x: f"{x:.4f}"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH} ({len(out)} Zeilen, Spalten: {list(out.columns)})")


if __name__ == "__main__":
    main()
