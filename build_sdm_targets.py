"""
build_sdm_targets.py

Phase 2 (INSTRUCTIONS_sdm.md): normiert die TS062-NS-SeC-Rohdaten (data/raw/sdm/London_SDM.csv,
manuell von Nomis geladen, siehe MANIFEST.md) zu einer normierten Zielverteilung pro LSOA
fuer die spaetere Socioeconomic-Distribution-Mapping-Auswertung (Phase 6/7).

Nomis-Bulk-Format (Phase-1-Bericht): echte Kopfzeile in Zeile 8 (1-indiziert), Daten ab
Zeile 10, Footer (Disclosure-Control-Hinweis) ab Zeile 5004. Join-Key ist 'mnemonic'
(LSOA21CD) - NICHT der Name, der ist nicht als stabiler Schluessel gedacht (siehe
INSTRUCTIONS_sdm.md Phase 3: "Join ... NIEMALS ueber den Namen").

Zwei Zielvarianten (Nutzervorgabe, nachdem sich AETHERs eigene Kategorienwahl nicht
rekonstruieren liess - das oeffentliche Repo enthaelt nur die 13-Datei-Pretraining-Pipeline,
keinen Metrik-/Downstream-Code, keine Klassifikationsschicht in den Checkpoints):
  - 9cat: alle neun NS-SeC-Kategorien, inkl. L15 (Vollzeitstudierende).
  - 8cat: dieselben acht Kategorien OHNE L15, erneut auf deren eigene Summe normiert (NICHT
    einfach die 9cat-Anteile ohne L15 weiterverwenden - sonst summieren die Zeilen nicht mehr
    auf 1).
Die Total-Spalte geht in KEINER Variante als Kategorie ein, dient ausschliesslich der
Validierung (Summe der 9cat-Kategorien vs. Total).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

SDM_DIR = cfg.DATA / "raw" / "sdm"  # umgezogen von data/SDM/ (ausserhalb dieser Session) - Pfad hier nachgezogen
TS062_PATH = SDM_DIR / "London_SDM.csv"
OUT_PATH_9CAT = cfg.DATA / "processed" / "sdm_targets_london_9cat.parquet"
OUT_PATH_8CAT = cfg.DATA / "processed" / "sdm_targets_london_8cat.parquet"

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
CAT_KEYS_9 = list(CATEGORY_COLS.keys())
CAT_KEYS_8 = [k for k in CAT_KEYS_9 if k != "l15"]


def load_ts062(path: Path = TS062_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=HEADER_ROW, nrows=N_DATA_ROWS)
    assert len(df) == N_DATA_ROWS, (
        f"{len(df)} Zeilen statt erwarteter {N_DATA_ROWS} - Header-/Footer-Grenzen "
        f"(HEADER_ROW/N_DATA_ROWS) gegen die aktuelle CSV pruefen."
    )
    missing = [c for c in [NAME_COL, CODE_COL, TOTAL_COL, *CATEGORY_COLS.values()] if c not in df.columns]
    assert not missing, f"Spalten fehlen in {path}: {missing}"
    return df


def build_variant(base: pd.DataFrame, cat_keys: list, label: str, out_path: Path,
                   check_against_total: bool = False) -> pd.DataFrame:
    """Baut eine Zielvariante (9cat oder 8cat): Summe ueber GENAU die cat_keys dieser Variante
    (nicht wiederverwendet aus einer anderen Variante - jede Variante normiert auf ihre eigene
    Summe, das ist der Punkt der Uebung fuer 8cat). check_against_total nur fuer 9cat sinnvoll,
    da die Total-Spalte alle neun Kategorien abdeckt, nicht acht."""
    out = base[["lsoa_code", "lsoa_name", "total", *cat_keys]].copy()
    out["category_sum"] = out[cat_keys].sum(axis=1)

    print(f"\n=== Variante {label} ({len(cat_keys)} Kategorien: {cat_keys}) ===")
    if check_against_total:
        mismatch = out["category_sum"] != out["total"]
        n_mismatch = int(mismatch.sum())
        print(f"Validierung vor Normierung:")
        print(f"  Zeilen mit Summe({len(cat_keys)} Kategorien) != Total: {n_mismatch} von {len(out)}")
        if n_mismatch > 0:
            diffs = out.loc[mismatch, "category_sum"] - out.loc[mismatch, "total"]
            print(f"    Differenz (Summe-Total): min={diffs.min()} max={diffs.max()} "
                  f"mean={diffs.mean():.4f}")
            print(f"    Beispielzeilen (bis zu 5): "
                  f"{out.loc[mismatch, ['lsoa_code', 'total', 'category_sum']].head(5).to_dict('records')}")

    zero_sum = out["category_sum"] == 0
    n_zero = int(zero_sum.sum())
    print(f"  Zeilen mit Summe({len(cat_keys)} Kategorien) == 0: {n_zero}")
    if n_zero > 0:
        print(f"    Betroffene lsoa_code: {out.loc[zero_sum, 'lsoa_code'].tolist()}")

    prop_cols = [f"p_{k}" for k in cat_keys]
    props = out[cat_keys].div(out["category_sum"].replace(0, np.nan), axis=0)
    props.columns = prop_cols
    out = pd.concat([out, props], axis=1)

    row_sum = out[prop_cols].sum(axis=1)
    valid = ~zero_sum
    max_dev = (row_sum[valid] - 1.0).abs().max()
    print(f"Normierung: max. Abweichung der Zeilensumme von 1.0 (gueltige Zeilen, "
          f"n={int(valid.sum())}): {max_dev:.2e}")
    assert max_dev < 1e-6, f"Normierung fehlerhaft: max. Abweichung {max_dev:.2e} > 1e-6"
    if n_zero > 0:
        assert out.loc[zero_sum, prop_cols].isna().all(axis=None), (
            "Summe-0-Zeilen sollten NaN-Anteile haben (0/0), nicht stillschweigend 0."
        )

    print(f"\nAnteile je Kategorie (Mittelwert/Median/Min/Max, ueber gueltige Zeilen):")
    stats = out.loc[valid, prop_cols].agg(["mean", "median", "min", "max"]).T
    stats = stats.sort_values("mean", ascending=False)
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nDrei haeufigste Kategorien (nach Mittelwert):")
    print(stats.head(3).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nDrei seltenste Kategorien (nach Mittelwert):")
    print(stats.tail(3).to_string(float_format=lambda x: f"{x:.4f}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\nGespeichert: {out_path} ({len(out)} Zeilen, Spalten: {list(out.columns)})")
    return out


def report_shift(out_9cat: pd.DataFrame, out_8cat: pd.DataFrame) -> None:
    """Vergleicht die Anteile der acht gemeinsamen Kategorien zwischen 9cat und 8cat -
    8cat MUSS auf der eigenen (8er-)Summe normiert worden sein (siehe build_variant), sonst
    waere diese Verschiebung tautologisch klein. Zusaetzlich die LSOAs mit dem hoechsten
    L15-Anteil (9cat) zeigen, dort sollte die Verschiebung am deutlichsten ausfallen."""
    merged = out_9cat[["lsoa_code", "p_l15"]].merge(
        out_8cat[["lsoa_code"] + [f"p_{k}" for k in CAT_KEYS_8]], on="lsoa_code"
    ).merge(
        out_9cat[["lsoa_code"] + [f"p_{k}" for k in CAT_KEYS_8]], on="lsoa_code",
        suffixes=("_8cat", "_9cat")
    )
    print("\n=== Verschiebung 8cat vs. 9cat (pro Kategorie, absolute Aenderung) ===")
    rows = []
    for k in CAT_KEYS_8:
        delta = (merged[f"p_{k}_8cat"] - merged[f"p_{k}_9cat"]).abs()
        rows.append({"kategorie": k, "mean_abs_delta": delta.mean(), "max_abs_delta": delta.max()})
    shift = pd.DataFrame(rows).sort_values("mean_abs_delta", ascending=False)
    print(shift.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    top_l15 = merged.sort_values("p_l15", ascending=False).head(5)
    print("\nDie 5 LSOAs mit dem hoechsten L15-Anteil (9cat) - Verschiebung dort zur Kontrolle:")
    show_cols = ["lsoa_code", "p_l15"] + [f"p_{k}_9cat" for k in CAT_KEYS_8[:2]] + [f"p_{k}_8cat" for k in CAT_KEYS_8[:2]]
    print(top_l15[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    df = load_ts062()
    print(f"Eingelesen: {len(df)} Zeilen aus {TS062_PATH}")

    base = df[[CODE_COL, NAME_COL, TOTAL_COL] + [CATEGORY_COLS[k] for k in CAT_KEYS_9]].copy()
    base = base.rename(columns={CODE_COL: "lsoa_code", NAME_COL: "lsoa_name", TOTAL_COL: "total"})
    base = base.rename(columns={CATEGORY_COLS[k]: k for k in CAT_KEYS_9})

    n_dup = int(base["lsoa_code"].duplicated().sum())
    print(f"Doppelte lsoa_code-Werte (sollte 0 sein): {n_dup}")

    out_9cat = build_variant(base, CAT_KEYS_9, "9cat", OUT_PATH_9CAT, check_against_total=True)
    out_8cat = build_variant(base, CAT_KEYS_8, "8cat (ohne L15)", OUT_PATH_8CAT, check_against_total=False)
    report_shift(out_9cat, out_8cat)


if __name__ == "__main__":
    main()
