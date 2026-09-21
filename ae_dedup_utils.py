"""
ae_dedup_utils.py

AlphaEarth-Dedup (2026-09-21, Jakob): AlphaEarth nach EXAKT derselben Prozedur wie TESSERA
deduplizieren (Eindeutigkeitsmaske + AOI-Polygon-Maske + Restaufloesung ueber Weltkoordinaten).
Die Masken- und Restaufloesungslogik wird UNVERAENDERT aus tessera_dedup_utils.py importiert
(beide Funktionen sind bereits backbone-parametrisiert: build_full_mask_lookup(backbone),
resolve_residual_duplicates(df, meta, candidates)). Dieses Modul ergaenzt nur die Zerlegung der
entfernten Zeilen nach Ursache (dieselben drei disjunkten Kategorien wie
diag_removal_rate_breakdown.py fuer TESSERA), damit sie fuer alle drei Artefakte identisch
berechnet wird.
"""
import numpy as np
import pandas as pd

from diag_removal_rate_breakdown import build_aoi_only_mask_lookup, lookup_bool

BACKBONE = "ae"


def primary_and_aoi_flags(df: pd.DataFrame, combined_lookup: dict, aoi_only_lookup: dict) -> tuple:
    """Pro Zeile: (primary_ok, aoi_ok) - Eindeutigkeit+AOI kombiniert bzw. nur AOI-Polygon."""
    return lookup_bool(df, combined_lookup), lookup_bool(df, aoi_only_lookup)


def breakdown_counts(primary_ok: np.ndarray, aoi_ok: np.ndarray, kept_final: np.ndarray) -> dict:
    """Disjunkte Zerlegung der entfernten Zeilen (identische Definition wie bei TESSERA):
      - residual:   primaer OK, aber im Endergebnis nicht mehr vorhanden (Restaufloesung)
      - aoi:        primaer nicht OK UND ausserhalb AOI-Polygon
      - uniqueness: primaer nicht OK, aber innerhalb AOI-Polygon (Eindeutigkeitsverlust)
    kept_final: bool je Zeile, True wo die Zeile im Endergebnis steht."""
    residual = primary_ok & ~kept_final
    aoi = (~primary_ok) & (~aoi_ok)
    uniq = (~primary_ok) & aoi_ok
    n_removed = int((~kept_final).sum())
    out = {"n_before": len(primary_ok), "n_removed": n_removed,
           "uniqueness": int(uniq.sum()), "aoi": int(aoi.sum()), "residual": int(residual.sum())}
    assert out["uniqueness"] + out["aoi"] + out["residual"] == n_removed, \
        "Kategorien summieren nicht auf die Gesamtzahl - Bug."
    return out


def print_breakdown(label: str, c: dict) -> None:
    n, r = c["n_before"], c["n_removed"]
    print(f"\nZerlegung der entfernten Zeilen ({label}): {r} von {n} ({100 * r / n:.3f}%)")
    for key, name in (("uniqueness", "Eindeutigkeitsmaske (innerhalb AOI, nicht naechste Kachel)"),
                      ("aoi", "AOI-Polygon-Maske (ausserhalb Greater London)"),
                      ("residual", "Restaufloesung (Grenzlinien-Rest)")):
        v = c[key]
        print(f"  {name}: {v} ({100 * v / n:.3f}% von allen, "
              f"{100 * v / r if r else 0:.1f}% der Entfernten)")


def pixel_key(df: pd.DataFrame, tile_names: list) -> np.ndarray:
    """Eindeutiger int64-Schluessel je (tile,row,col) - vektorisierte Alternative zu Tupel-Sets."""
    code = {t: i for i, t in enumerate(sorted(tile_names))}
    tc = df["tile"].map(code).to_numpy(dtype=np.int64)
    r = df["row"].to_numpy(dtype=np.int64)
    c = df["col"].to_numpy(dtype=np.int64)
    assert r.max() < (1 << 15) and c.max() < (1 << 15), "row/col ueberschreiten 15 Bit - Schluessel kollidiert"
    return (tc << 30) | (r << 15) | c
