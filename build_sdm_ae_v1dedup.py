"""
build_sdm_ae_v1dedup.py

Schritt 3 AlphaEarth (2026-09-21, Jakob): rechnet die SDM-Poolung fuer AlphaEarth ALLE DREI Arme
neu (A/B/C), Arm B/C mit frisch projiziertem Feld (Arm C: NEUER checkpoints/ae_v1_dedup/best.pt).
Eigene Datei; das TESSERA-Pendant (build_sdm_arm_c_tessera_v1dedup.py, nur Arm C) bleibt
unveraendert.

Arm A (rohes Raster + Maske) und Arm B (fester Random-Head, Seed 0 + Maske) haengen weder vom
neuen Checkpoint noch von pixels_ae ab - die Dateien sdm_pooled_ae_arm{a,b}_dedup.parquet vom
17.9. sind dieselbe Rechnung. Deshalb hier zusaetzlich die Bit-Identitaets-Pruefung der neu
berechneten Arm-A/B-Dateien gegen die 17.9.-Dateien (Reproduzierbarkeitscheck, Nutzervorgabe).
Arm C wird gegen die 17.9.-Datei (alter Checkpoint ae_v1) nur als Differenz berichtet.

Wiederverwendet (unveraendert) build_sdm_projected_field.py::run_backbone_arm() und
build_sdm_pooled_embeddings_dedup.py::pool_backbone_arm_dedup() via Monkeypatch der Pfade
(dieselbe Technik wie build_sdm_arm_c_tessera_v1dedup.py). Ersetzt nichts.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import config as cfg
import build_sdm_pooled_embeddings as bspe
import build_sdm_projected_field as bspf
import run_landuse_probe as rlp
from build_sdm_lsoa_splits import get_aoi_utm_reference, load_lsoa_geometries_for_london
from build_sdm_pooled_embeddings_dedup import load_cell_lookup, pool_backbone_arm_dedup
from build_sdm_targets import CODE_COL, load_ts062

BACKBONE = "ae"
NEW_CKPT = Path("checkpoints/ae_v1_dedup/best.pt")
NEW_PROJECTED_ROOT = cfg.DATA / "interim" / "sdm_projected_ae_v1dedup"
OUT_DIR = cfg.DATA / "processed"

rlp.CHECKPOINTS["ae"] = NEW_CKPT            # load_trained_head("ae") liest ab jetzt den neuen Checkpoint
bspf.OUT_ROOT = NEW_PROJECTED_ROOT           # run_backbone_arm() schreibt ins neue Verzeichnis
bspe.PROJECTED_DIR = NEW_PROJECTED_ROOT      # raster_paths_for_arm() liest aus dem neuen Verzeichnis


def reproject():
    aoi = gpd.read_file(bspf.AOI_PATH)
    for arm in ("b", "c"):
        print(f"=== Arm {arm.upper()}, AlphaEarth: Reprojektion "
              f"({'Random-Head Seed 0' if arm == 'b' else NEW_CKPT}) ===")
        results = bspf.run_backbone_arm(BACKBONE, arm, aoi, device="cpu")
        print(f"{len(results)} Kacheln projiziert nach {NEW_PROJECTED_ROOT / f'{BACKBONE}_arm{arm}'}")


def pool_all():
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    aoi_gpd = gpd.read_file(bspf.AOI_PATH)
    ts062 = load_ts062()
    lsoa_national = load_lsoa_geometries_for_london(aoi)
    lsoa_utm = lsoa_national[lsoa_national["LSOA21CD"].isin(set(ts062[CODE_COL]))].copy()
    lsoa_utm = lsoa_utm.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]].to_crs(utm_crs)
    print(f"  {len(lsoa_utm)} LSOAs")
    split_a = pd.read_csv(cfg.DATA / "splits" / "sdm_lsoa_splits_london.csv")
    lsoa_split = dict(zip(split_a["lsoa_code"], split_a["split"]))
    cell_lookup = load_cell_lookup()

    for arm in ("a", "b", "c"):
        print(f"\n=== Arm {arm.upper()}, AlphaEarth: Poolung (mit Dedup-Maske) ===")
        df = pool_backbone_arm_dedup(BACKBONE, arm, lsoa_utm, lsoa_split, cell_lookup,
                                      utm_crs, minx, miny, aoi_gpd)
        n_zero_all = int((df["n_pixels_all"] == 0).sum())
        n_zero_pure = int((df["n_pixels_pure"] == 0).sum())
        print(f"  LSOAs erreicht: {len(df)}/{len(lsoa_utm)}, 0-Pixel (alle): {n_zero_all}, (rein): {n_zero_pure}")
        out_path = OUT_DIR / f"sdm_pooled_{BACKBONE}_arm{arm}_v1dedup_ckpt.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Gespeichert: {out_path}")


def compare_with_old() -> None:
    print("\n=== Vergleich mit den Dateien vom 17.9. (sdm_pooled_ae_arm*_dedup.parquet) ===")
    for arm in ("a", "b", "c"):
        new = pd.read_parquet(OUT_DIR / f"sdm_pooled_{BACKBONE}_arm{arm}_v1dedup_ckpt.parquet")
        old = pd.read_parquet(OUT_DIR / f"sdm_pooled_{BACKBONE}_arm{arm}_dedup.parquet")
        new = new.sort_values("lsoa_code").reset_index(drop=True)
        old = old.sort_values("lsoa_code").reset_index(drop=True)
        same_cols = list(new.columns) == list(old.columns)
        same_codes = same_cols and (new["lsoa_code"].to_numpy() == old["lsoa_code"].to_numpy()).all()
        if not same_codes:
            print(f"  Arm {arm.upper()}: Spalten/LSOA-Codes weichen ab (Spalten gleich: {same_cols}) - kein Vergleich moeglich")
            continue
        num = [c for c in new.columns if c != "lsoa_code" and new[c].dtype.kind in "fiu"]
        identical = all(np.array_equal(new[c].to_numpy(), old[c].to_numpy(), equal_nan=True)
                        if new[c].dtype.kind == "f" else np.array_equal(new[c].to_numpy(), old[c].to_numpy())
                        for c in num)
        maxdiff = max(float(np.nanmax(np.abs(new[c].to_numpy(dtype=float) - old[c].to_numpy(dtype=float))))
                      for c in num)
        n_diff_cols = sum(not np.array_equal(new[c].to_numpy(), old[c].to_numpy(), equal_nan=True)
                          if new[c].dtype.kind == "f" else not np.array_equal(new[c].to_numpy(), old[c].to_numpy())
                          for c in num)
        tag = "BIT-IDENTISCH" if identical else "NICHT bit-identisch"
        print(f"  Arm {arm.upper()}: {tag} ({len(new)} LSOAs, {len(num)} numerische Spalten, "
              f"{n_diff_cols} mit Abweichung, max|diff|={maxdiff:.3e})")


def main():
    reproject()
    pool_all()
    compare_with_old()


if __name__ == "__main__":
    main()
