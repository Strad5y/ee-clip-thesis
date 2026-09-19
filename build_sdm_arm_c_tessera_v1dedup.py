"""
build_sdm_arm_c_tessera_v1dedup.py

Schritt 3 (2026-09-19, Jakob): projiziert und poolt NUR Arm C fuer TESSERA neu, mit dem neuen
checkpoints/tessera_v1_dedup/best.pt - Arm A/B und ganz AlphaEarth sind vom neuen Checkpoint
nicht betroffen, werden NICHT neu gerechnet.

Wiederverwendet build_sdm_projected_field.py::run_backbone_arm() (Reprojektion) und
build_sdm_pooled_embeddings_dedup.py::pool_backbone_arm_dedup() (Poolung MIT der
Eindeutigkeits+AOI-Maske, UNVERAENDERT - die Maske haengt nur an der Kachelgeometrie, nicht am
Checkpoint) via Monkeypatch der Checkpoint-/Ausgabepfade (dieselbe Technik wie in den
vorherigen build_sdm_*_dedup.py-Skripten). Ersetzt nichts - neues Ausgabeverzeichnis
(data/interim/sdm_projected_tessera_v1dedup/) und neue Poolungsdatei.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

import config as cfg
import build_sdm_pooled_embeddings as bspe
import build_sdm_projected_field as bspf
import run_landuse_probe as rlp
from build_sdm_lsoa_splits import get_aoi_utm_reference, load_lsoa_geometries_for_london
from build_sdm_pooled_embeddings_dedup import load_cell_lookup, pool_backbone_arm_dedup
from build_sdm_targets import CODE_COL, load_ts062

NEW_CKPT = Path("checkpoints/tessera_v1_dedup/best.pt")
NEW_PROJECTED_ROOT = cfg.DATA / "interim" / "sdm_projected_tessera_v1dedup"
OUT_DIR = cfg.DATA / "processed"

rlp.CHECKPOINTS["tessera"] = NEW_CKPT      # load_trained_head("tessera") liest ab jetzt den neuen Checkpoint
bspf.OUT_ROOT = NEW_PROJECTED_ROOT          # run_backbone_arm() schreibt ins neue Verzeichnis
bspe.PROJECTED_DIR = NEW_PROJECTED_ROOT     # raster_paths_for_arm() liest aus dem neuen Verzeichnis


def reproject():
    print("=== Arm C, TESSERA, neuer Checkpoint: Reprojektion ===")
    aoi = gpd.read_file(bspf.AOI_PATH)
    results = bspf.run_backbone_arm("tessera", "c", aoi, device="cpu")
    print(f"{len(results)} Kacheln projiziert nach {NEW_PROJECTED_ROOT / 'tessera_armc'}")
    return results


def pool():
    print("\n=== Arm C, TESSERA, neuer Checkpoint: Poolung (mit Dedup-Maske) ===")
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    aoi_gpd = gpd.read_file(bspf.AOI_PATH)
    ts062 = load_ts062()
    ts062_codes = set(ts062[CODE_COL])
    lsoa_national = load_lsoa_geometries_for_london(aoi)
    lsoa_utm = lsoa_national[lsoa_national["LSOA21CD"].isin(ts062_codes)].copy()
    lsoa_utm = lsoa_utm.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]].to_crs(utm_crs)
    print(f"  {len(lsoa_utm)} LSOAs")

    split_a = pd.read_csv(cfg.DATA / "splits" / "sdm_lsoa_splits_london.csv")
    lsoa_split = dict(zip(split_a["lsoa_code"], split_a["split"]))
    cell_lookup = load_cell_lookup()

    df = pool_backbone_arm_dedup("tessera", "c", lsoa_utm, lsoa_split, cell_lookup,
                                  utm_crs, minx, miny, aoi_gpd)
    n_zero_all = int((df["n_pixels_all"] == 0).sum())
    n_zero_pure = int((df["n_pixels_pure"] == 0).sum())
    print(f"  LSOAs erreicht: {len(df)}/{len(lsoa_utm)}, 0-Pixel (alle): {n_zero_all}, (rein): {n_zero_pure}")

    out_path = OUT_DIR / "sdm_pooled_tessera_armc_v1dedup_ckpt.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Gespeichert: {out_path}")
    return df


def main():
    reproject()
    pool()


if __name__ == "__main__":
    main()
