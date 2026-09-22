"""
build_sdm_arm_ab_tessera_dedupnorm.py

TESSERA-Normalisierung nachziehen (2026-09-22, Jakob): Arm A (roh) und Arm B (zufaellig
projiziert) fuer TESSERA NEU rechnen, mit denselben z-score-Statistiken, die Arm C (der
Dedup-Checkpoint) bereits nutzt (`norm_stats_tessera_pixels_dedup.json`, via
`configs/tessera_v1_dedup.yaml`) - statt der alten `norm_stats_tessera_pixels.json`
(`configs/tessera_v1.yaml`), die Arm A/B bisher benutzt haben (siehe MANIFEST.md-Befund: max.
Kanalmittel-Shift 0,048 Std, Median 0,016 Std - klein, aber nicht Null).

Nichts wird ueberschrieben: neue Projektions-/Poolungsdateien mit Suffix `_dedupnorm`. Arm C
(bereits mit den Dedup-Statistiken gerechnet) wird NICHT angefasst.

Technik: `load_backbone_yaml_config` in den jeweils aufrufenden Modulen (build_sdm_projected_
field.py fuer Arm B, build_sdm_pooled_embeddings_dedup.py fuer Arm A) per Monkeypatch so
umgeleitet, dass fuer backbone='tessera' `configs/tessera_v1_dedup.yaml` statt `configs/
tessera_v1.yaml` gelesen wird - identisches Schema (normalization/data.norm_stats_path), zeigt
aber auf die Dedup-Statistik. Kein neues Statistik-Fitting, keine Aenderung an train.py/
apply_normalization - Wiederverwendung der bereits beim Dedup-Training gefitteten Datei.
Arm A braucht keine Reprojektion (poolt direkt ueber das rohe Raster); Arm B braucht eine neue
Reprojektion (Random-Head-Eingabe aendert sich mit der Normalisierung).
"""
import yaml

import geopandas as gpd
import pandas as pd

import config as cfg
import build_sdm_pooled_embeddings as bspe
import build_sdm_pooled_embeddings_dedup as bspd
import build_sdm_projected_field as bspf
import run_landuse_probe as rlp
from build_sdm_lsoa_splits import get_aoi_utm_reference, load_lsoa_geometries_for_london
from build_sdm_targets import CODE_COL, load_ts062

BACKBONE = "tessera"
DEDUPNORM_YAML = cfg.ROOT / "configs" / "tessera_v1_dedup.yaml"
NEW_PROJECTED_ROOT = cfg.DATA / "interim" / "sdm_projected_tessera_dedupnorm"
OUT_DIR = cfg.DATA / "processed"


def load_backbone_yaml_config_dedupnorm(name: str) -> dict:
    """Wie run_landuse_probe.load_backbone_yaml_config(), aber fuer 'tessera' wird
    configs/tessera_v1_dedup.yaml gelesen (dieselbe Statistik wie Arm C) statt configs/
    tessera_v1.yaml (die alte, bisher fuer Arm A/B verwendete)."""
    if name == "tessera":
        with open(DEDUPNORM_YAML) as f:
            return yaml.safe_load(f)
    return rlp.load_backbone_yaml_config(name)


bspf.load_backbone_yaml_config = load_backbone_yaml_config_dedupnorm  # fuer Arm B (Reprojektion)
bspd.load_backbone_yaml_config = load_backbone_yaml_config_dedupnorm  # fuer Arm A (Poolung)
bspf.OUT_ROOT = NEW_PROJECTED_ROOT
bspe.PROJECTED_DIR = NEW_PROJECTED_ROOT


def reproject_arm_b():
    print(f"=== Arm B, TESSERA, Dedup-Statistik ({DEDUPNORM_YAML}): Reprojektion ===")
    aoi = gpd.read_file(bspf.AOI_PATH)
    results = bspf.run_backbone_arm(BACKBONE, "b", aoi, device="cpu")
    print(f"{len(results)} Kacheln projiziert nach {NEW_PROJECTED_ROOT / f'{BACKBONE}_armb'}")


def pool_arm(arm: str):
    print(f"\n=== Arm {arm.upper()}, TESSERA, Dedup-Statistik: Poolung (mit Dedup-Maske) ===")
    aoi, utm_crs, minx, miny = get_aoi_utm_reference()
    aoi_gpd = gpd.read_file(bspf.AOI_PATH)
    ts062 = load_ts062()
    lsoa_national = load_lsoa_geometries_for_london(aoi)
    lsoa_utm = lsoa_national[lsoa_national["LSOA21CD"].isin(set(ts062[CODE_COL]))].copy()
    lsoa_utm = lsoa_utm.rename(columns={"LSOA21CD": "lsoa_code"})[["lsoa_code", "geometry"]].to_crs(utm_crs)
    split_a = pd.read_csv(cfg.DATA / "splits" / "sdm_lsoa_splits_london.csv")
    lsoa_split = dict(zip(split_a["lsoa_code"], split_a["split"]))
    cell_lookup = bspe.load_cell_lookup()

    df = bspd.pool_backbone_arm_dedup(BACKBONE, arm, lsoa_utm, lsoa_split, cell_lookup,
                                       utm_crs, minx, miny, aoi_gpd)
    n_zero_all = int((df["n_pixels_all"] == 0).sum())
    n_zero_pure = int((df["n_pixels_pure"] == 0).sum())
    print(f"  LSOAs erreicht: {len(df)}/{len(lsoa_utm)}, 0-Pixel (alle): {n_zero_all}, (rein): {n_zero_pure}")
    out_path = OUT_DIR / f"sdm_pooled_{BACKBONE}_arm{arm}_dedupnorm.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Gespeichert: {out_path}")


def main():
    reproject_arm_b()
    pool_arm("a")
    pool_arm("b")


if __name__ == "__main__":
    main()
