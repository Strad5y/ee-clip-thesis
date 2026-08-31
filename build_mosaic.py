"""
build_mosaic.py

Baut das AlphaEarth-Mosaik (data/ee/alphaearth_london.tif) aus den 2 heruntergeladenen
Rohkacheln (unterschiedliche UTM-Zonen).

Reihenfolge ist kritisch (Nutzervorgabe): ERST dequantisieren, DANN reprojizieren -
Interpolation in der Wurzel-Domaene (rohe int8-Werte vor der vorzeichenerhaltenden
quadratischen Dequantisierung, siehe dequantize.py) wuerde falsche Werte erzeugen.
Resampling ist nearest, nie bilinear - Embeddings benachbarter Landbedeckungen duerfen
nicht gemischt/gemittelt werden. Ziel-CRS fuer London: EPSG:32630.

TESSERA bekommt KEIN Mosaik mehr (siehe Sitzungsvorfall: der Reproject-Schritt pro Band
war fuer 128 Baender x 20 Kacheln viel zu langsam) - spatial_join.py joint TESSERA
stattdessen kachelweise direkt gegen die Rohkacheln (join_pois_to_tiles()), jede Kachel
in ihrer eigenen nativen CRS, kein Rasterreproject noetig. AlphaEarth bleibt bei einem
fertigen Mosaik (eine Kachel in der Kachelliste), damit beide Backbones denselben
Join-Codepfad in spatial_join.py durchlaufen.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject

import config as cfg
from dequantize import dequantize_ae

TARGET_CRS = "EPSG:32630"
AE_TILES = [
    cfg.DATA / "raw" / "alphaearth" / "tiles" / "xv3kz2lbcvqtied12-0000008192-0000000000.tiff",
    cfg.DATA / "raw" / "alphaearth" / "tiles" / "xxoxxdt2k4ayimzh1-0000008192-0000008192.tiff",
]
AE_INTERIM_DIR = cfg.DATA / "interim" / "ae_dequant_reproj"
AE_OUT = cfg.DATA / "ee" / "alphaearth_london.tif"


def dequantize_and_reproject_ae_tile(src_path: Path, out_path: Path, target_crs=TARGET_CRS) -> None:
    """Schritt 1 (dequantisieren) MUSS vor Schritt 2 (reprojizieren) passieren - sonst
    interpoliert reproject() in der quantisierten int8-Wurzel-Domaene statt im echten
    Embedding-Raum. Ein einziger reproject()-Aufruf fuer alle Baender (nicht pro Band -
    das war die Ursache des TESSERA-Hangs, siehe Modul-Docstring)."""
    with rasterio.open(src_path) as src:
        raw = src.read()  # (bands, H, W) int8, native CRS/Transform

        # 1) Dequantisieren IN DER NATIVEN CRS/TRANSFORM (vor jeder Reprojektion)
        bands, h, w = raw.shape
        dequant = dequantize_ae(raw.transpose(1, 2, 0).reshape(-1, bands)).reshape(h, w, bands)
        dequant = dequant.transpose(2, 0, 1).astype(np.float32)  # zurueck zu (bands, H, W)

        # 2) Reprojizieren NACH der Dequantisierung, resampling=nearest, EIN Aufruf fuer
        # alle Baender gleichzeitig (source/destination als 3D-Array)
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        dst_profile = src.profile.copy()
        dst_profile.update(
            crs=target_crs, transform=dst_transform, width=dst_w, height=dst_h,
            dtype="float32", nodata=np.nan, compress="lzw", BIGTIFF="YES",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **dst_profile) as dst:
            reproject(
                source=dequant, destination=rasterio.band(dst, list(range(1, bands + 1))),
                src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                dst_transform=dst_transform, dst_crs=target_crs, dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
    print(f"  {src_path.name} -> {out_path} (CRS {src.crs} -> {target_crs}, "
          f"{dst_w}x{dst_h}, nearest)")


def build_alphaearth_mosaic() -> Path:
    print("=== AlphaEarth: dequantisieren -> reprojizieren -> mergen ===")
    reproj_paths = []
    for tile_path in AE_TILES:
        out_path = AE_INTERIM_DIR / f"{tile_path.stem}_dequant_32630.tif"
        if out_path.exists():
            print(f"  {out_path.name} existiert bereits, ueberspringe Dequant/Reproject")
        else:
            dequantize_and_reproject_ae_tile(tile_path, out_path)
        reproj_paths.append(out_path)

    print("  Merge (resampling=nearest)...")
    AE_OUT.parent.mkdir(parents=True, exist_ok=True)
    mosaic, out_transform = merge(reproj_paths, resampling=Resampling.nearest, nodata=np.nan)

    with rasterio.open(reproj_paths[0]) as ref:
        profile = ref.profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=out_transform,
                    compress="lzw", nodata=np.nan, BIGTIFF="YES")
    with rasterio.open(AE_OUT, "w", **profile) as dst:
        dst.write(mosaic)
    print(f"  Gespeichert: {AE_OUT} ({mosaic.shape[1]}x{mosaic.shape[2]}, {mosaic.shape[0]} Baender)")
    return AE_OUT


def verify_l2_norm(mosaic_path: Path, n_samples: int = 500, seed: int = 3) -> None:
    """Abnahme-Kriterium: L2-Norm-Stichprobe auf dem MOSAIK muss weiterhin nahe 1.0 liegen -
    bestaetigt, dass die Reihenfolge dequantisieren->reprojizieren die Werte nicht verfaelscht hat."""
    rng = np.random.default_rng(seed)
    with rasterio.open(mosaic_path) as src:
        rows = rng.integers(0, src.height, n_samples)
        cols = rng.integers(0, src.width, n_samples)
        arr = src.read()
        vecs = arr[:, rows, cols].T

    nan_mask = np.isnan(vecs).any(axis=1)
    valid = vecs[~nan_mask]
    norms = np.linalg.norm(valid, axis=1)
    print(f"  L2-Norm auf Mosaik ({mosaic_path.name}, n={len(valid)}/{n_samples} gueltig): "
          f"mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")


if __name__ == "__main__":
    ae_path = build_alphaearth_mosaic()
    verify_l2_norm(ae_path)
