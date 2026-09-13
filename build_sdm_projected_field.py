"""
build_sdm_projected_field.py

Phase 4 (INSTRUCTIONS_sdm.md): projiziert jedes rohe EE-Pixel (TESSERA/AlphaEarth, Greater
London) einzeln durch den Projektionskopf des jeweiligen Backbones/Arms und persistiert das
Ergebnis als 256-Kanal-GeoTIFF je Kachel - EINMAL gerechnet, unabhaengig von Ziel- und
Split-Variante (haengt nur an Backbone/Arm), wird von SDM, Land-Use und klassenbasiertem
Retrieval gleichermassen wiederverwendet.

Arme (nur B und C brauchen ein Projektionsfeld - A poolt direkt ueber das rohe Raster, siehe
INSTRUCTIONS_sdm.md Phase 4):
  - B: untrainierter Kopf, IDENTISCHE Architektur/Seed wie run_landuse_probe.py::random_head()
  - C: trainierter Kopf aus dem jeweiligen best.pt (run_landuse_probe.py::load_trained_head())

Kritische Reihenfolge (Nutzervorgabe): ERST projizieren, DANN mitteln (Phase 5) - der Kopf ist
nichtlinear (GELU, L2-Norm) und wurde auf Einzelpixeln trainiert, ein gemittelter Rohvektor
liegt ausserhalb des Wertebereichs, den er je gesehen hat.

Normalisierung: identisch zu run_landuse_probe.py - z-score fuer TESSERA
(norm_stats_tessera_pixels.json), none fuer AlphaEarth (bereits L2-normalisiert, siehe
check_norm_stats_available() fuer den Pre-Flight-Check auf beide Faelle).

Raeumlicher Zuschnitt: nur der Bounding-Box-Ausschnitt jeder Kachel, der die AOI 'Greater
London' schneidet, wird gelesen/projiziert/geschrieben - die AlphaEarth-Rohkacheln sind
8192x8192 (81,92 km Kantenlaenge, Google-Tiling-Schema), Greater London deckt davon nur einen
Bruchteil ab; den Rest zu projizieren waere reine Verschwendung (keine der drei
Weiterverwendungen braucht Pixel ausserhalb der AOI). Rechteckiger BBox-Zuschnitt genuegt -
Phase 5/exactextract macht ohnehin die praezise flaechengewichtete Zuordnung zum LSOA-Polygon.
"""
import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

import config as cfg
import spatial_join as sj
from dequantize import dequantize_ae
from normalize import NONE, ZSCORE, apply_zscore, load_stats
from run_landuse_probe import (
    EE_DIMS, LATENT_DIM, check_architecture, load_backbone_yaml_config, load_trained_head,
    project as project_batch, random_head, resolve_normalization,
)

AOI_PATH = cfg.AOI_DIR / "london.gpkg"
OUT_ROOT = cfg.DATA / "interim" / "sdm_projected"
ARMS = ("b", "c")
BACKBONES_LIST = ("ae", "tessera")


def check_norm_stats_available() -> dict:
    """Pre-Flight (Nutzervorgabe): prueft fuer BEIDE Backbones, ob die fuer die konfigurierte
    Normalisierung noetige Statistikdatei vorhanden UND (fuer die Provenienz) im Git-Index
    getrackt ist - TESSERA brauchte dafuer 'git add -f' an der .gitignore vorbei (data/processed/
    ist sonst ignoriert), die AE-Entsprechung koennte denselben Fallstrick haben oder (da AE
    normalization='none' nutzt) schlicht nicht existieren muessen."""
    import subprocess
    report = {}
    for name in BACKBONES_LIST:
        yaml_cfg = load_backbone_yaml_config(name)
        mode, stats_path = resolve_normalization(yaml_cfg, f"configs/{name}_v1.yaml")
        if mode == NONE:
            report[name] = {"mode": mode, "stats_path": None, "exists": None, "tracked": None}
            continue
        exists = Path(stats_path).exists()
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", stats_path],
                                  capture_output=True).returncode == 0
        report[name] = {"mode": mode, "stats_path": stats_path, "exists": exists, "tracked": tracked}
    return report


def get_head(backbone: str, arm: str):
    if arm == "b":
        return random_head(backbone), load_backbone_yaml_config(backbone)
    elif arm == "c":
        head, ckpt_cfg = load_trained_head(backbone)
        return head, ckpt_cfg
    raise ValueError(f"Unbekannter Arm fuer Projektion: {arm!r} (nur 'b'/'c' brauchen einen Kopf)")


ROW_CHUNK = 200  # Sicherheitsgrenze: der AlphaEarth-AOI-Ausschnitt einer Kachel erreichte beim
# unverkleinerten Testlauf (h=4509, w=5827) ueber 100 GB RSS - der 64/128->256-Kanal-Sprung
# der Projektion vervierfacht/verdoppelt den Speicherbedarf gerade an der Stelle, wo die Kachel
# ohnehin schon Millionen Pixel hat. Nur die PROJEKTION (nicht Lesen/Dequantisieren/Normieren,
# die bleiben auf der ganzen Kachel guenstig) wird deshalb zeilenweise gemacht und direkt in
# den Ziel-GeoTIFF geschrieben - nie mehr als ein Chunk des 256-Kanal-Outputs im Speicher.


def project_tile(tile_path: Path, backbone: str, head, norm_mode: str, norm_stats,
                  aoi: gpd.GeoDataFrame, out_dir: Path, device: str = "cpu") -> dict | None:
    with rasterio.open(tile_path) as src:
        aoi_geom = aoi.to_crs(src.crs).geometry.iloc[0]
        minx, miny, maxx, maxy = aoi_geom.bounds
        tb = src.bounds
        ix0, iy0 = max(minx, tb.left), max(miny, tb.bottom)
        ix1, iy1 = min(maxx, tb.right), min(maxy, tb.top)
        if ix0 >= ix1 or iy0 >= iy1:
            return None  # Kachel liegt komplett ausserhalb der AOI-BBox

        window = from_bounds(ix0, iy0, ix1, iy1, transform=src.transform)
        window = window.round_lengths().round_offsets()
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width <= 0 or window.height <= 0:
            return None

        t0 = time.time()
        arr = src.read(window=window)  # (bands, h, w) - ganze Kachel, aber VOR der Projektion
        transform = src.window_transform(window)
        crs = src.crs

    bands, h, w = arr.shape
    x = np.moveaxis(arr, 0, -1)  # (h, w, bands) - dequantize_ae erwartet Baender zuletzt
    if backbone == "ae":
        x = dequantize_ae(x)
    valid = np.isfinite(x).all(axis=-1)  # (h, w)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0)
    peak_pre_project_bytes = arr.nbytes + x.nbytes  # dominanter Speicherposten VOR der Projektion

    if norm_mode == ZSCORE:
        x = apply_zscore(x.reshape(-1, bands), norm_stats).reshape(h, w, bands)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tile_path.stem}_proj256.tif"
    # tiled=True + interleave='band' ist KRITISCH, nicht kosmetisch (Nachtrag nach Phase-5-
    # Vorfall): der erste Lauf schrieb ungetilte Streifen mit interleave='pixel' (GDAL-Default) -
    # exact_extract() (Phase 5) braucht fuer cell_id/coverage nur Band 1, musste bei
    # Pixel-Interleave aber trotzdem JEDE Zeile ueber alle 256 Baender dekomprimieren, pro
    # Kachel x Polygon. Bei der 19-GB-AlphaEarth-Kachel fuehrte das zu einem Lauf, der nach
    # >26h noch keine einzige Kachel fertig hatte (siehe MANIFEST.md). Mit Tiling + Band-
    # Interleave liest exact_extract nur die tatsaechlich beruehrten Bloecke von Band 1.
    profile = dict(driver="GTiff", height=h, width=w, count=LATENT_DIM, dtype="float32",
                   crs=crs, transform=transform, compress="lzw", nodata=np.nan, BIGTIFF="YES",
                   tiled=True, blockxsize=256, blockysize=256, interleave="band")

    n_valid = int(valid.sum())
    running_min, running_max, running_sum = np.inf, -np.inf, 0.0
    peak_chunk_bytes = 0
    with rasterio.open(out_path, "w", **profile) as dst:
        for r0 in range(0, h, ROW_CHUNK):
            r1 = min(r0 + ROW_CHUNK, h)
            chunk_valid = valid[r0:r1]  # (rc, w)
            chunk_flat = x[r0:r1].reshape(-1, bands)  # (rc*w, bands)
            out_flat = project_batch(head, chunk_flat, device=device)  # (rc*w, 256)
            peak_chunk_bytes = max(peak_chunk_bytes, chunk_flat.nbytes + out_flat.nbytes)
            out_chunk = out_flat.reshape(r1 - r0, w, LATENT_DIM)
            out_chunk[~chunk_valid] = np.nan
            out_chunk = np.moveaxis(out_chunk, -1, 0).astype(np.float32)  # (256, rc, w)
            dst.write(out_chunk, window=Window(0, r0, w, r1 - r0))

            if chunk_valid.any():
                finite_chunk = out_chunk[:, chunk_valid]  # (256, n_valid_chunk)
                running_min = min(running_min, float(np.nanmin(finite_chunk)))
                running_max = max(running_max, float(np.nanmax(finite_chunk)))
                running_sum += float(np.nansum(finite_chunk))
    dt = time.time() - t0

    stats = {
        "tile": tile_path.name, "out_path": str(out_path), "h": h, "w": w,
        "n_pixels": h * w, "n_valid": n_valid, "seconds": dt,
        "peak_array_bytes": peak_pre_project_bytes + peak_chunk_bytes,
    }
    if n_valid > 0:
        stats["min"] = running_min
        stats["max"] = running_max
        stats["mean"] = running_sum / (n_valid * LATENT_DIM)
    return stats


def run_backbone_arm(backbone: str, arm: str, aoi: gpd.GeoDataFrame, device: str = "cpu") -> list:
    print(f"\n=== {backbone} / Arm {arm.upper()} ===")
    head, cfg_used = get_head(backbone, arm)
    if arm == "c":
        check_architecture(backbone, cfg_used)
    mode, stats_path = resolve_normalization(cfg_used, f"{backbone}/{arm}")
    norm_stats = load_stats(stats_path) if mode == ZSCORE else None
    print(f"  Normalisierung: {mode}" + (f" ({stats_path})" if stats_path else ""))

    tile_paths = sj.get_backbones("london")[backbone]
    print(f"  {len(tile_paths)} Rohkacheln (London)")

    out_dir = OUT_ROOT / f"{backbone}_arm{arm}"
    results = []
    t_start = time.time()
    for tp in tile_paths:
        r = project_tile(tp, backbone, head, mode, norm_stats, aoi, out_dir, device=device)
        if r is not None:
            results.append(r)
            print(f"    {r['tile']}: {r['h']}x{r['w']} ({r['n_valid']}/{r['n_pixels']} gueltig), "
                  f"{r['seconds']:.1f}s, min={r.get('min', float('nan')):.4f} "
                  f"max={r.get('max', float('nan')):.4f}")
        else:
            print(f"    {tp.name}: ausserhalb AOI-BBox, uebersprungen")
    total_s = time.time() - t_start

    n_valid_total = sum(r["n_valid"] for r in results)
    print(f"  Fertig: {len(results)} Kacheln geschrieben nach {out_dir}, "
          f"{n_valid_total:,} gueltige Pixel gesamt, {total_s:.1f}s "
          f"({n_valid_total / total_s:,.0f} Pixel/s)" if total_s > 0 else "")
    return results


def sample_l2_norm_check(results: list, n_files: int = 20, window_size: int = 64) -> None:
    """Stichprobencheck (Nutzervorgabe): projizierte Pixelvektoren muessen L2-Norm 1 haben
    (ProjectionHead endet mit F.normalize). Liest NUR ein kleines Fenster pro Datei (nicht die
    ganze Kachel - manche Outputs sind selbst zig GB gross, siehe ROW_CHUNK-Kommentar)."""
    rng = np.random.default_rng(0)
    idx_pool = rng.permutation(len(results))[:n_files] if len(results) > n_files else range(len(results))
    checked = 0
    max_dev = 0.0
    for i in idx_pool:
        r = results[i]
        vecs = np.empty((0, LATENT_DIM), dtype=np.float32)
        with rasterio.open(r["out_path"]) as src:
            h, w = src.height, src.width
            ws = min(window_size, h, w)
            # Retry bei leerem Fenster (Nutzervorgabe-Nachtrag): AlphaEarth-Kacheln haben grosse
            # Nodata-Raender ausserhalb des AOI-Zuschnitts - ein einzelnes Zufallsfenster kann
            # leer sein, das darf die Kachel nicht stillschweigend aus der Stichprobe werfen.
            for _attempt in range(10):
                r0 = int(rng.integers(0, h - ws + 1))
                c0 = int(rng.integers(0, w - ws + 1))
                arr = src.read(window=Window(c0, r0, ws, ws))  # (256, ws, ws)
                valid = np.isfinite(arr).all(axis=0)
                vecs = arr[:, valid].T  # (n_valid, 256)
                if len(vecs) > 0:
                    break
        if len(vecs) == 0:
            continue
        take = min(200, len(vecs))
        sub_idx = rng.choice(len(vecs), size=take, replace=False)
        norms = np.linalg.norm(vecs[sub_idx], axis=1)
        max_dev = max(max_dev, float(np.abs(norms - 1.0).max()))
        checked += take
    print(f"  Stichprobencheck L2-Norm: {checked} Pixel aus {len(list(idx_pool))} Kacheln "
          f"({window_size}x{window_size}-Fenster je Kachel), max. Abweichung von 1.0: {max_dev:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES_LIST), choices=BACKBONES_LIST)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("=== Pre-Flight: norm_stats-Verfuegbarkeit ===")
    for name, info in check_norm_stats_available().items():
        print(f"  {name}: {info}")

    aoi = gpd.read_file(AOI_PATH)

    for backbone in args.backbones:
        for arm in args.arms:
            results = run_backbone_arm(backbone, arm, aoi, device=args.device)
            if results:
                sample_l2_norm_check(results)


if __name__ == "__main__":
    main()
