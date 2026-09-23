"""
maps.py

INSTRUCTIONS_maps_aether.md, Phase 4: PNG-Karten (300 dpi) aus den Similarity-Feldern
(Phase 2) und Referenzkarten (Phase 3), auf einem gemeinsamen Kartengitter.

Gemeinsames Kartengitter NUR fuer die Darstellung (Entscheidung 2026-09-17,
INSTRUCTIONS_text_query.md Phase 3 - wiederverwendete Konvention, dort fuer denselben Zweck
festgelegt): EPSG:32630, 50 m, damit beide Backbones und alle Kacheln in EINEM Bild
darstellbar sind. Kennzahlen (Perzentile, Wertebereiche) werden NICHT auf diesem Gitter
berechnet, sondern auf den nativen Similarity-/Referenz-Kacheln aus Phase 2/3 - das
Kartengitter ist reine Visualisierung.

Ausgabe: outputs/text_query/maps/similarity_{backbone}_{query}.png (6),
outputs/text_query/maps/reference_{backbone}_{reference}.png (4),
outputs/text_query/similarity_value_ranges.csv (Phase-2-Wertebereiche + Phase-4-
Perzentilgrenzen je Query, Nutzervorgabe 2026-09-23).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Repo-Root

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

import config as cfg

SIM_ROOT = cfg.ROOT / "outputs" / "text_query" / "similarity"
REF_ROOT = cfg.ROOT / "outputs" / "text_query" / "reference"
OUT_MAPS = cfg.ROOT / "outputs" / "text_query" / "maps"
OUT_CSV = cfg.ROOT / "outputs" / "text_query" / "similarity_value_ranges.csv"
AOI_PATH = cfg.AOI_DIR / "london.gpkg"

GRID_CRS = "EPSG:32630"
GRID_RES_M = 50.0
FIG_SIZE = (8, 6.5)  # Zoll, gleich fuer ALLE Karten (Nutzervorgabe: gleiche Bildgroesse/Ausschnitt)
DPI = 300

BACKBONES = ("tessera", "ae")
QUERY_NAMES = ("green", "airport", "control", "cbd", "green_park", "green_space",
               "green_reserve", "qual_tourist", "qual_busy", "qual_quiet", "qual_learn")
REFERENCE_NAMES = ("greencover", "airports")  # nur fuer die urspruenglichen 2 AETHER-Referenzen


def build_common_grid() -> tuple:
    aoi = gpd.read_file(AOI_PATH).to_crs(GRID_CRS)
    minx, miny, maxx, maxy = aoi.total_bounds
    width = int(np.ceil((maxx - minx) / GRID_RES_M))
    height = int(np.ceil((maxy - miny) / GRID_RES_M))
    transform = rasterio.transform.from_origin(minx, maxy, GRID_RES_M, GRID_RES_M)
    aoi_mask = rasterize([(aoi.geometry.iloc[0], 1)], out_shape=(height, width),
                          transform=transform, fill=0, dtype="uint8").astype(bool)
    print(f"Gemeinsames Kartengitter: {height}x{width} Pixel, {GRID_RES_M}m, {GRID_CRS}, "
          f"AOI-Pixel: {int(aoi_mask.sum())}")
    return transform, (height, width), aoi, aoi_mask


def mosaic_to_common_grid(tile_paths: list, shape: tuple, transform, crs, resampling,
                           fill) -> np.ndarray:
    """Reprojiziert jede Kachel (natives CRS) einzeln in ein Ziel-Array derselben Groesse wie
    das gemeinsame Gitter, dann werden alle Kachel-Ergebnisse zusammengefuehrt - Kacheln
    ueberlappen sich in ihren GUELTIGEN Bereichen nicht (Eindeutigkeitsmaske aus Phase 2/3),
    "letzter gueltiger Wert gewinnt" ist deshalb unschaedlich."""
    dest = np.full(shape, fill, dtype=np.float32 if fill != 255 else np.uint8)
    for path in tile_paths:
        with rasterio.open(path) as src:
            src_arr = src.read(1)
            src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata
        tmp = np.full(shape, fill, dtype=dest.dtype)
        reproject(source=src_arr, destination=tmp, src_transform=src_transform, src_crs=src_crs,
                  dst_transform=transform, dst_crs=crs, src_nodata=src_nodata, dst_nodata=fill,
                  resampling=resampling)
        if fill == 255:
            valid = tmp != fill
        else:
            valid = np.isfinite(tmp)
        dest[valid] = tmp[valid]
    return dest


def compute_query_stats_and_percentiles(query_name: str) -> tuple:
    """Liest ALLE nativen Similarity-Kacheln (nicht das Kartengitter) beider Backbones fuer
    eine Query - liefert je Backbone dieselben Kennzahlen wie der Phase-2-Bericht (min/max/
    mean/n_valid/n_total/frac_nodata, hier frisch aus den Dateien berechnet statt kopiert) UND
    das 1./99. Perzentil ueber die VEREINIGUNG beider Backbones (Phase-4-Skalengrenzen)."""
    per_backbone = {}
    all_vals = []
    for backbone in BACKBONES:
        vals = []
        n_total = 0
        for p in sorted((SIM_ROOT / backbone / query_name).glob("*.tif")):
            with rasterio.open(p) as src:
                band = src.read(1)
            n_total += band.size
            finite = band[np.isfinite(band)]
            if finite.size:
                vals.append(finite)
        backbone_vals = np.concatenate(vals) if vals else np.array([], dtype=np.float32)
        per_backbone[backbone] = {
            "min": float(backbone_vals.min()) if backbone_vals.size else float("nan"),
            "max": float(backbone_vals.max()) if backbone_vals.size else float("nan"),
            "mean": float(backbone_vals.mean()) if backbone_vals.size else float("nan"),
            "n_valid": int(backbone_vals.size), "n_total": int(n_total),
            "frac_nodata": 1.0 - backbone_vals.size / n_total if n_total else float("nan"),
        }
        all_vals.append(backbone_vals)
    union = np.concatenate(all_vals)
    p1, p99 = np.percentile(union, [1, 99])
    return per_backbone, float(p1), float(p99), int(union.size)


def _extent(transform, shape) -> tuple:
    height, width = shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return (left, right, bottom, top)


def render_similarity_map(backbone: str, query_name: str, vmin: float, vmax: float,
                           transform, shape, aoi_gdf, aoi_mask, out_path: Path) -> None:
    tile_paths = sorted((SIM_ROOT / backbone / query_name).glob("*.tif"))
    grid = mosaic_to_common_grid(tile_paths, shape, transform, GRID_CRS,
                                  Resampling.average, fill=np.nan)
    grid = np.where(aoi_mask, grid, np.nan)  # sauberer AOI-Rand, nicht nur die Kachel-Maske

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=(0, 0, 0, 0))  # NoData/ausserhalb AOI -> transparent

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, extent=_extent(transform, shape),
                    origin="upper")
    aoi_gdf.boundary.plot(ax=ax, color="black", linewidth=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("cosine similarity")
    ax.set_title(f"{backbone} — {query_name}", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)


def render_reference_map(backbone: str, reference_name: str, transform, shape, aoi_gdf,
                          aoi_mask, out_path: Path) -> None:
    tile_paths = sorted((REF_ROOT / backbone / reference_name).glob("*.tif"))
    grid = mosaic_to_common_grid(tile_paths, shape, transform, GRID_CRS,
                                  Resampling.nearest, fill=255)
    grid_f = grid.astype(np.float32)
    grid_f[grid == 255] = np.nan
    grid_f[~aoi_mask] = np.nan

    cmap = ListedColormap(["white", "green"])
    cmap.set_bad(color=(0, 0, 0, 0))

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.imshow(grid_f, cmap=cmap, vmin=0, vmax=1, extent=_extent(transform, shape), origin="upper")
    aoi_gdf.boundary.plot(ax=ax, color="black", linewidth=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{backbone} — {reference_name} (>50% Flächenanteil)", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    transform, shape, aoi_gdf, aoi_mask = build_common_grid()

    csv_rows = []
    limits_by_query = {}
    for query_name in QUERY_NAMES:
        per_backbone, p1, p99, n_union = compute_query_stats_and_percentiles(query_name)
        limits_by_query[query_name] = (p1, p99)
        print(f"\n[{query_name}] Skalengrenzen (1./99. Perzentil, Vereinigung beider "
              f"Backbones, n={n_union}): vmin={p1:.4f} vmax={p99:.4f}")
        for backbone, s in per_backbone.items():
            print(f"  [{backbone}/{query_name}] min={s['min']:.4f} max={s['max']:.4f} "
                  f"mean={s['mean']:.4f} n_valid={s['n_valid']}/{s['n_total']} "
                  f"NoData={100 * s['frac_nodata']:.2f}%")
            csv_rows.append({
                "backbone": backbone, "query": query_name,
                "value_min": s["min"], "value_max": s["max"], "value_mean": s["mean"],
                "n_valid": s["n_valid"], "n_total": s["n_total"], "frac_nodata": s["frac_nodata"],
                "map_vmin_p1_shared": p1, "map_vmax_p99_shared": p99,
                "n_pixels_for_percentile_union": n_union,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    print(f"\nGespeichert: {OUT_CSV}")

    print("\n=== Similarity-Karten ===")
    for backbone in BACKBONES:
        for query_name in QUERY_NAMES:
            vmin, vmax = limits_by_query[query_name]
            out_path = OUT_MAPS / f"similarity_{backbone}_{query_name}.png"
            render_similarity_map(backbone, query_name, vmin, vmax, transform, shape,
                                   aoi_gdf, aoi_mask, out_path)
            print(f"  {out_path}")

    print("\n=== Referenzkarten ===")
    for backbone in BACKBONES:
        for reference_name in REFERENCE_NAMES:
            out_path = OUT_MAPS / f"reference_{backbone}_{reference_name}.png"
            render_reference_map(backbone, reference_name, transform, shape, aoi_gdf, aoi_mask,
                                  out_path)
            print(f"  {out_path}")


if __name__ == "__main__":
    main()
