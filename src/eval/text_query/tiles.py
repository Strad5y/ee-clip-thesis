"""
tiles.py

Phase 1 (INSTRUCTIONS_text_query.md, Entscheidung 2026-09-17): stellt die bereits
bestehenden Arm-C-256d-Projektionsfelder (build_sdm_projected_field.py, trainierter Kopf,
data/interim/sdm_projected/{backbone}_armc/) pro Backbone als einheitliche Kachel-Liste
bereit - KEIN Neurechnen, KEINE Reprojektion hier. Jede Kachel bleibt in ihrem nativen
CRS; eine gemeinsame Reprojektion fuer Karten passiert erst spaeter in maps.py (Phase 3),
getrennt von diesem Loader.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Repo-Root fuer config/spatial_join

import rasterio
from rasterio.crs import CRS

import config as cfg
import spatial_join as sj

PROJ_ROOT = cfg.DATA / "interim" / "sdm_projected"
ARM = "c"  # trainierter Kopf - settled (Phase 1, 2026-09-17: nur Arm C wird fuer Text-Query genutzt)


@dataclass
class TilePair:
    backbone: str
    tile_name: str    # Dateiname der Rohkachel (Schluessel, den build_sdm_projected_field.py verwendet hat)
    raw_path: Path     # Rohkachel, exakt wie sie projiziert wurde (sj.get_backbones())
    proj_path: Path    # zugehoeriges *_proj256.tif (Arm C, 256 Kanaele, float32, nodata=nan)
    crs: CRS


def list_tile_pairs(backbone: str, region: str = "london") -> list[TilePair]:
    """Ordnet jede Rohkachel (sj.get_backbones()) ihrem Arm-C-Projektionsfeld zu - ueber den
    Dateinamen-Stamm (project_tile() in build_sdm_projected_field.py haengt immer genau
    '_proj256.tif' an raw_path.stem an). Kacheln ohne Arm-C-Feld (z.B. weil sie ausserhalb
    der AOI-BBox lagen und in build_sdm_projected_field.py uebersprungen wurden) werden
    gemeldet, nicht stillschweigend uebergangen."""
    proj_dir = PROJ_ROOT / f"{backbone}_arm{ARM}"
    raw_tiles = sj.get_backbones(region)[backbone]

    pairs = []
    missing = []
    for raw_path in raw_tiles:
        proj_path = proj_dir / f"{raw_path.stem}_proj256.tif"
        if not proj_path.exists():
            missing.append(raw_path.name)
            continue
        with rasterio.open(raw_path) as src:
            crs = src.crs
        pairs.append(TilePair(backbone=backbone, tile_name=raw_path.name,
                               raw_path=raw_path, proj_path=proj_path, crs=crs))
    if missing:
        print(f"  {backbone}: {len(missing)}/{len(raw_tiles)} Rohkachel(n) ohne Arm-C-Feld "
              f"(vermutlich ausserhalb AOI-BBox, siehe build_sdm_projected_field.py): {missing}")
    return pairs


def _smoke_test():
    for backbone in ("tessera", "ae"):
        pairs = list_tile_pairs(backbone)
        assert len(pairs) > 0, f"{backbone}: keine Kachel-Paare gefunden"
        for p in pairs:
            assert p.raw_path.exists() and p.proj_path.exists()
            assert p.crs is not None
        print(f"{backbone}: {len(pairs)} Kachel-Paare (Roh <-> Arm-C), "
              f"CRS-Beispiel={pairs[0].crs}")
    print("Smoke-Test OK: list_tile_pairs() liefert fuer beide Backbones vollstaendige, "
          "existierende Roh<->Arm-C-Paare mit nativem CRS.")


if __name__ == "__main__":
    _smoke_test()
