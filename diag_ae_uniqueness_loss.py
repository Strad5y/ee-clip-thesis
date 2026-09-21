"""
diag_ae_uniqueness_loss.py  (2026-09-21)

Klaerung waehrend AE-Dedup Schritt 1: die Eindeutigkeitsmaske entfernt aus pixels_ae 5.605
Zeilen, aber im Original gibt es nur 1.432 Dopplungs-Weltorte (find_duplicate_rows). Woher die
Differenz? Kategorisiert die durch die Eindeutigkeitsmaske entfernten Zeilen (innerhalb AOI,
primaer nicht OK):
  A) Rivalen-Kachel hat am Weltort gueltige Daten UND eine Zeile im Original  -> echte Dopplung
  B) Rivalen-Kachel hat gueltige Daten, aber KEINE Zeile im Original            -> Zeile "verloren"
     (spatial_join.assign_utm_zone hatte den POI der anderen Zone zugewiesen, die Maske
      spricht den Weltort aber der Rivalen-Kachel zu -> weder Original- noch Rivalen-Zeile
      bleibt uebrig)
  C) kein Rival mit gueltigen Daten am Weltort                                  -> unerklaert
Nur lesen/berichten.
"""
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import config as cfg
import spatial_join as sj
from ae_dedup_utils import BACKBONE, pixel_key, primary_and_aoi_flags
from check_tile_overlap_side_effects import _tile_meta, find_overlap_candidates
from diag_removal_rate_breakdown import build_aoi_only_mask_lookup
from tessera_dedup_utils import build_full_mask_lookup


def main():
    orig = pd.read_parquet(cfg.DATA / "processed" / "pixels_ae.parquet", columns=["tile", "row", "col"])
    tile_names = [p.name for p in sj.get_backbones("london")[BACKBONE]]
    primary_ok, aoi_ok = primary_and_aoi_flags(orig, build_full_mask_lookup(BACKBONE),
                                               build_aoi_only_mask_lookup(BACKBONE))
    uniq = orig[(~primary_ok) & aoi_ok].reset_index(drop=True)
    print(f"Eindeutigkeits-entfernte Zeilen: {len(uniq)}")

    meta = _tile_meta("london", BACKBONE)
    cands = find_overlap_candidates(meta)
    rivals = {t: [b if a == t else a for a, b in cands if t in (a, b)] for t in tile_names}
    orig_keys = set(pixel_key(orig, tile_names).tolist())
    code = {t: i for i, t in enumerate(sorted(tile_names))}

    cat = np.array(["C"] * len(uniq))
    for t, sub in uniq.groupby("tile"):
        idx = sub.index.to_numpy()
        m = meta[t]
        xs, ys = rasterio.transform.xy(m["transform"], sub["row"].to_numpy(), sub["col"].to_numpy())
        xs, ys = np.asarray(xs), np.asarray(ys)
        for rv in rivals[t]:
            mr = meta[rv]
            xb, yb = Transformer.from_crs(str(m["crs"]), str(mr["crs"]), always_xy=True).transform(xs, ys)
            rr, rc = rasterio.transform.rowcol(mr["transform"], xb, yb)
            rr, rc = np.asarray(rr), np.asarray(rc)
            inb = (rr >= 0) & (rr < mr["shape"][0]) & (rc >= 0) & (rc < mr["shape"][1])
            with rasterio.open(mr["path"]) as src:
                b1 = src.read(1)
                nod = src.nodata
            valid = np.zeros(len(idx), dtype=bool)
            valid[inb] = (b1[rr[inb], rc[inb]] != nod) if nod is not None else True
            has_row = np.zeros(len(idx), dtype=bool)
            keys = (code[rv] << 30) | (rr.astype(np.int64) << 15) | rc.astype(np.int64)
            has_row[inb] = np.array([k in orig_keys for k in keys[inb]])
            cur = cat[idx]
            cat[idx] = np.where(valid & has_row, "A", np.where(valid & (cur != "A"), "B", cur))
    for k, name in (("A", "echte Dopplung (Rival hat Zeile im Original)"),
                    ("B", "Rival gueltig, aber KEINE Zeile im Original (Zeile verloren)"),
                    ("C", "kein Rival mit gueltigen Daten (unerklaert)")):
        print(f"  {k}) {name}: {int((cat == k).sum())}")
    print(f"  (nodata der Kacheln: {[rasterio.open(meta[t]['path']).nodata for t in tile_names]})")


if __name__ == "__main__":
    main()
