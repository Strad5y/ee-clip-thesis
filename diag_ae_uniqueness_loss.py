"""
diag_ae_uniqueness_loss.py  (2026-09-21; auf TESSERA erweitert, Argument --backbone)

Klaerung waehrend AE-Dedup Schritt 1: die Eindeutigkeitsmaske entfernt aus pixels_<bb> mehr
Zeilen, als es Dopplungs-Weltorte im Original gibt (find_duplicate_rows). Kategorisiert die
durch die Eindeutigkeitsmaske entfernten Zeilen (innerhalb AOI, primaer nicht OK):
  A) Rivalen-Kachel hat am Weltort gueltige Daten UND eine Zeile im Original -> echte Dopplung
  B) Rivalen-Kachel hat gueltige Daten, aber KEINE Zeile im Original          -> Ort verschwindet
     (spatial_join.assign_utm_zone hat den POI nur der anderen UTM-Zone zugewiesen; die Maske
      spricht den Weltort aber der Rivalen-Kachel zu -> weder Original- noch Rivalen-Zeile
      bleibt uebrig). B wird nach Zonen aufgeteilt: Rivale in ANDERER CRS (= Zonenzuweisung
      als Ursache) vs. GLEICHER CRS (dort gibt es keinen Zonenfilter -> waere unerwartet).
  C) kein Rival mit gueltigen Daten am Weltort                                -> unerklaert
Exakte Gegenprobe gegen das Dedup-Ergebnis (pixels_<bb>_dedup): ueberlebt der Ort in irgendeiner
Rivalen-Kachel? ('Ort verschwindet' = kein Rivalen-Pixel im Dedup-Ergebnis).
B-Zusatzpruefung: hat die 3x3-Nachbarschaft des Rivalen-Pixels eine Zeile im Original? (Gitter-
phasen-Hypothese: die Grids der Kacheln sind nicht phasengleich, ein POI-Polygon kann das Pixel
der einen Kachel abdecken, das naechstliegende Pixel der anderen aber nicht.)
Gueltig (Rivale): AE: != nodata(-128); TESSERA (kein nodata): np.isfinite(Band 1) - wie die
Gueltigkeit der Arm-C-Felder/Masken. Nur lesen/berichten.
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import config as cfg
import spatial_join as sj
from ae_dedup_utils import pixel_key, primary_and_aoi_flags
from check_tile_overlap_side_effects import _tile_meta, find_overlap_candidates
from diag_removal_rate_breakdown import build_aoi_only_mask_lookup
from tessera_dedup_utils import build_full_mask_lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ae", choices=["ae", "tessera"])
    bb = ap.parse_args().backbone

    orig = pd.read_parquet(cfg.DATA / "processed" / f"pixels_{bb}.parquet", columns=["tile", "row", "col"])
    final = pd.read_parquet(cfg.DATA / "processed" / f"pixels_{bb}_dedup.parquet", columns=["tile", "row", "col"])
    tile_names = [p.name for p in sj.get_backbones("london")[bb]]
    primary_ok, aoi_ok = primary_and_aoi_flags(orig, build_full_mask_lookup(bb), build_aoi_only_mask_lookup(bb))
    uniq = orig[(~primary_ok) & aoi_ok].reset_index(drop=True)
    print(f"[{bb}] Zeilen pixels_{bb}: {len(orig)}, Eindeutigkeits-entfernt (innerhalb AOI): {len(uniq)}")

    meta = _tile_meta("london", bb)
    cands = find_overlap_candidates(meta)
    rivals = {t: [b if a == t else a for a, b in cands if t in (a, b)] for t in tile_names}
    orig_keys = set(pixel_key(orig, tile_names).tolist())
    final_keys = set(pixel_key(final, tile_names).tolist())
    code = {t: i for i, t in enumerate(sorted(tile_names))}

    n = len(uniq)
    valid_any = np.zeros(n, dtype=bool)      # irgendein Rival mit gueltigen Daten
    row_any = np.zeros(n, dtype=bool)        # irgendein Rival mit Zeile im Original
    final_any = np.zeros(n, dtype=bool)      # irgendein Rival-Pixel im Dedup-Ergebnis
    b_cross = np.zeros(n, dtype=bool)        # gueltiger Rival ohne Zeile ist in anderer CRS
    b_same = np.zeros(n, dtype=bool)
    nb_any = np.zeros(n, dtype=bool)         # 3x3-Nachbarschaft des Rivalen-Pixels hat eine Zeile im Original
    band_cache = {}
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
            if rv not in band_cache:
                with rasterio.open(mr["path"]) as src:
                    b1 = np.array(src.read(1))
                    nod = src.nodata
                band_cache[rv] = (b1 != nod) if nod is not None else np.isfinite(b1)
            valid = np.zeros(len(idx), dtype=bool)
            valid[inb] = band_cache[rv][rr[inb], rc[inb]]
            keys = (code[rv] << 30) | (rr.astype(np.int64) << 15) | rc.astype(np.int64)
            has_row = np.zeros(len(idx), dtype=bool)
            in_final = np.zeros(len(idx), dtype=bool)
            has_row[inb] = np.fromiter((k in orig_keys for k in keys[inb]), bool, int(inb.sum()))
            in_final[inb] = np.fromiter((k in final_keys for k in keys[inb]), bool, int(inb.sum()))
            valid_any[idx] |= valid
            row_any[idx] |= has_row
            final_any[idx] |= in_final
            no_row_valid = valid & ~has_row
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    sel = np.where(no_row_valid & inb)[0]
                    nk = (code[rv] << 30) | ((rr[sel] + dr).astype(np.int64) << 15) | (rc[sel] + dc).astype(np.int64)
                    nb_any[idx[sel]] |= np.fromiter((k in orig_keys for k in nk), bool, len(sel))
            if str(m["crs"]) != str(mr["crs"]):
                b_cross[idx] |= no_row_valid
            else:
                b_same[idx] |= no_row_valid

    A = row_any
    B = valid_any & ~row_any
    C = ~valid_any & ~row_any
    Bc = B & b_cross
    Bs = B & ~b_cross
    lost = ~final_any
    res = {
        "backbone": bb, "n_pixels_orig": len(orig), "n_uniqueness_removed": n,
        "A_true_duplicate": int(A.sum()), "B_location_vanishes": int(B.sum()),
        "B_rival_other_crs(zone_assignment)": int(Bc.sum()), "B_rival_same_crs": int(Bs.sum()),
        "B_with_row_in_3x3_neighbourhood_of_rival_pixel": int((B & nb_any).sum()),
        "B_rival_other_crs_without_neighbour_row": int((Bc & ~nb_any).sum()),
        "B_rival_same_crs_without_neighbour_row": int((Bs & ~nb_any).sum()),
        "C_no_valid_rival": int(C.sum()),
        "check_location_absent_in_dedup_result": int(lost.sum()),
        "check_A_but_absent_in_dedup_result": int((A & lost).sum()),
        "share_of_all_orig_pixels_B": B.sum() / len(orig),
    }
    for k, v in res.items():
        print(f"  {k}: {v}")
    out = cfg.DATA / "processed" / f"diag_uniqueness_loss_{bb}.csv"
    pd.DataFrame([res]).to_csv(out, index=False)
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
