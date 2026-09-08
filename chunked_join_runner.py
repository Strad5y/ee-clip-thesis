"""
chunked_join_runner.py

Gemeinsame Infrastruktur fuer robuste Vollstaendige Joins (AlphaEarth, TESSERA) - urspruenglich
in run_ae_join.py, hier herausgezogen, damit run_tessera_join.py dieselbe getestete Logik
wiederverwendet statt sie zu duplizieren. Join-Logik selbst (spatial_join.py) wird hier NICHT
veraendert - dieses Modul ruft sie nur blockweise auf:

  - Bloecke (chunk_size POIs, sortiert nach poi_id -> deterministisch, resume-stabil), jeder
    Block wird SOFORT als eigene Parquet-Datei gespeichert.
  - Bereits vorhandene Bloecke werden beim Start uebersprungen (Resume nach Abbruch).
  - Fortschrittsausgabe mindestens alle 60s (eigener Heartbeat-Thread, unabhaengig davon, wie
    lange ein einzelner Block tatsaechlich braucht) - Konsole UND Logdatei unter logs/.
  - n_pixels_poi wird ERST NACH dem Merge aller Bloecke neu berechnet, nie pro Block
    uebernommen - unabhaengig davon, ob Bloecke nach poi_id gebildet wurden.
"""
import sys
import threading
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

import config as cfg
from spatial_join import DEFAULT_ZONES, ID_COL, POIS_IN, join_pois_to_tiles

LOG_DIR = cfg.ROOT / "logs"
HEARTBEAT_INTERVAL_S = 60


class _Tee:
    """Spiegelt sys.stdout zusaetzlich in eine Logdatei - faengt auch die print()-Ausgaben
    aus spatial_join.py (z.B. 'Points: N nicht in dieser Kachel') mit ein, ohne dass dort
    etwas geaendert werden muss."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class ProgressState:
    def __init__(self, total_pois: int, n_blocks: int):
        self.lock = threading.Lock()
        self.total_pois = total_pois
        self.n_blocks = n_blocks
        self.n_pois_done = 0
        self.n_pairs_done = 0
        self.current_block = 0
        self.t_start = time.time()

    def update(self, n_pois_done, n_pairs_done, current_block):
        with self.lock:
            self.n_pois_done = n_pois_done
            self.n_pairs_done = n_pairs_done
            self.current_block = current_block

    def snapshot(self):
        with self.lock:
            return dict(n_pois_done=self.n_pois_done, n_pairs_done=self.n_pairs_done,
                        current_block=self.current_block, n_blocks=self.n_blocks,
                        total_pois=self.total_pois, t_start=self.t_start)


def _format_progress(s: dict) -> str:
    elapsed = time.time() - s["t_start"]
    rate = s["n_pois_done"] / elapsed if elapsed > 0 else 0.0
    remaining_pois = s["total_pois"] - s["n_pois_done"]
    remaining_s = remaining_pois / rate if rate > 0 else float("nan")
    return (f"[Fortschritt] Block {s['current_block']}/{s['n_blocks']}: "
            f"{s['n_pois_done']}/{s['total_pois']} POIs, {s['n_pairs_done']} Paare bisher, "
            f"verstrichen={elapsed / 60:.1f} min, geschaetzte Restzeit={remaining_s / 60:.1f} min")


def _heartbeat_loop(state: ProgressState, stop_event: threading.Event):
    while not stop_event.wait(HEARTBEAT_INTERVAL_S):
        print(_format_progress(state.snapshot()))


def make_chunks(gdf: gpd.GeoDataFrame, chunk_size: int) -> list:
    gdf = gdf.sort_values(ID_COL).reset_index(drop=True)  # deterministisch -> resume-stabil
    n = len(gdf)
    return [gdf.iloc[i:i + chunk_size].reset_index(drop=True) for i in range(0, n, chunk_size)]


def run(label: str, tile_paths, dequantize, chunk_dir: Path, out_path: Path,
        chunk_size: int = 20000, pois_in: Path = POIS_IN, zones=DEFAULT_ZONES) -> pd.DataFrame:
    """Fuehrt den vollstaendigen blockweisen Join durch und schreibt out_path.
    label: nur fuer Logdatei-Namen/Ausgaben (z.B. 'ae', 'tessera')."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}_join_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    print(f"=== {label}-Join gestartet ===")
    print(f"Log-Datei: {log_path}")
    print(f"Chunk-Groesse: {chunk_size}")
    print(f"Kacheln: {len(tile_paths)}")

    print(f"Lese POIs: {pois_in}")
    gdf = gpd.read_parquet(pois_in)
    n_raw = len(gdf)
    gdf = gdf.drop_duplicates(subset=ID_COL).reset_index(drop=True)
    print(f"  {n_raw} Tag-Zeilen -> {len(gdf)} eindeutige POI-Geometrien")

    chunks = make_chunks(gdf, chunk_size)
    n_blocks = len(chunks)
    print(f"{n_blocks} Bloecke a maximal {chunk_size} POIs")

    chunk_dir.mkdir(parents=True, exist_ok=True)

    state = ProgressState(total_pois=len(gdf), n_blocks=n_blocks)
    stop_event = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(state, stop_event), daemon=True)
    heartbeat.start()

    n_pois_done = 0
    n_pairs_done = 0

    for i, chunk in enumerate(chunks):
        chunk_path = chunk_dir / f"chunk_{i:04d}.parquet"
        if chunk_path.exists():
            existing = pd.read_parquet(chunk_path, columns=[ID_COL])
            n_pois_done += len(chunk)
            n_pairs_done += len(existing)
            state.update(n_pois_done, n_pairs_done, i + 1)
            print(f"Block {i + 1}/{n_blocks}: bereits vorhanden ({chunk_path.name}), uebersprungen")
            continue

        print(f"Block {i + 1}/{n_blocks}: {len(chunk)} POIs, starte Join...")
        pairs = join_pois_to_tiles(chunk, tile_paths, dequantize=dequantize, zones=zones)
        pairs.to_parquet(chunk_path)

        n_pois_done += len(chunk)
        n_pairs_done += len(pairs)
        state.update(n_pois_done, n_pairs_done, i + 1)
        print(f"Block {i + 1}/{n_blocks} fertig: {len(pairs)} Paare, gespeichert unter {chunk_path}")
        print(_format_progress(state.snapshot()))

    stop_event.set()

    print("\nMerge aller Bloecke...")
    chunk_paths = sorted(chunk_dir.glob("chunk_*.parquet"))
    print(f"  {len(chunk_paths)} Block-Dateien gefunden")
    parts = [pd.read_parquet(p) for p in chunk_paths]
    pairs_all = pd.concat(parts, ignore_index=True)

    # n_pixels_poi NACH dem Merge neu berechnen, nie den Block-lokalen Wert uebernehmen -
    # unabhaengig davon, dass Bloecke hier nach poi_id gebildet wurden.
    if "n_pixels_poi" in pairs_all.columns:
        pairs_all = pairs_all.drop(columns=["n_pixels_poi"])
    pairs_all["n_pixels_poi"] = pairs_all.groupby(ID_COL)[ID_COL].transform("size")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_all.to_parquet(out_path)
    print(f"Gespeichert: {out_path}")
    print(f"  Zeilen: {len(pairs_all)}")
    print(f"  Eindeutige poi_id: {pairs_all[ID_COL].nunique()}")
    print(f"=== {label}-Join fertig ===")
    return pairs_all
