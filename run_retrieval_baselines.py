"""
run_retrieval_baselines.py

Phase B (Baselines-und-Evaluation.md): Retrieval-Baselines fuer den AE-v1-Checkpoint
(checkpoints/ae_v1/best.pt), auf Val UND Test. Kein Training - laedt nur best.pt und wertet aus.

Ergaenzt gegenueber train.py::evaluate():
  - Test-Evaluation (train.py verwirft test_idx bisher als _test_idx - hier nachgeholt, gleiche
    Metrik-Funktionen wie im Training, nur zusaetzlich auf dem Test-Split ausgefuehrt).
  - Zufallsniveau class_recall@10 (Key- UND Value-Ebene): Kandidatenpool = derselbe In-Batch-Pool
    wie beim gemessenen Recall (train.py verwendet In-Batch-Retrieval, sim ist (n,n) pro Batch,
    n = batch_size) - Monte-Carlo (R=100 Ziehungen je Query), aber als R GANZE Durchlaeufe ueber
    den kompletten Split akkumuliert (nicht 100 Ziehungen PRO Query einzeln gemittelt), damit
    Mittelwert+Std ueber die 100 Wiederholungen konsistent zum Bootstrap-Stil aus
    run_landuse_probe.py berichtet werden koennen (Std = Streuung des GESAMT-Recall ueber die
    100 Wiederholungen, nicht Streuung zwischen einzelnen Pixeln).
  - Zufallsniveau recall@10_diag: geschlossene Form 10/n (nur EIN spezifisches Ziel pro Query,
    kein Simulationsbedarf - anders als beim klassenbezogenen Recall, wo mehrere Kandidaten im
    Pool dieselbe Klasse teilen koennen).
  - Haeufigkeitsniveau class_recall@10 (Key- UND Value-Ebene): feste Menge = die 10 global
    haeufigsten Klassen IM JEWEILIGEN SPLIT (Val bzw. Test getrennt) - klassifikationsartige
    Regel, unabhaengig vom Batch/Kandidatenpool.
  - Value-Ebene (key=value statt nur key): neue Funktion extract_primary_class() (siehe unten),
    NICHT in dataset.py/train.py eingebaut (dortiger primary_key/class_recall_at_k bleibt
    unveraendert, um den bestehenden Trainingspfad nicht anzufassen) - wiederverwendet aber
    train.py::class_recall_at_k() (ist bereits generisch auf beliebige String-Labels, nicht
    hart auf 'primary_key' verdrahtet).
  - Verteilung der primary_key/primary_value-Werte im Testset + Anteil leerer primary_key
    (bzw. zusaetzlich leerer primary_value bei vorhandenem primary_key).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train as T
from chunked_join_runner import LOG_DIR, _Tee
from dataset import PairDataset, extract_primary_key
from model import TwoTower
from normalize import NONE, VALID_MODES, ZSCORE, apply_zscore, load_stats
from poi_classes import extract_primary_class

TEXT_EMB_PATH = "data/processed/pixel_text_embeddings.parquet"
OUT_DIR = Path("data/processed")

BATCH_SIZE = 256          # wie train.py (cfg["train"]["batch_size"] in ae_v1.yaml)
EVAL_SEED = 0              # Batch-Zusammensetzung (Permutation der Split-Indizes)
N_RANDOM_TRIALS = 100      # Nutzervorgabe: mindestens 100 Ziehungen
RANDOM_SEED_BASE = 1000
FREQ_TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _smoke_test_extract_primary_class() -> None:
    assert extract_primary_class("A building: house. building usually intended for living in") \
        == ("building", "house")
    assert extract_primary_class("Greggs — a public amenity or service: fast food.") \
        == ("amenity", "fast food")
    assert extract_primary_class("A pitch leisure spot and a soccer sports facility.") \
        == ("leisure", "pitch")
    print("Smoke-Test OK: extract_primary_class() liefert Key UND Value fuer Single- und "
          "Multi-Tag-Form korrekt.")


def load_dataset_with_values(pixels_path: str) -> PairDataset:
    """Laedt PairDataset (identischer Pfad wie train.py/dataset.py) UND haengt zusaetzlich
    primary_value (Value-Ebene, siehe extract_primary_class) an - unabhaengig von der Datei
    zweites Mal ueber 'pixel_text' aus derselben Pixel-Parquet-Datei gelesen, dann via
    Alignment-Check gegen ds.primary_key abgesichert (schliesst Reihenfolge-Drift beim Join in
    dataset.py::PairDataset.__init__ aus, statt sie stillschweigend anzunehmen)."""
    ds = PairDataset(pixels_path, TEXT_EMB_PATH)

    raw = pd.read_parquet(pixels_path, columns=["pixel_text"])
    if len(raw) != len(ds):
        raise ValueError(f"Laengen-Mismatch: pixels_path hat {len(raw)} Zeilen, ds hat {len(ds)}.")

    unique_texts = raw["pixel_text"].drop_duplicates().reset_index(drop=True)
    key_check = unique_texts.map(lambda t: extract_primary_key(t) or "")
    class_map = unique_texts.map(extract_primary_class)
    value_by_text = dict(zip(unique_texts, class_map.map(
        lambda kv: f"{kv[0]}={kv[1]}" if kv[0] and kv[1] else "")))
    key_by_text = dict(zip(unique_texts, key_check))

    primary_key_check = raw["pixel_text"].map(key_by_text).to_numpy()
    n_mismatch = int((primary_key_check != ds.primary_key).sum())
    if n_mismatch > 0:
        raise ValueError(
            f"Alignment-Check fehlgeschlagen: {n_mismatch} Zeilen, bei denen primary_key aus "
            f"unabhaengig gelesenem 'pixel_text' nicht zu ds.primary_key passt - Reihenfolge "
            f"zwischen PairDataset-Merge und direktem Parquet-Read weicht ab."
        )
    ds.primary_value = raw["pixel_text"].map(value_by_text).to_numpy()
    print(f"Alignment-Check OK: primary_key aus unabhaengigem Read stimmt fuer alle "
          f"{len(ds)} Zeilen mit ds.primary_key ueberein.")
    return ds


def load_model(ckpt_path: Path) -> tuple[TwoTower, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = T.build_model(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Modell geladen: {ckpt_path} (Epoche {ckpt['epoch']}, cfg backbone={cfg.get('backbone')})")
    return model, cfg


def apply_checkpoint_normalization(ds: PairDataset, cfg: dict) -> None:
    """Normalisiert ds.ee IN-PLACE gemaess der im Checkpoint hinterlegten Config - die
    Entscheidung (zscore vs. none) kommt bewusst aus ckpt['cfg'], nicht aus einem eigenen
    Skript-Argument (Nutzervorgabe T4: sonst kann derselbe Fehler - Eval ohne Normalisierung
    bei einem zscore-Backbone - erneut passieren, unabhaengig davon, welcher Checkpoint
    tatsaechlich geladen wird). Bricht bei fehlender Statistikdatei explizit ab, rechnet nicht
    stillschweigend mit rohen Embeddings weiter."""
    mode = cfg.get("normalization", NONE)
    if mode not in VALID_MODES:
        raise ValueError(f"Unbekannter normalization-Modus in Checkpoint-Config: {mode!r} "
                          f"(erwartet {VALID_MODES})")
    if mode == NONE:
        print("Normalisierung: none (No-Op, wie in Checkpoint-Config 'normalization')")
        return

    stats_path = cfg.get("data", {}).get("norm_stats_path")
    if not stats_path or not Path(stats_path).exists():
        raise FileNotFoundError(
            f"Checkpoint-Config verlangt normalization='{mode}', aber die Statistikdatei "
            f"({stats_path!r}) existiert nicht - Abbruch (Nutzervorgabe T4: kein "
            f"stillschweigendes Weiterrechnen mit rohen, unnormalisierten Embeddings)."
        )
    stats = load_stats(stats_path)
    ds.ee = apply_zscore(ds.ee, stats)
    print(f"Normalisierung: {mode}, Statistik geladen aus {stats_path}")


def batches(idx: np.ndarray, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    for i in range(0, len(perm), batch_size):
        yield perm[i:i + batch_size]


def _new_bucket_acc(n_trials: int) -> dict:
    return {
        "n": 0,
        "hit1_diag": 0.0, "hit10_diag": 0.0, "ceil1": 0.0, "ceil10": 0.0,
        "class10_key_sum": 0.0, "class10_key_n": 0,
        "class10_val_sum": 0.0, "class10_val_n": 0,
        "rand10_diag_sum": 0.0,
        "rand_class10_key_trials": np.zeros(n_trials), "rand_class10_key_trials_n": np.zeros(n_trials),
        "rand_class10_val_trials": np.zeros(n_trials), "rand_class10_val_trials_n": np.zeros(n_trials),
        "freq_class10_key_sum": 0.0, "freq_class10_key_n": 0,
        "freq_class10_val_sum": 0.0, "freq_class10_val_n": 0,
    }


@torch.no_grad()
def evaluate_full(model: TwoTower, ds: PairDataset, idx: np.ndarray, top_key: set, top_val: set,
                   n_trials: int = N_RANDOM_TRIALS, batch_size: int = BATCH_SIZE,
                   seed: int = EVAL_SEED) -> dict:
    """Ein Durchlauf ueber den gesamten Split (fixe Zufalls-Permutation fuer die
    Batch-Zusammensetzung, In-Batch-Kandidatenpool wie beim Training). Je Batch: gemessener
    Recall (Modell), Ceiling/Ausschoepfung, klassenbezogener Recall (Key+Value), Zufallsniveau
    (Ziehungs-Simulation, vektorisiert ueber eine (n_trials,n)-Draw-Maske) und Haeufigkeitsniveau
    (feste Top-10-Mengen aus dem Split selbst). Getrennt aufsummiert fuer 'all'/'single'/'multi'
    (n_pois==1 vs. >1) - gleiche Buckets wie train.py::evaluate()."""
    acc = {"all": _new_bucket_acc(n_trials), "single": _new_bucket_acc(n_trials),
           "multi": _new_bucket_acc(n_trials)}

    for b, batch_idx in enumerate(batches(idx, batch_size, seed)):
        n = len(batch_idx)
        ee = torch.from_numpy(ds.ee[batch_idx]).to(DEVICE)
        text = torch.from_numpy(ds.text[batch_idx]).to(DEVICE)
        text_id = torch.from_numpy(ds.text_id[batch_idx]).to(DEVICE)
        ee_z, text_z = model(ee, text)

        diag_mask = torch.eye(n, device=DEVICE, dtype=torch.bool)
        hit1_diag = T.recall_hits(ee_z, text_z, diag_mask, k=1).cpu().numpy()
        hit10_diag = T.recall_hits(ee_z, text_z, diag_mask, k=10).cpu().numpy()
        ceil1 = T.recall_ceiling(text_id, k=1).cpu().numpy()
        ceil10 = T.recall_ceiling(text_id, k=10).cpu().numpy()

        primary_key = ds.primary_key[batch_idx]
        primary_val = ds.primary_value[batch_idx]
        class10_key = T.class_recall_at_k(ee_z, text_z, primary_key, k=10)
        class10_val = T.class_recall_at_k(ee_z, text_z, primary_val, k=10)

        rand10_diag = np.clip(10.0 / n, 0.0, 1.0) * np.ones(n)

        # Zufallsniveau, klassenbezogen: (n_trials, n) Draw-Maske - pro Ziehung 10 verschiedene
        # der n Kandidaten (unabhaengig von der Modell-Similaritaet), vektorisiert per
        # Rang-Trick (argsort auf Zufallszahlen je Ziehung).
        rng_r = np.random.default_rng(RANDOM_SEED_BASE + b)
        rand_vals = rng_r.random((n_trials, n))
        draw_rank = np.argsort(rand_vals, axis=1)
        draw_mask = np.zeros((n_trials, n), dtype=np.float32)
        rows = np.repeat(np.arange(n_trials), min(10, n))
        cols = draw_rank[:, :10].reshape(-1)
        draw_mask[rows, cols] = 1.0

        def rand_class_hits(labels: np.ndarray) -> np.ndarray:
            valid = labels != ""
            match = (labels[:, None] == labels[None, :]) & valid[None, :] & valid[:, None]
            counts = draw_mask @ match.T.astype(np.float32)  # (n_trials, n)
            hits = (counts > 0).astype(np.float32)
            hits[:, ~valid] = np.nan
            return hits  # (n_trials, n)

        rand_hits_key = rand_class_hits(primary_key)
        rand_hits_val = rand_class_hits(primary_val)

        freq_hit_key = np.array([k in top_key if k != "" else np.nan for k in primary_key])
        freq_hit_val = np.array([v in top_val if v != "" else np.nan for v in primary_val])

        n_pois = ds.n_pois[batch_idx]
        for name, m in (("all", np.ones(n, dtype=bool)), ("single", n_pois == 1), ("multi", n_pois > 1)):
            cnt = int(m.sum())
            if cnt == 0:
                continue
            a = acc[name]
            a["n"] += cnt
            a["hit1_diag"] += hit1_diag[m].sum()
            a["hit10_diag"] += hit10_diag[m].sum()
            a["ceil1"] += ceil1[m].sum()
            a["ceil10"] += ceil10[m].sum()
            a["rand10_diag_sum"] += rand10_diag[m].sum()

            v = ~np.isnan(class10_key[m])
            a["class10_key_sum"] += class10_key[m][v].sum(); a["class10_key_n"] += int(v.sum())
            v = ~np.isnan(class10_val[m])
            a["class10_val_sum"] += class10_val[m][v].sum(); a["class10_val_n"] += int(v.sum())

            rk = rand_hits_key[:, m]
            valid_k = ~np.isnan(rk)
            a["rand_class10_key_trials"] += np.nansum(rk, axis=1)
            a["rand_class10_key_trials_n"] += valid_k.sum(axis=1)
            rv = rand_hits_val[:, m]
            valid_v = ~np.isnan(rv)
            a["rand_class10_val_trials"] += np.nansum(rv, axis=1)
            a["rand_class10_val_trials_n"] += valid_v.sum(axis=1)

            fk = freq_hit_key[m]; v = ~np.isnan(fk)
            a["freq_class10_key_sum"] += np.nansum(fk); a["freq_class10_key_n"] += int(v.sum())
            fv = freq_hit_val[m]; v = ~np.isnan(fv)
            a["freq_class10_val_sum"] += np.nansum(fv); a["freq_class10_val_n"] += int(v.sum())

    return acc


def summarize(acc: dict) -> dict:
    out = {}
    for name, a in acc.items():
        if a["n"] == 0:
            out[name] = None
            continue
        r1, r10 = a["hit1_diag"] / a["n"], a["hit10_diag"] / a["n"]
        c1, c10 = a["ceil1"] / a["n"], a["ceil10"] / a["n"]
        rand_trials_key = a["rand_class10_key_trials"] / np.maximum(a["rand_class10_key_trials_n"], 1)
        rand_trials_val = a["rand_class10_val_trials"] / np.maximum(a["rand_class10_val_trials_n"], 1)
        out[name] = {
            "n": a["n"],
            "recall@1_diag": r1, "recall@10_diag": r10,
            "ceiling@1_diag": c1, "ceiling@10_diag": c10,
            "exploitation@1_diag": r1 / c1 if c1 > 0 else float("nan"),
            "exploitation@10_diag": r10 / c10 if c10 > 0 else float("nan"),
            "class_recall@10_key": a["class10_key_sum"] / a["class10_key_n"] if a["class10_key_n"] else float("nan"),
            "class_recall@10_value": a["class10_val_sum"] / a["class10_val_n"] if a["class10_val_n"] else float("nan"),
            "random_recall@10_diag": a["rand10_diag_sum"] / a["n"],
            "random_class_recall@10_key_mean": float(np.mean(rand_trials_key)),
            "random_class_recall@10_key_std": float(np.std(rand_trials_key)),
            "random_class_recall@10_value_mean": float(np.mean(rand_trials_val)),
            "random_class_recall@10_value_std": float(np.std(rand_trials_val)),
            "freq_class_recall@10_key": a["freq_class10_key_sum"] / a["freq_class10_key_n"] if a["freq_class10_key_n"] else float("nan"),
            "freq_class_recall@10_value": a["freq_class10_val_sum"] / a["freq_class10_val_n"] if a["freq_class10_val_n"] else float("nan"),
        }
    return out


def top_k_set(labels: np.ndarray, k: int = FREQ_TOP_K) -> set:
    vc = pd.Series(labels[labels != ""]).value_counts()
    return set(vc.head(k).index)


def report_distribution(name: str, labels: np.ndarray, top_n: int = 15) -> None:
    n_total = len(labels)
    n_empty = int((labels == "").sum())
    vc = pd.Series(labels[labels != ""]).value_counts()
    print(f"\n{name}: {n_total} Zeilen, {n_empty} ({n_empty / n_total:.4%}) mit leerem Label "
          f"(aus Mittel ausgeschlossen), {vc.size} unterschiedliche Klassen unter den restlichen "
          f"{n_total - n_empty}.")
    print(f"  Top-{top_n} (Anteil an nicht-leeren Zeilen):")
    share = vc / vc.sum()
    for cls, cnt in vc.head(top_n).items():
        print(f"    {cls:30s} n={cnt:>9d}  {share[cls]:.4%}")


def print_table(summ: dict, split_name: str) -> None:
    print(f"\n=== {split_name} ===")
    for bucket in ("all", "single", "multi"):
        s = summ[bucket]
        if s is None:
            print(f"  [{bucket}] keine Zeilen")
            continue
        print(f"\n  [{bucket}] n={s['n']}")
        print(f"    Modell        recall@1_diag={s['recall@1_diag']:.4f}  recall@10_diag={s['recall@10_diag']:.4f}")
        print(f"    Ceiling       ceil@1_diag={s['ceiling@1_diag']:.4f}  ceil@10_diag={s['ceiling@10_diag']:.4f}")
        print(f"    Ausschoepfung @1={s['exploitation@1_diag']:.4f}  @10={s['exploitation@10_diag']:.4f}")
        print(f"    class_recall@10   Key={s['class_recall@10_key']:.4f}  Value={s['class_recall@10_value']:.4f}")
        print(f"    Zufallsniveau recall@10_diag={s['random_recall@10_diag']:.4f}")
        print(f"    Zufallsniveau class_recall@10  Key={s['random_class_recall@10_key_mean']:.4f}"
              f"+-{s['random_class_recall@10_key_std']:.4f}  "
              f"Value={s['random_class_recall@10_value_mean']:.4f}+-{s['random_class_recall@10_value_std']:.4f}")
        print(f"    Haeufigkeitsniveau class_recall@10  Key={s['freq_class_recall@10_key']:.4f}  "
              f"Value={s['freq_class_recall@10_value']:.4f}")


def to_dataframe(summ: dict, split_name: str) -> pd.DataFrame:
    rows = []
    for bucket, s in summ.items():
        if s is None:
            continue
        row = {"split": split_name, "bucket": bucket}
        row.update(s)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B/T4: Retrieval-Baselines")
    parser.add_argument("--backbone", type=str, default="ae", choices=["ae", "tessera"],
                         help="checkpoints/<backbone>_v1/best.pt und "
                              "data/processed/pixels_<backbone>.parquet werden daraus abgeleitet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbone = args.backbone
    ckpt_path = Path(f"checkpoints/{backbone}_v1/best.pt")
    pixels_path = f"data/processed/pixels_{backbone}.parquet"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"retrieval_baselines_{backbone}_v1_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"=== run_retrieval_baselines.py gestartet (Phase B, {backbone}_v1) ===")
    print(f"Log-Datei: {log_path}")
    print(f"Device: {DEVICE}")

    _smoke_test_extract_primary_class()

    model, cfg = load_model(ckpt_path)
    ds = load_dataset_with_values(pixels_path)
    apply_checkpoint_normalization(ds, cfg)

    idx_val = np.where(ds.split == "val")[0]
    idx_test = np.where(ds.split == "test")[0]
    print(f"\nVal: {len(idx_val)} Zeilen, Test: {len(idx_test)} Zeilen")

    print("\n--- Verteilung primary_key / primary_value (Testset) ---")
    report_distribution("primary_key (Test)", ds.primary_key[idx_test])
    report_distribution("primary_value (Test)", ds.primary_value[idx_test])
    n_key_nonempty = int((ds.primary_key[idx_test] != "").sum())
    n_val_missing_given_key = int(((ds.primary_key[idx_test] != "") & (ds.primary_value[idx_test] == "")).sum())
    print(f"\n  Zusatz: bei vorhandenem primary_key aber NICHT parsebarem Value: "
          f"{n_val_missing_given_key} von {n_key_nonempty} "
          f"({n_val_missing_given_key / n_key_nonempty:.4%}) - zusaetzlich aus dem Value-Mittel "
          f"ausgeschlossen (ueber die primary_key-Leere hinaus).")

    print("\n--- Verteilung primary_key / primary_value (Val, zur Diagnose) ---")
    report_distribution("primary_key (Val)", ds.primary_key[idx_val])
    report_distribution("primary_value (Val)", ds.primary_value[idx_val])

    top_key_test = top_k_set(ds.primary_key[idx_test])
    top_val_test = top_k_set(ds.primary_value[idx_test])
    top_key_val = top_k_set(ds.primary_key[idx_val])
    top_val_val = top_k_set(ds.primary_value[idx_val])
    print(f"\nHaeufigkeitsniveau-Referenzmenge (Test), Key:   {sorted(top_key_test)}")
    print(f"Haeufigkeitsniveau-Referenzmenge (Test), Value: {sorted(top_val_test)}")

    t0 = time.time()
    acc_test = evaluate_full(model, ds, idx_test, top_key_test, top_val_test)
    print(f"\nTest-Auswertung fertig ({time.time() - t0:.1f}s)")
    summ_test = summarize(acc_test)
    print_table(summ_test, "TEST")

    t0 = time.time()
    acc_val = evaluate_full(model, ds, idx_val, top_key_val, top_val_val)
    print(f"\nVal-Auswertung fertig ({time.time() - t0:.1f}s)")
    summ_val = summarize(acc_val)
    print_table(summ_val, "VAL")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([to_dataframe(summ_test, "test"), to_dataframe(summ_val, "val")], ignore_index=True)
    out_path = OUT_DIR / f"retrieval_baselines_{backbone}_v1.csv"
    df.to_csv(out_path, index=False)
    print(f"\nErgebnistabelle gespeichert: {out_path}")


if __name__ == "__main__":
    main()
