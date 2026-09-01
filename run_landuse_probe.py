"""
run_landuse_probe.py

Phase 3 (CLAUDE_CODE_TASKS_urban_atlas_eval.md): Linear Probe auf gefrorenen EE-Embeddings
gegen Urban-Atlas-Landnutzungsklassen. Sechs Konfigurationen (roh / zufaellig projiziert /
trainiert projiziert, je Backbone), raeumlich geblockter Split (train/val/test-Spalte aus
build_landuse_pairs.py, NICHT der zufaellige train_frac-Split, den eval_landuse.py::
linear_probe() bisher gemacht hat).

Nutzervorgaben vor Phase 3:
  - Klasse 12400 (Airports) ausgeschlossen (wie 33000/22000: zu wenige Test-Pixel, sonst
    unterschiedliche Klassenmengen zwischen den Backbones -> Macro-F1 nicht vergleichbar).
  - split='missing' explizit ausgeschlossen aus train/val/test.
  - TESSERA-z-score-Statistik aus dem gespeicherten Artefakt (norm_stats_tessera.json) laden,
    NICHT neu fitten (sonst Leakage/Inkonsistenz mit dem Trainingslauf).
  - Vor dem vollen Lauf (3,7 Mio Trainingszeilen) eine Laufzeitschaetzung an einer Stichprobe
    berichten; bei > 2h vor dem Start melden statt einfach laufen zu lassen.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as cfg
from eval_landuse import linear_probe
from model import ProjectionHead
from normalize import apply_zscore, load_stats

EXCLUDED_CLASS = "12400"  # Airports - Nutzerentscheidung, siehe Docstring
NORM_STATS_PATH = cfg.DATA / "processed" / "norm_stats_tessera.json"
CHECKPOINTS = {
    "ae": cfg.ROOT / "checkpoints" / "ae_v1" / "best.pt",
    "tessera": cfg.ROOT / "checkpoints" / "tessera_v1" / "best.pt",
}
EE_DIMS = {"ae": 64, "tessera": 128}
LATENT_DIM = 256
HIDDEN_DIM = 256
DROPOUT = 0.1
RANDOM_HEAD_SEED = 0


def load_clean(name: str) -> pd.DataFrame:
    path = cfg.DATA / "processed" / f"landuse_pairs_{name}.parquet"
    df = pd.read_parquet(path, columns=["code_2021", "split", "embedding"])
    n_before = len(df)
    n_excl_class = int((df["code_2021"] == EXCLUDED_CLASS).sum())
    n_excl_missing = int((df["split"] == "missing").sum())
    df = df[(df["code_2021"] != EXCLUDED_CLASS) & (df["split"] != "missing")].reset_index(drop=True)
    n_after = len(df)
    print(f"  {name}: {n_before} -> {n_after} Zeilen "
          f"(-{n_before - n_after}: {n_excl_class} Klasse {EXCLUDED_CLASS}, "
          f"{n_excl_missing} split='missing', Ueberlappung moeglich)")
    print(f"    Klassen nach Ausschluss: {df['code_2021'].nunique()}")
    return df


def load_trained_head(name: str) -> ProjectionHead:
    ckpt = torch.load(CHECKPOINTS[name], map_location="cpu", weights_only=False)
    head = ProjectionHead(EE_DIMS[name], LATENT_DIM, HIDDEN_DIM, DROPOUT)
    state = {k.replace("ee_head.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("ee_head.")}
    head.load_state_dict(state)
    head.eval()
    print(f"  {name}: trainierter Kopf geladen ({CHECKPOINTS[name]}, Epoche {ckpt['epoch']})")
    return head


def random_head(name: str, seed: int = RANDOM_HEAD_SEED) -> ProjectionHead:
    torch.manual_seed(seed)
    head = ProjectionHead(EE_DIMS[name], LATENT_DIM, HIDDEN_DIM, DROPOUT)
    head.eval()
    return head


@torch.no_grad()
def project(head: ProjectionHead, x: np.ndarray, batch_size: int = 200_000, device: str = "cpu") -> np.ndarray:
    head = head.to(device)
    out = np.empty((len(x), LATENT_DIM), dtype=np.float32)
    for i in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[i:i + batch_size]).to(device)
        out[i:i + batch_size] = head(batch).cpu().numpy()
    return out


def build_configs() -> dict:
    """Baut die 6 (X, y, split)-Konfigurationen. Rueckgabe: {config_name: (X, y, split)}."""
    configs = {}
    for name in ["ae", "tessera"]:
        print(f"\nLade {name}...")
        df = load_clean(name)
        emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)

        if name == "tessera":
            stats = load_stats(NORM_STATS_PATH)
            print(f"  z-score-Stats geladen: {NORM_STATS_PATH}")
            emb_raw = apply_zscore(emb, stats)
        else:
            emb_raw = emb  # AlphaEarth: bereits L2-normalisiert, kein Normalisierungsschritt

        y = df["code_2021"].to_numpy()
        split = df["split"].to_numpy()

        trained_head = load_trained_head(name)
        rand_head = random_head(name)

        configs[f"{name}_roh"] = (emb_raw, y, split)
        configs[f"{name}_zufaellig_projiziert"] = (project(rand_head, emb_raw), y, split)
        configs[f"{name}_trainiert_projiziert"] = (project(trained_head, emb_raw), y, split)

    return configs


def benchmark(configs: dict, sample_sizes=(20_000, 50_000, 100_000), c_grid=(0.1, 1.0, 10.0)) -> None:
    """Laufzeitschaetzung: LogisticRegression bei wachsenden Stichprobengroessen auf der
    teuersten (256-dim) Konfiguration timen, linear/quadratisch extrapolieren auf die volle
    Trainingsmenge, mit der Anzahl C-Werte im Hyperparameter-Grid multiplizieren."""
    name = "ae_trainiert_projiziert"  # 256-dim, repraesentativ fuer alle projizierten Configs
    X, y, split = configs[name]
    train_mask = split == "train"
    X_train_full, y_train_full = X[train_mask], y[train_mask]
    n_train_full = len(X_train_full)
    print(f"\nLaufzeit-Benchmark auf '{name}' (volle Trainingsmenge: {n_train_full} Zeilen, "
          f"{X.shape[1]} Dim, {len(np.unique(y))} Klassen)")

    rng = np.random.default_rng(0)
    times = []
    for n in sample_sizes:
        idx = rng.choice(n_train_full, size=min(n, n_train_full), replace=False)
        Xs, ys = X_train_full[idx], y_train_full[idx]
        t0 = time.time()
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, max_iter=200)
        clf.fit(Xs, ys)
        dt = time.time() - t0
        times.append(dt)
        print(f"  n={n}: {dt:.2f}s (n_iter={clf.n_iter_})")

    # Skalierung schaetzen: log-log-Steigung zwischen den Messpunkten (Potenzgesetz-Fit)
    log_n = np.log(sample_sizes)
    log_t = np.log(times)
    slope = np.polyfit(log_n, log_t, 1)[0]
    print(f"  Skalierungsexponent (Zeit ~ n^{slope:.2f})")

    t_full_single = times[-1] * (n_train_full / sample_sizes[-1]) ** slope
    n_fits = len(configs) * len(c_grid)
    t_total = t_full_single * n_fits
    print(f"\n  Geschaetzte Zeit fuer EINEN vollen Fit ({n_train_full} Zeilen): {t_full_single / 60:.1f} min")
    print(f"  Configs: {len(configs)}, C-Werte im Grid: {len(c_grid)} -> {n_fits} volle Fits")
    print(f"  GESCHAETZTE GESAMTLAUFZEIT: {t_total / 60:.1f} min ({t_total / 3600:.2f} h)")

    return t_total


def run_probe(configs: dict, c_grid=(0.1, 1.0, 10.0)) -> pd.DataFrame:
    rows = []
    for name, (X, y, split) in configs.items():
        train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"\n=== {name} ===  train={len(X_train)} val={len(X_val)} test={len(X_test)} dim={X.shape[1]}")
        best = None
        for C in c_grid:
            t0 = time.time()
            res = linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, C=C)
            dt = time.time() - t0
            print(f"  C={C}: val_macro_f1={res['val_macro_f1']:.4f} val_oa={res['val_oa']:.4f} "
                  f"test_macro_f1={res['test_macro_f1']:.4f} test_oa={res['test_oa']:.4f} ({dt:.1f}s)")
            if best is None or res["val_macro_f1"] > best["val_macro_f1"]:
                best = res
        rows.append({"config": name, "dim": X.shape[1], **best})

    result = pd.DataFrame(rows)
    return result


def main():
    configs = build_configs()
    t_total = benchmark(configs)

    if t_total > 2 * 3600:
        print("\nGeschaetzte Laufzeit > 2h - Abbruch vor dem vollen Lauf (Nutzervorgabe).")
        return

    print("\n=== Voller Lauf: 6 Konfigurationen x C-Grid ===")
    result = run_probe(configs)
    out_path = cfg.DATA / "processed" / "landuse_probe_results.csv"
    result.to_csv(out_path, index=False)
    print(f"\nErgebnisse gespeichert: {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
