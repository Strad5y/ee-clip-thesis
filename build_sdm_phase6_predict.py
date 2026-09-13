"""
build_sdm_phase6_predict.py

Phase 6 (INSTRUCTIONS_sdm.md): Vorhersagekopf (zweischichtiger MLP, Embeddings eingefroren,
Softmax auf C NS-SeC-Kategorien) + zwei Baselines (Mittelwertsvorhersage, Random-Embedding).
Fuenf Seeds je Konfiguration, Mittelwert +- Standardabweichung.

Metrik hier ist die distributionale Cross-Entropy (Trainings-/Val-/Test-Loss) - die eigentlichen
Bewertungsmetriken (KL/L1/Chebyshev mit den in Phase 7 noch zu treffenden Entscheidungen zu
KL-Richtung, Null-Behandlung, L1-Definition, Kachel-Bootstrap) werden BEWUSST NICHT hier
vorweggenommen, sondern erst in Phase 7 berechnet - Phase 6 prueft nur, dass die Koepfe
trainieren/konvergieren und liefert die Test-Predictions, die Phase 7 dann auswertet.

Konfigurationsraum (Nutzervorgabe, kein voller Kreuzprodukt-Sweep):
  Hauptlauf:      9cat + Split-Variante A + partitionsreine Poolung
  Robustheit 1:   8cat  (sonst wie Hauptlauf)      - isoliert den Kategorien-Effekt
  Robustheit 2:   Split-Variante B (sonst Hauptlauf) - isoliert den Split-Variante-Effekt
  Robustheit 3:   Poolung ueber alle Pixel (sonst Hauptlauf) - isoliert den Poolungs-Effekt
Jedes Szenario: 2 Backbones x 3 Arme x 5 Seeds.

Baselines:
  - Mittelwertsvorhersage: deterministisch (kein Seed noetig), nur von (target, split) abhaengig,
    NICHT von Backbone/Arm/Poolung - 3 Szenarien (target,split)-Paare aus den vier obigen.
  - Random-Embedding: 5 Seeds (Embedding UND Kopf-Init pro Seed neu gewuerfelt), gleiche
    Dimensionalitaet wie das jeweils verglichene Arm (64/128 fuer Arm A, 256 fuer Arm B/C) -
    fairer, dimensionsangepasster Vergleich statt einer einzigen globalen Zahl.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from run_landuse_probe import EE_DIMS, LATENT_DIM

SEEDS = [0, 1, 2, 3, 4]
HIDDEN_DIM = 128
MAX_EPOCHS = 500
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

SPLITS_DIR = cfg.DATA / "splits"
SPLIT_PATHS = {
    "A": SPLITS_DIR / "sdm_lsoa_splits_london.csv",
    "B": SPLITS_DIR / "sdm_lsoa_splits_london_min60.csv",
}
TARGET_PATHS = {
    "9cat": cfg.DATA / "processed" / "sdm_targets_london_9cat.parquet",
    "8cat": cfg.DATA / "processed" / "sdm_targets_london_8cat.parquet",
}


@dataclass
class Scenario:
    name: str
    target: str = "9cat"
    split: str = "A"
    pooling: str = "pure"  # "pure" oder "all"


SCENARIOS = [
    Scenario("Hauptlauf (9cat/A/pure)"),
    Scenario("Robustheit: 8cat", target="8cat"),
    Scenario("Robustheit: Split B", split="B"),
    Scenario("Robustheit: Poolung alle Pixel", pooling="all"),
]


class MLPHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, n_classes))

    def forward(self, x):
        return self.net(x)  # Logits - Softmax steckt in der Loss (log_softmax)


def soft_ce_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Distributionale Cross-Entropy fuer Soft-Targets (Zeilen summieren auf 1, keine
    One-Hot-Klassen)."""
    log_p = F.log_softmax(logits, dim=1)
    return -(target_probs * log_p).sum(dim=1).mean()


def load_target(target: str) -> tuple:
    df = pd.read_parquet(TARGET_PATHS[target])
    cat_cols = [c for c in df.columns if c.startswith("p_")]
    return df[["lsoa_code"] + cat_cols].copy(), cat_cols


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_PATHS[split], usecols=["lsoa_code", "split"])
    return df


def load_pooled(backbone: str, arm: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.DATA / "processed" / f"sdm_pooled_{backbone}_arm{arm}.parquet")


def build_xy(backbone: str, arm: str, scenario: Scenario) -> tuple:
    pooled = load_pooled(backbone, arm)
    prefix = "pure_" if scenario.pooling == "pure" else "all_"
    emb_cols = [c for c in pooled.columns if c.startswith(prefix)]
    n_pix_col = "n_pixels_pure" if scenario.pooling == "pure" else "n_pixels_all"

    targets, cat_cols = load_target(scenario.target)
    splits = load_split(scenario.split)

    merged = pooled[["lsoa_code", n_pix_col] + emb_cols].merge(
        targets, on="lsoa_code", how="inner"
    ).merge(splits, on="lsoa_code", how="inner")
    merged = merged[merged[n_pix_col] > 0]  # 0 erwartet (siehe Phase 5), Sicherheitsnetz

    X = merged[emb_cols].to_numpy(dtype=np.float32)
    y = merged[cat_cols].to_numpy(dtype=np.float32)
    split_arr = merged["split"].to_numpy()
    lsoa_codes = merged["lsoa_code"].to_numpy()
    return X, y, split_arr, len(cat_cols), lsoa_codes


def fit_mlp(X: np.ndarray, y: np.ndarray, split: np.ndarray, seed: int, d_in: int, n_classes: int,
            lsoa_codes: np.ndarray = None) -> dict:
    torch.manual_seed(seed)
    train_m, val_m, test_m = split == "train", split == "val", split == "test"
    Xtr, ytr = torch.from_numpy(X[train_m]), torch.from_numpy(y[train_m])
    Xva, yva = torch.from_numpy(X[val_m]), torch.from_numpy(y[val_m])
    Xte, yte = torch.from_numpy(X[test_m]), torch.from_numpy(y[test_m])

    model = MLPHead(d_in, HIDDEN_DIM, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = soft_ce_loss(model(Xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = soft_ce_loss(model(Xva), yva).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_loss = soft_ce_loss(model(Xtr), ytr).item()
        test_loss = soft_ce_loss(model(Xte), yte).item()
        test_pred = F.softmax(model(Xte), dim=1).numpy()
    return {"train_loss": train_loss, "val_loss": best_val, "test_loss": test_loss,
            "epochs": epoch + 1, "test_pred": test_pred, "y_test": y[test_m],
            "test_lsoa": lsoa_codes[test_m] if lsoa_codes is not None else None}


def run_seeds(X, y, split, d_in, n_classes, seeds=SEEDS) -> pd.DataFrame:
    rows = []
    for s in seeds:
        r = fit_mlp(X, y, split, s, d_in, n_classes)
        rows.append({"seed": s, "train_loss": r["train_loss"], "val_loss": r["val_loss"],
                      "test_loss": r["test_loss"], "epochs": r["epochs"]})
    return pd.DataFrame(rows)


def mean_baseline(y: np.ndarray, split: np.ndarray) -> dict:
    train_mean = y[split == "train"].mean(axis=0)
    train_mean = train_mean / train_mean.sum()
    test_y = y[split == "test"]
    pred = np.tile(train_mean, (len(test_y), 1))
    eps = 1e-8
    test_loss = float(-(test_y * np.log(pred + eps)).sum(axis=1).mean())
    return {"test_loss": test_loss, "train_mean_dist": train_mean.tolist()}


def random_embedding_baseline(y, split, d_in, n_classes, seeds=SEEDS) -> pd.DataFrame:
    rows = []
    n = len(y)
    for s in seeds:
        rng = np.random.default_rng(s)
        X_rand = rng.standard_normal((n, d_in)).astype(np.float32)
        r = fit_mlp(X_rand, y, split, s, d_in, n_classes)
        rows.append({"seed": s, "train_loss": r["train_loss"], "val_loss": r["val_loss"],
                      "test_loss": r["test_loss"], "epochs": r["epochs"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, col: str = "test_loss") -> str:
    return f"{df[col].mean():.4f} +- {df[col].std():.4f}"


def main():
    t_start = time.time()
    all_rows = []
    baseline_rows = []

    for scenario in SCENARIOS:
        print(f"\n{'=' * 10} Szenario: {scenario.name} {'=' * 10}")
        _, cat_cols = load_target(scenario.target)
        n_classes = len(cat_cols)

        # Mittelwertsvorhersage-Baseline: nur von (target, split) abhaengig
        # (einmal je Szenario - identisch fuer alle Backbones/Arme, daher hier statt
        # in der Backbone/Arm-Schleife berechnet)
        # y/split haengen nur von (target, split) ab, nicht von backbone/arm/pooling - "tessera/a"
        # ist hier nur ein beliebiger Traeger, um sie ueber build_xy() konsistent gejoint zu bekommen.
        _, y_any, split_any, _, _ = build_xy("tessera", "a", scenario)
        mb = mean_baseline(y_any, split_any)
        print(f"  Mittelwertsvorhersage-Baseline: test_loss={mb['test_loss']:.4f} (deterministisch)")
        baseline_rows.append({"scenario": scenario.name, "baseline": "mean", "backbone": "-", "arm": "-",
                               "test_loss_mean": mb["test_loss"], "test_loss_std": 0.0})

        for backbone in ["ae", "tessera"]:
            for arm in ["a", "b", "c"]:
                X, y, split_arr, n_classes, _lsoa = build_xy(backbone, arm, scenario)
                d_in = X.shape[1]
                n_train, n_val, n_test = (split_arr == "train").sum(), (split_arr == "val").sum(), (split_arr == "test").sum()

                res = run_seeds(X, y, split_arr, d_in, n_classes)
                print(f"  {backbone}/{arm} (d={d_in}, n={len(X)}={n_train}/{n_val}/{n_test}): "
                      f"test_loss={summarize(res)}  (val={summarize(res, 'val_loss')})")
                for _, row in res.iterrows():
                    all_rows.append({"scenario": scenario.name, "backbone": backbone, "arm": arm,
                                      **row.to_dict()})

                rb = random_embedding_baseline(y, split_arr, d_in, n_classes)
                print(f"    Random-Embedding-Baseline (d={d_in}): test_loss={summarize(rb)}")
                baseline_rows.append({"scenario": scenario.name, "baseline": "random_embedding",
                                       "backbone": backbone, "arm": arm,
                                       "test_loss_mean": rb["test_loss"].mean(),
                                       "test_loss_std": rb["test_loss"].std()})

    results = pd.DataFrame(all_rows)
    baselines = pd.DataFrame(baseline_rows)
    out_results = cfg.DATA / "processed" / "sdm_phase6_head_results.csv"
    out_baselines = cfg.DATA / "processed" / "sdm_phase6_baselines.csv"
    results.to_csv(out_results, index=False)
    baselines.to_csv(out_baselines, index=False)
    print(f"\nGespeichert: {out_results}, {out_baselines}")
    print(f"Gesamtlaufzeit: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
