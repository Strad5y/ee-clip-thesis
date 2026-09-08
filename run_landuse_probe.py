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
  - Normalisierung (z-score fuer TESSERA, none fuer AE) kommt aus der Config - fuer den
    trainierten Kopf aus ckpt['cfg'], fuer roh/zufaellig aus configs/{backbone}_v1.yaml (siehe
    resolve_normalization()); beide muessen uebereinstimmen. NIE neu fitten (sonst Leakage/
    Inkonsistenz mit dem Trainingslauf) - fehlende Statistikdatei bricht das Skript ab, faellt
    nicht stillschweigend auf eine andere Statistik zurueck.
  - Vor dem vollen Lauf eine Laufzeitschaetzung berichten (benchmark()) - Nutzervorgabe T5:
    reine Information, kein automatischer Abbruch mehr bei grosser geschaetzter Laufzeit; Laeufe
    sollen vollstaendig durchlaufen. Einziger Skip-Schalter ist das explizite
    --skip-train-bootstrap.

Nutzervorgaben nach dem ersten vollen Lauf (Streuung fuer den Vergleich trainiert vs. roh):
  - LogisticRegression+lbfgs ist bei fixen Daten deterministisch (konvexes Problem, kein
    stochastischer Solver) - "5 Seeds wiederholen" braucht also eine echte Zufallsquelle:
    Trainings-Bootstrap (Resampling der Trainingszeilen MIT Zuruecklegen, gleiche Groesse,
    C aus der Selektion fix, Refit, Auswertung auf FIXEM Val/Test) fuer die Verfahrensstreuung.
  - Test-Bootstrap (1000x, kein Refit, nur Resampling der bereits vorliegenden Testvorhersagen)
    fuer die Praezision der Punktschaetzung - billig, da kein Modell neu fitten noetig.
  - Gepaarter Test-Bootstrap: DIESELBEN 1000 Resample-Indizes fuer alle 3 Configs eines
    Backbones (roh/zufaellig/trainiert projiziert teilen sich Split + y_test), damit die
    Differenz trainiert-roh pro Resample paarweise berechnet werden kann. Bericht: Mittelwert
    + 95%-Perzentilintervall der Differenz; schliesst das Intervall die Null ein, gilt der
    Unterschied als nicht belegt.
  - Accuracy (= "oa"/Overall Accuracy, identisch zu accuracy_score) wird ueberall neben
    Macro-F1 ausgewiesen.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config as cfg
from eval_landuse import linear_probe
from model import ProjectionHead
from normalize import NONE, VALID_MODES, ZSCORE, apply_zscore, load_stats

EXCLUDED_CLASS = "12400"  # Airports - Nutzerentscheidung, siehe Docstring
CONFIG_DIR = cfg.ROOT / "configs"
CHECKPOINTS = {
    "ae": cfg.ROOT / "checkpoints" / "ae_v1" / "best.pt",
    "tessera": cfg.ROOT / "checkpoints" / "tessera_v1" / "best.pt",
}
# Erwartete Architektur (settled decision, CLAUDE.md: latent_dim/hidden_dim/dropout identisch
# ueber Backbones, nur ee_dim variiert). Nutzervorgabe Phase T5: diese Konstanten werden NICHT
# blind verwendet, sondern gegen ckpt["cfg"] verifiziert - bei Abweichung Abbruch, statt
# stillschweigend eine andere Architektur zu bauen (siehe check_architecture()).
EE_DIMS = {"ae": 64, "tessera": 128}
LATENT_DIM = 256
HIDDEN_DIM = 256
DROPOUT = 0.1
RANDOM_HEAD_SEED = 0

N_TRAIN_BOOTSTRAP = 5          # Verfahrensstreuung (Refit, teuer)
TRAIN_BOOTSTRAP_SEED0 = 1000
N_TEST_BOOTSTRAP = 1000        # Praezision der Schaetzung + gepaarter Vergleich (kein Refit, billig)
TEST_BOOTSTRAP_SEED0 = 2000
# Nutzervorgabe T5: kein Zeitbudget-Abbruch mehr - Laeufe sollen vollstaendig durchlaufen, auch
# ueber mehrere Stunden. benchmark() bleibt als reine (geloggte) Schaetzung erhalten.


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


def load_backbone_yaml_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}_v1.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_normalization(cfg_dict: dict, context: str) -> tuple[str, str | None]:
    """Liest normalization-Modus + (falls zscore) norm_stats_path aus einer Config (entweder
    configs/{backbone}_v1.yaml fuer roh/zufaellig, oder ckpt['cfg'] fuer den trainierten Kopf -
    Nutzervorgabe Phase T5: NICHT mehr am Backbone-Namen hartcodiert festmachen, sonst kann
    derselbe Fehler wie bei run_retrieval_baselines.py erneut passieren). Bricht bei fehlender
    Statistikdatei explizit ab, kein stillschweigendes Rohrechnen."""
    mode = cfg_dict.get("normalization", NONE)
    if mode not in VALID_MODES:
        raise ValueError(f"[{context}] Unbekannter normalization-Modus: {mode!r} (erwartet {VALID_MODES})")
    if mode == NONE:
        return mode, None
    stats_path = cfg_dict.get("data", {}).get("norm_stats_path")
    if not stats_path or not Path(stats_path).exists():
        raise FileNotFoundError(
            f"[{context}] normalization='{mode}' verlangt eine Statistikdatei, aber "
            f"{stats_path!r} existiert nicht - Abbruch, kein stillschweigendes Rohrechnen."
        )
    return mode, stats_path


def check_architecture(name: str, ckpt_cfg: dict) -> None:
    """Nutzervorgabe Phase T5: ee_dim/latent_dim/hidden_dim/dropout aus der Checkpoint-Config
    gegen die hier erwarteten Konstanten pruefen - bei Abweichung ABBRUCH, statt still eine
    andere (falsche) Architektur zu bauen, die dann ggf. sogar unbemerkt beim state_dict-Load
    durchrutscht (z.B. wenn Shapes zufaellig passen)."""
    expected = {"ee_dim": EE_DIMS[name], "latent_dim": LATENT_DIM,
                "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT}
    for key, want in expected.items():
        got = ckpt_cfg.get(key)
        if got != want:
            raise ValueError(
                f"{name}: Checkpoint-Config {key}={got!r} weicht von der erwarteten Konstante "
                f"{want!r} ab - Abbruch (Nutzervorgabe T5: keine stillschweigend andere "
                f"Architektur bauen)."
            )


def load_trained_head(name: str) -> tuple[ProjectionHead, dict]:
    ckpt_path = CHECKPOINTS[name]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["cfg"]
    check_architecture(name, ckpt_cfg)
    head = ProjectionHead(EE_DIMS[name], LATENT_DIM, HIDDEN_DIM, DROPOUT)
    state = {k.replace("ee_head.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("ee_head.")}
    head.load_state_dict(state)
    head.eval()
    print(f"  {name}: trainierter Kopf geladen ({ckpt_path}, Epoche {ckpt['epoch']})")
    return head, ckpt_cfg


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


def build_configs(backbones: tuple = ("ae", "tessera")) -> dict:
    """Baut die (X, y, split)-Konfigurationen je Backbone in `backbones`. Rueckgabe:
    {config_name: (X, y, split)}.

    Nutzervorgabe Phase T5: Normalisierung fuer 'roh'/'zufaellig projiziert' kommt aus
    configs/{backbone}_v1.yaml (kein Checkpoint noetig fuer diese beiden Arme), fuer
    'trainiert projiziert' aus der im Checkpoint selbst gespeicherten Config (ckpt['cfg']) -
    NICHT mehr am Backbone-Namen hartcodiert (siehe resolve_normalization()/check_architecture()).
    Beide Quellen muessen uebereinstimmen, sonst waeren die drei Arme nicht identisch
    vorverarbeitet - das wird explizit geprueft, nicht angenommen."""
    configs = {}
    for name in backbones:
        print(f"\nLade {name}...")
        df = load_clean(name)
        emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)
        y = df["code_2021"].to_numpy()
        split = df["split"].to_numpy()

        yaml_cfg = load_backbone_yaml_config(name)
        yaml_mode, yaml_stats_path = resolve_normalization(yaml_cfg, f"configs/{name}_v1.yaml")
        print(f"  Normalisierung (roh/zufaellig projiziert, aus configs/{name}_v1.yaml): "
              f"{yaml_mode}" + (f", Statistik={yaml_stats_path}" if yaml_stats_path else ""))

        trained_head, ckpt_cfg = load_trained_head(name)
        ckpt_mode, ckpt_stats_path = resolve_normalization(ckpt_cfg, f"Checkpoint-Config ({name})")
        print(f"  Normalisierung (trainiert projiziert, aus Checkpoint-Config): "
              f"{ckpt_mode}" + (f", Statistik={ckpt_stats_path}" if ckpt_stats_path else ""))

        if ckpt_mode != yaml_mode or ckpt_stats_path != yaml_stats_path:
            raise ValueError(
                f"{name}: Normalisierung weicht zwischen configs/{name}_v1.yaml "
                f"({yaml_mode!r}, {yaml_stats_path!r}) und der Checkpoint-Config "
                f"({ckpt_mode!r}, {ckpt_stats_path!r}) ab - alle drei Arme muessen identisch "
                f"vorverarbeitet werden. Abbruch, nicht stillschweigend eine der beiden nehmen."
            )

        if yaml_mode == ZSCORE:
            stats = load_stats(yaml_stats_path)
            emb_raw = apply_zscore(emb, stats)
        else:
            emb_raw = emb  # AlphaEarth: bereits L2-normalisiert, kein Normalisierungsschritt

        rand_head = random_head(name)

        configs[f"{name}_roh"] = (emb_raw, y, split)
        configs[f"{name}_zufaellig_projiziert"] = (project(rand_head, emb_raw), y, split)
        configs[f"{name}_trainiert_projiziert"] = (project(trained_head, emb_raw), y, split)

    return configs


def benchmark(configs: dict, sample_sizes=(20_000, 50_000, 100_000), c_grid=(0.1, 1.0, 10.0),
              max_iter: int = 1000) -> None:
    """Laufzeitschaetzung (reine Information, Nutzervorgabe T5: fuehrt zu KEINEM Ueberspringen
    mehr): LogisticRegression bei wachsenden Stichprobengroessen auf einer 256-dim projizierten
    Konfiguration timen, linear/quadratisch extrapolieren auf die volle Trainingsmenge, mit der
    Anzahl C-Werte im Hyperparameter-Grid multiplizieren. Name der Referenz-Config generisch aus
    `configs` ermittelt (nicht mehr hartcodiert auf 'ae_trainiert_projiziert' - das brach bei
    z.B. --backbones tessera, wo dieser Key gar nicht existiert)."""
    name = next(n for n in configs if n.endswith("_trainiert_projiziert"))
    X, y, split = configs[name]
    train_mask = split == "train"
    X_train_full, y_train_full = X[train_mask], y[train_mask]
    n_train_full = len(X_train_full)
    print(f"\nLaufzeit-Benchmark auf '{name}' (volle Trainingsmenge: {n_train_full} Zeilen, "
          f"{X.shape[1]} Dim, {len(np.unique(y))} Klassen, max_iter={max_iter})")

    rng = np.random.default_rng(0)
    times = []
    for n in sample_sizes:
        idx = rng.choice(n_train_full, size=min(n, n_train_full), replace=False)
        Xs, ys = X_train_full[idx], y_train_full[idx]
        t0 = time.time()
        clf = LogisticRegression(C=1.0, max_iter=max_iter)
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


def run_probe(configs: dict, c_grid=(0.1, 1.0, 10.0), max_iter: int = 1000) -> tuple[pd.DataFrame, dict, list]:
    """Waehlt C ueber val_macro_f1 (wie bisher). Behaelt zusaetzlich pro Config den fertigen
    Fit (clf, pred_val, pred_test) sowie X_train/y_train/X_val/X_test/y_val/y_test - fuer die
    Bootstraps unten, damit dort nichts aus dieser Selektion neu geladen/projiziert werden muss.
    Gibt zusaetzlich die einzelnen Fit-Dauern zurueck, um daraus (statt einer neuen Stichprobe)
    die Laufzeit des Trainings-Bootstraps unten empirisch zu schaetzen."""
    rows = []
    fits = {}
    fit_durations = []
    for name, (X, y, split) in configs.items():
        train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"\n=== {name} ===  train={len(X_train)} val={len(X_val)} test={len(X_test)} dim={X.shape[1]}")
        best = None
        for C in c_grid:
            t0 = time.time()
            res = linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, C=C,
                                max_iter=max_iter, return_fit=True)
            dt = time.time() - t0
            fit_durations.append(dt)
            print(f"  C={C}: val_macro_f1={res['val_macro_f1']:.4f} val_oa={res['val_oa']:.4f} "
                  f"test_macro_f1={res['test_macro_f1']:.4f} test_oa={res['test_oa']:.4f} ({dt:.1f}s)")
            if best is None or res["val_macro_f1"] > best["val_macro_f1"]:
                best = res
        rows.append({"config": name, "dim": X.shape[1],
                      **{k: v for k, v in best.items() if k not in ("clf", "pred_val", "pred_test")}})
        fits[name] = {
            "C": best["C"], "pred_val": best["pred_val"], "pred_test": best["pred_test"],
            "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
        }

    result = pd.DataFrame(rows)
    return result, fits, fit_durations


def train_bootstrap(fits: dict, n_seeds: int = N_TRAIN_BOOTSTRAP, seed0: int = TRAIN_BOOTSTRAP_SEED0,
                     max_iter: int = 1000) -> pd.DataFrame:
    """Verfahrensstreuung: je Seed die Trainingszeilen MIT Zuruecklegen neu ziehen (gleiche
    Groesse wie X_train), mit der in run_probe() gewaehlten C neu fitten, auf dem FIXEN Val/Test
    auswerten (Split bleibt der raeumlich geblockte - nur die Trainingsstichprobe variiert).
    Ohne dieses Resampling waere "5 Seeds" hier bedeutungslos: LogisticRegression+lbfgs ist bei
    unveraenderten Daten deterministisch, es gibt sonst keine Zufallsquelle."""
    rows = []
    for name, fit in fits.items():
        X_train, y_train = fit["X_train"], fit["y_train"]
        X_val, y_val = fit["X_val"], fit["y_val"]
        X_test, y_test = fit["X_test"], fit["y_test"]
        n_train = len(X_train)
        for i in range(n_seeds):
            seed = seed0 + i
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n_train, size=n_train)
            clf = LogisticRegression(C=fit["C"], max_iter=max_iter)
            clf.fit(X_train[idx], y_train[idx])
            pred_val = clf.predict(X_val)
            pred_test = clf.predict(X_test)
            rows.append({
                "config": name, "seed": seed,
                "val_oa": accuracy_score(y_val, pred_val),
                "val_macro_f1": f1_score(y_val, pred_val, average="macro"),
                "test_oa": accuracy_score(y_test, pred_test),
                "test_macro_f1": f1_score(y_test, pred_test, average="macro"),
            })
        print(f"  {name}: Trainings-Bootstrap {n_seeds}x fertig (C={fit['C']})")
    return pd.DataFrame(rows)


def summarize_bootstrap(df: pd.DataFrame, metrics=("val_oa", "val_macro_f1", "test_oa", "test_macro_f1")) -> pd.DataFrame:
    agg = df.groupby("config")[list(metrics)].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


def test_bootstrap(fits: dict, n_resamples: int = N_TEST_BOOTSTRAP, seed0: int = TEST_BOOTSTRAP_SEED0) -> dict:
    """Praezision der Punktschaetzung: die bereits vorliegenden Testvorhersagen (aus dem in
    run_probe() gewaehlten Fit) werden n_resamples-mal MIT Zuruecklegen resampled - kein Refit,
    daher billig. WICHTIG fuer den gepaarten Vergleich: innerhalb eines Backbones teilen sich
    roh/zufaellig/trainiert projiziert denselben Split (identisches y_test, identische
    Reihenfolge) - dieselbe seed0-Sequenz liefert dann fuer alle drei Configs eines Backbones
    DIESELBEN Resample-Indizes, sodass die Differenz trainiert-roh pro Resample-Index paarweise
    gebildet werden kann (siehe paired_diff())."""
    def backbone_of(name: str) -> str:
        return name.split("_", 1)[0]

    resample_idx_by_backbone = {}
    samples = {}
    for name, fit in fits.items():
        bb = backbone_of(name)
        y_test, pred_test = fit["y_test"], fit["pred_test"]
        n_test = len(y_test)
        if bb not in resample_idx_by_backbone:
            rng = np.random.default_rng(seed0)
            resample_idx_by_backbone[bb] = [rng.integers(0, n_test, size=n_test) for _ in range(n_resamples)]
        idx_list = resample_idx_by_backbone[bb]

        oa = np.empty(n_resamples)
        f1 = np.empty(n_resamples)
        for i, idx in enumerate(idx_list):
            oa[i] = accuracy_score(y_test[idx], pred_test[idx])
            f1[i] = f1_score(y_test[idx], pred_test[idx], average="macro")
        samples[name] = {"oa": oa, "macro_f1": f1}
        print(f"  {name}: Test-Bootstrap {n_resamples}x fertig "
              f"(test_oa={oa.mean():.4f}+-{oa.std():.4f}, test_macro_f1={f1.mean():.4f}+-{f1.std():.4f})")
    return samples


def paired_diff(samples: dict, alpha: float = 0.05, backbones: tuple = ("ae", "tessera")) -> pd.DataFrame:
    """Gepaarter Test-Bootstrap: Differenz trainiert-roh PRO RESAMPLE (dieselben Indizes, siehe
    test_bootstrap()), daraus Mittelwert + 95%-Perzentilintervall der Differenz. Schliesst das
    Intervall die Null ein, gilt der Unterschied als nicht belegt - das ist die Zahl, auf die es
    ankommt (Nutzervorgabe), nicht die blosse Differenz der Punktschaetzungen."""
    rows = []
    for bb in backbones:
        trained = samples[f"{bb}_trainiert_projiziert"]
        roh = samples[f"{bb}_roh"]
        for metric in ["macro_f1", "oa"]:
            diff = trained[metric] - roh[metric]
            lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            rows.append({
                "backbone": bb, "metric": metric,
                "mean_diff_trainiert_minus_roh": diff.mean(),
                "ci_low_95": lo, "ci_high_95": hi,
                "significant": bool(not (lo <= 0 <= hi)),
            })
    return pd.DataFrame(rows)


def majority_baseline(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> dict:
    """Nutzervorgabe Phase C: Haeufigkeitsniveau - ein Klassifikator, der IMMER die haeufigste
    Urban-Atlas-Klasse aus dem Trainingssplit beantwortet (analog zu einem echten Klassifikator:
    'gelernt' aus Train, ausgewertet auf Val/Test - keine Peek auf Test-Verteilung)."""
    majority_class = pd.Series(y_train).value_counts().idxmax()
    pred_val = np.full(len(y_val), majority_class)
    pred_test = np.full(len(y_test), majority_class)
    return {
        "majority_class": majority_class,
        "val_oa": accuracy_score(y_val, pred_val),
        "val_macro_f1": f1_score(y_val, pred_val, average="macro", zero_division=0),
        "test_oa": accuracy_score(y_test, pred_test),
        "test_macro_f1": f1_score(y_test, pred_test, average="macro", zero_division=0),
        "pred_test": pred_test,
    }


def per_class_f1(y_test: np.ndarray, pred_test: np.ndarray) -> pd.Series:
    """Nutzervorgabe Phase C: Per-Klassen-F1 auf Test, ueber ALLE in y_test vorkommenden Klassen
    (nicht nur die von pred_test getroffenen), damit auch nie vorhergesagte Klassen mit F1=0
    sichtbar bleiben statt stillschweigend zu fehlen."""
    labels = sorted(pd.unique(y_test))
    scores = f1_score(y_test, pred_test, labels=labels, average=None, zero_division=0)
    return pd.Series(scores, index=labels, name="f1")


def confusion(y_test: np.ndarray, pred_test: np.ndarray) -> pd.DataFrame:
    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, pred_test, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def main():
    parser = argparse.ArgumentParser(description="Phase 3/C: Land-Use-Linear-Probe")
    parser.add_argument("--backbones", type=str, default="ae,tessera",
                         help="Komma-getrennte Backbone-Liste, z.B. 'ae' fuer AlphaEarth only "
                              "(Phase-C-Nutzervorgabe: TESSERA vorerst aussen vor).")
    parser.add_argument("--max-iter", type=int, default=1000,
                         help="max_iter fuer alle LogisticRegression-Fits (Punktschaetzung, "
                              "Trainings-Bootstrap, Laufzeit-Benchmark) - Nutzervorgabe T5: "
                              "einheitlich 1000 statt sklearn-Default 200, um Nichtkonvergenz "
                              "(ConvergenceWarning) zu vermeiden.")
    parser.add_argument("--suffix", type=str, default=None,
                         help="Dateisuffix fuer alle Output-Dateien (z.B. '_ae_dedup'). Default: "
                              "'_' + Backbone-Liste (bisheriges Verhalten).")
    parser.add_argument("--skip-train-bootstrap", action="store_true",
                         help="Trainings-Bootstrap (Verfahrensstreuung ueber Resampling, 5 "
                              "Refits/Config) ueberspringen - das ist NICHT dasselbe wie "
                              "Phase D (Seeds fuer Kopf-Init/Zufallsprojektion, fixer Split); "
                              "Phase D braucht eigene Freigabe und eine eigene Seed-Schleife. "
                              "Nutzervorgabe T5: der einzige noch verbliebene Skip-Schalter - "
                              "muss explizit gesetzt werden, es gibt keinen automatischen "
                              "Zeitbudget-Abbruch mehr.")
    args = parser.parse_args()
    backbones = tuple(args.backbones.split(","))
    max_iter = args.max_iter
    suffix = args.suffix if args.suffix is not None else "_" + "-".join(backbones)

    configs = build_configs(backbones)
    # Nutzervorgabe T5: reine Information, kein Abbruch mehr bei geschaetzt >2h - Laeufe sollen
    # vollstaendig durchlaufen, auch wenn sie mehrere Stunden dauern.
    benchmark(configs, max_iter=max_iter)

    n_configs = len(configs)

    print(f"\n=== Voller Lauf: {n_configs} Konfigurationen x C-Grid (Punktschaetzung, max_iter={max_iter}) ===")
    result, fits, fit_durations = run_probe(configs, max_iter=max_iter)
    out_path = cfg.DATA / "processed" / f"landuse_probe_results{suffix}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nErgebnisse gespeichert: {out_path}")
    print(result.to_string(index=False))

    # Speicherreduktion (methodikneutral, Nutzervorgabe): die vollen X-Arrays in `configs` werden
    # nach run_probe() nur noch fuer Haeufigkeitsniveau/Per-Klassen-F1 der Majority-Baseline
    # gebraucht - dort ausschliesslich y/split (Indizes 1/2), nie X. fits(){} haelt bereits eigene
    # X_train/X_val/X_test-Kopien, keine Abhaengigkeit von den vollen configs-Arrays mehr. X hier
    # explizit droppen, damit der Referenzzaehler auf 0 faellt und der Speicher frei wird, bevor
    # der (deutlich teurere) Trainings-Bootstrap startet.
    configs = {name: (None, y, split) for name, (_, y, split) in configs.items()}
    print("\nSpeicherreduktion: volle X-Arrays aus 'configs' freigegeben (nur y/split behalten) - "
          "fits{} traegt bereits eigene Train/Val/Test-Kopien, keine Abhaengigkeit mehr.")

    # Nutzervorgabe Phase C: Haeufigkeitsniveau (immer die haeufigste Urban-Atlas-Klasse) je
    # Backbone - y/split sind ueber die 3 Repraesentationen eines Backbones identisch, deshalb
    # aus EINER beliebigen Config (z.B. '{bb}_roh') genommen, nicht dreifach berechnet.
    print("\n=== Haeufigkeitsniveau (immer die haeufigste Klasse aus Train) ===")
    majority_rows = []
    majority_fits = {}
    for bb in backbones:
        _, y_bb, split_bb = configs[f"{bb}_roh"]
        y_train_arr = y_bb[split_bb == "train"]
        y_val_arr = y_bb[split_bb == "val"]
        y_test_arr = y_bb[split_bb == "test"]
        maj = majority_baseline(y_train_arr, y_val_arr, y_test_arr)
        majority_fits[bb] = maj
        print(f"  {bb}: Mehrheitsklasse={maj['majority_class']}  "
              f"val_oa={maj['val_oa']:.4f} val_macro_f1={maj['val_macro_f1']:.4f}  "
              f"test_oa={maj['test_oa']:.4f} test_macro_f1={maj['test_macro_f1']:.4f}")
        majority_rows.append({"config": f"{bb}_haeufigkeitsniveau", "dim": 0,
                               "C": None, "val_oa": maj["val_oa"], "val_macro_f1": maj["val_macro_f1"],
                               "test_oa": maj["test_oa"], "test_macro_f1": maj["test_macro_f1"]})
    majority_df = pd.DataFrame(majority_rows)
    majority_out = cfg.DATA / "processed" / f"landuse_probe_majority_baseline{suffix}.csv"
    majority_df.to_csv(majority_out, index=False)
    print(f"\nHaeufigkeitsniveau gespeichert: {majority_out}")

    # Nutzervorgabe Phase C: Per-Klassen-F1 + Konfusionsmatrix auf Test, je Config UND fuer das
    # Haeufigkeitsniveau (gleiche y_test, damit direkt vergleichbar).
    print("\n=== Per-Klassen-F1 (Test) ===")
    per_class_frames = {}
    for name, fit in fits.items():
        s = per_class_f1(fit["y_test"], fit["pred_test"])
        per_class_frames[name] = s
        print(f"\n  {name}:")
        print(s.sort_values(ascending=False).to_string())
    for bb, maj in majority_fits.items():
        y_test = configs[f"{bb}_roh"][1][configs[f"{bb}_roh"][2] == "test"]
        s = per_class_f1(y_test, maj["pred_test"])
        per_class_frames[f"{bb}_haeufigkeitsniveau"] = s
        print(f"\n  {bb}_haeufigkeitsniveau:")
        print(s.sort_values(ascending=False).to_string())
    per_class_df = pd.DataFrame(per_class_frames)
    per_class_out = cfg.DATA / "processed" / f"landuse_probe_per_class_f1{suffix}.csv"
    per_class_df.to_csv(per_class_out)
    print(f"\nPer-Klassen-F1 gespeichert: {per_class_out}")

    print("\n=== Konfusionsmatrizen (Test) ===")
    for name, fit in fits.items():
        cm = confusion(fit["y_test"], fit["pred_test"])
        cm_out = cfg.DATA / "processed" / f"landuse_probe_confusion_{name}.csv"
        cm.to_csv(cm_out)
        print(f"  {name}: gespeichert unter {cm_out} ({cm.shape[0]}x{cm.shape[1]})")

    avg_fit_time = float(np.mean(fit_durations))
    n_train_bootstrap_fits = N_TRAIN_BOOTSTRAP * len(fits)
    projected_train_bootstrap_s = avg_fit_time * n_train_bootstrap_fits
    print(f"\nTrainings-Bootstrap-Schaetzung: {n_train_bootstrap_fits} Refits x ~{avg_fit_time:.1f}s "
          f"(empirisch aus der C-Selektion oben) = {projected_train_bootstrap_s / 60:.1f} min "
          f"({projected_train_bootstrap_s / 3600:.2f} h)")

    if args.skip_train_bootstrap:
        print("\n--skip-train-bootstrap gesetzt: Trainings-Bootstrap (Verfahrensstreuung, "
              "Phase-D-naher Resampling-Ansatz) wird in diesem Lauf NICHT ausgefuehrt - "
              "Phase D (Seeds fuer Kopf-Init/Zufallsprojektion, fixer Split) braucht eine "
              "eigene Freigabe und eine eigene Implementierung (dieses Resampling ist nicht "
              "dasselbe wie Seed-Variation).")
    else:
        # Nutzervorgabe T5: kein automatischer Zeitbudget-Abbruch mehr - die Schaetzung oben
        # ist reine Information, der Bootstrap laeuft immer, ausser bei explizitem --skip.
        print(f"\n=== Trainings-Bootstrap ({N_TRAIN_BOOTSTRAP} Seeds je Config, Verfahrensstreuung, "
              f"max_iter={max_iter}) ===")
        train_boot = train_bootstrap(fits, max_iter=max_iter)
        train_boot.to_csv(cfg.DATA / "processed" / f"landuse_probe_train_bootstrap{suffix}.csv", index=False)
        train_boot_summary = summarize_bootstrap(train_boot)
        train_boot_summary.to_csv(cfg.DATA / "processed" / f"landuse_probe_train_bootstrap_summary{suffix}.csv", index=False)
        print(train_boot_summary.to_string(index=False))

    print(f"\n=== Test-Bootstrap ({N_TEST_BOOTSTRAP}x je Config, Praezision der Schaetzung) ===")
    test_samples = test_bootstrap(fits)
    test_boot_summary = pd.DataFrame([
        {
            "config": name,
            "test_oa_mean": s["oa"].mean(), "test_oa_std": s["oa"].std(),
            "test_macro_f1_mean": s["macro_f1"].mean(), "test_macro_f1_std": s["macro_f1"].std(),
        }
        for name, s in test_samples.items()
    ])
    test_boot_summary.to_csv(cfg.DATA / "processed" / f"landuse_probe_test_bootstrap_summary{suffix}.csv", index=False)
    print(test_boot_summary.to_string(index=False))

    print("\n=== Gepaarter Test-Bootstrap: trainiert - roh je Backbone (95%-Intervall) ===")
    diff_result = paired_diff(test_samples, backbones=backbones)
    diff_result.to_csv(cfg.DATA / "processed" / f"landuse_probe_paired_diff{suffix}.csv", index=False)
    print(diff_result.to_string(index=False))


if __name__ == "__main__":
    main()
