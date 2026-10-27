"""
eval_landuse_heads.py

Nutzervorgabe (2026-09-28, Jakob, via Betreuer): Land-Use-Probe nicht nur mit LogisticRegression
(eval_landuse.py::linear_probe()) auswerten, sondern zusaetzlich mit CatBoost, einem 2-Layer-MLP
und TabPFN - Frage, ob das schwache Delta roh<->trainiert projiziert (siehe
landuse_probe_results_*_dedup_ckpt.csv) ein Artefakt der Linearitaet des Probes ist oder in den
Embeddings selbst steckt. Alle drei Funktionen halten sich an dieselbe Signatur/Rueckgabeform wie
eval_landuse.py::linear_probe() (C-Aequivalent als erster Hyperparameter, return_fit-Flag), damit
run_landuse_probe_heads.py sie austauschbar aufrufen kann.

TabPFN-Subsample: TabPFN's eigene "officially supported" Grenze ist MAX_NUMBER_OF_SAMPLES=10_000
(tabpfn/inference_config.py) - SUBSAMPLE_SIZE hier uebernimmt genau diesen Wert, keine frei
erfundene Zahl. MAX_NUMBER_OF_CLASSES=10 < unsere 24 Landuse-Klassen - deshalb zwingend
ignore_pretraining_limits=True; das ist eine dokumentierte Abweichung vom offiziell unterstuetzten
Einsatzbereich, kein stillschweigender Trick (siehe Warnung in tabpfn_probe()).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

SUBSAMPLE_SIZE = 10_000   # = TabPFN MAX_NUMBER_OF_SAMPLES (offizielle Grenze, siehe Docstring)
MIN_PER_CLASS = 50        # Floor je Klasse - kleinste Landuse-Klasse hat 482 Trainingszeilen


def stratified_subsample(y_train: np.ndarray, target_total: int = SUBSAMPLE_SIZE,
                          min_per_class: int = MIN_PER_CLASS, seed: int = 42) -> np.ndarray:
    """Stratifiziertes Subsample-Indexset aus y_train (Groesse ~target_total, kann wegen
    Rundung/Floor leicht abweichen - tatsaechliche Groesse wird vom Aufrufer geloggt). Jede
    Klasse bekommt zuerst min(count, min_per_class) Zeilen garantiert (sonst waeren seltene
    Klassen im Subsample gar nicht vertreten -> Macro-F1 systematisch kaputt), der Rest wird
    proportional zur Klassenhaeufigkeit im vollen Train-Split aufgefuellt. Ziehung ohne
    Zuruecklegen, fixer Seed -> reproduzierbar, identisch fuer alle Koepfe."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    floor = np.minimum(counts, min_per_class)
    remaining_budget = max(target_total - floor.sum(), 0)
    remaining_pool = counts - floor
    pool_sum = remaining_pool.sum()
    extra = np.zeros_like(floor)
    if pool_sum > 0 and remaining_budget > 0:
        extra = np.minimum(remaining_pool, np.round(remaining_budget * remaining_pool / pool_sum).astype(int))
    per_class_n = floor + extra

    idx_by_class = {c: np.where(y_train == c)[0] for c in classes}
    chosen = []
    for c, n in zip(classes, per_class_n):
        pool = idx_by_class[c]
        n = min(n, len(pool))
        chosen.append(rng.choice(pool, size=n, replace=False))
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return idx


def _metrics(y_val, pred_val, y_test, pred_test) -> dict:
    return {
        "val_oa": accuracy_score(y_val, pred_val),
        "val_macro_f1": f1_score(y_val, pred_val, average="macro"),
        "test_oa": accuracy_score(y_test, pred_test),
        "test_macro_f1": f1_score(y_test, pred_test, average="macro"),
    }


def catboost_probe(X_train, y_train, X_val, y_val, X_test, y_test, depth: int = 6,
                    iterations: int = 1000, learning_rate: float = 0.1, task_type: str = "GPU",
                    early_stopping_rounds: int = 50, return_fit: bool = False, seed: int = 0) -> dict:
    """Gradient-Boosted-Trees-Probe (CatBoostClassifier, multiclass). Early Stopping auf dem
    Val-Split (identisch zum raeumlich geblockten Split der anderen Koepfe, kein Refit auf
    Val+Train). task_type='GPU' faellt bei Fehlern (z.B. kein Speicher) NICHT still auf CPU
    zurueck - das wuerde Laufzeiten zwischen Configs unvergleichbar machen; Fehler wird
    durchgereicht."""
    from catboost import CatBoostClassifier

    clf = CatBoostClassifier(
        depth=depth, iterations=iterations, learning_rate=learning_rate, task_type=task_type,
        loss_function="MultiClass", eval_metric="TotalF1:average=Macro", random_seed=seed,
        verbose=False, early_stopping_rounds=early_stopping_rounds,
    )
    clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    pred_val = clf.predict(X_val).ravel()
    pred_test = clf.predict(X_test).ravel()

    result = {"best_iteration": clf.get_best_iteration(), **_metrics(y_val, pred_val, y_test, pred_test)}
    if return_fit:
        result["clf"] = clf
        result["pred_val"] = pred_val
        result["pred_test"] = pred_test
    return result


def mlp_probe(X_train, y_train, X_val, y_val, X_test, y_test, hidden_dim: int = 256,
              dropout: float = 0.1, lr: float = 1e-3, batch_size: int = 4096, max_epochs: int = 100,
              patience: int = 10, device: str = "cuda", return_fit: bool = False, seed: int = 0) -> dict:
    """2-Layer-MLP-Probe (Linear -> GELU -> Dropout -> Linear, gleicher Stil wie model.py::
    ProjectionHead, nur mit Klassenanzahl statt Latentdim als Output und ohne L2-Normalize).
    Early Stopping auf Val-Macro-F1 (patience Epochen ohne Verbesserung), bestes Val-Checkpoint
    wird fuer die Test-Vorhersage restauriert. Labels werden intern ueber ein gemeinsames
    LabelEncoder-Fit auf allen drei Splits kodiert (train/val/test koennen im Subsample-Fall
    unterschiedliche Klassenmengen sehen, siehe run_landuse_probe_heads.py)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    yt, yv, yte = le.transform(y_train), le.transform(y_val), le.transform(y_test)
    n_classes = len(le.classes_)

    net = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(yt).long())
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    X_val_t = torch.from_numpy(X_val).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)

    best_val_f1, best_state, epochs_no_improve = -1.0, None, 0
    for epoch in range(max_epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            pred_val_epoch = net(X_val_t).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(yv, pred_val_epoch, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1, best_state, epochs_no_improve = val_f1, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred_val = net(X_val_t).argmax(dim=1).cpu().numpy()
        pred_test = net(X_test_t).argmax(dim=1).cpu().numpy()

    result = {"epochs_run": epoch + 1, **_metrics(yv, pred_val, yte, pred_test)}
    if return_fit:
        result["net"] = net
        result["label_encoder"] = le
        result["pred_val"] = le.inverse_transform(pred_val)
        result["pred_test"] = le.inverse_transform(pred_test)
    return result


class TabPFNUnavailable(RuntimeError):
    """Wird geworfen, wenn TabPFN wegen fehlender Lizenzannahme/API-Token nicht nutzbar ist -
    Aufrufer soll das explizit abfangen und den Kopf ueberspringen, NICHT stillschweigend."""


def tabpfn_probe(X_train, y_train, X_val, y_val, X_test, y_test, device: str = "cuda",
                  return_fit: bool = False, seed: int = 0) -> dict:
    """TabPFN-Probe (In-Context-Learner, kein Gradiententraining). X_train/y_train MUESSEN vom
    Aufrufer bereits auf stratified_subsample() reduziert sein (siehe SUBSAMPLE_SIZE-Docstring
    oben) - dieses Modul subsampled hier NICHT selbst, damit derselbe Subsample-Indexsatz
    nachvollziehbar von run_landuse_probe_heads.py geloggt wird.

    ignore_pretraining_limits=True ist noetig, weil 24 Landuse-Klassen > TabPFNs offiziell
    unterstuetzte MAX_NUMBER_OF_CLASSES=10 - das ist eine dokumentierte Abweichung vom offiziell
    unterstuetzten Einsatzbereich (Ergebnis entsprechend mit Vorbehalt zu lesen), kein Bug.

    Erfordert eine einmalige Lizenzannahme (TABPFN_TOKEN env var, siehe README/Fehlermeldung) -
    fehlt die, wird TabPFNUnavailable geworfen statt eines rohen Tracebacks."""
    from tabpfn import TabPFNClassifier
    from tabpfn.errors import TabPFNLicenseError

    if len(np.unique(y_train)) > 10:
        print("  WARNUNG [tabpfn_probe]: >10 Klassen, ausserhalb TabPFNs offiziell unterstuetztem "
              "Bereich (MAX_NUMBER_OF_CLASSES=10) - ignore_pretraining_limits=True gesetzt, "
              "Ergebnis mit Vorbehalt lesen.")

    clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True, random_state=seed)
    try:
        clf.fit(X_train, y_train)
    except TabPFNLicenseError as e:
        raise TabPFNUnavailable(str(e)) from e

    pred_val = clf.predict(X_val)
    pred_test = clf.predict(X_test)

    result = _metrics(y_val, pred_val, y_test, pred_test)
    if return_fit:
        result["clf"] = clf
        result["pred_val"] = pred_val
        result["pred_test"] = pred_test
    return result
