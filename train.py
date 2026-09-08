"""
train.py

Config-getriebener Trainings-Loop fuer das Two-Tower-SigLIP-Setup. Ein Skript fuer alle
Backbone x Ablation-Kombinationen (settled decision, CLAUDE.md) - siehe configs/*.yaml.
Nur die Projektionskoepfe sind trainierbar, Encoder liefern bereits fertige Embeddings.

Ohne --config laeuft nur der CPU-Smoke-Test mit Dummy-Tensoren (Forward/Loss/Backward),
da echte Text-Embeddings noch nicht existieren (compute_text_embeddings.py ist nur ein Geruest).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from chunked_join_runner import LOG_DIR, _Tee
from dataset import PairDataset
from losses import SigLIPLoss
from model import TwoTower
from normalize import NONE, VALID_MODES, ZSCORE, apply_zscore, fit_zscore, save_stats

CHECKPOINT_DIR = Path("checkpoints")


def build_model(cfg: dict) -> TwoTower:
    return TwoTower(
        ee_dim=cfg["ee_dim"],
        text_dim=cfg.get("text_dim", 1024),
        latent_dim=cfg.get("latent_dim", 256),
        hidden_dim=cfg.get("hidden_dim"),
        dropout=cfg.get("dropout", 0.1),
    )


def pixel_split(ds: PairDataset) -> tuple:
    """Phase 7: der Split steht bereits direkt in der Pixel-Datei (Phase-5-Nachtrag,
    build_splits_pixels.py) - kein separater CSV-Merge ueber poi_id mehr noetig (splits.py
    ist fuer diesen Pfad nicht mehr im Einsatz). Trotzdem explizit auf Leakage geprueft:
    ein Pixel-Schluessel kann nur in EINEM Split stehen, das ist hier aber schon durch die
    Konstruktion (1 Split-Wert pro Zeile) garantiert, nicht durch einen Mengen-Join wie frueher."""
    train_idx = np.where(ds.split == "train")[0]
    val_idx = np.where(ds.split == "val")[0]
    test_idx = np.where(ds.split == "test")[0]
    n_missing = len(ds) - len(train_idx) - len(val_idx) - len(test_idx)
    if n_missing > 0:
        print(f"  Warnung: {n_missing} Pixel ohne gueltigen Split-Wert (weder train/val/test) - ausgeschlossen.")
    return Subset(ds, train_idx), Subset(ds, val_idx), train_idx, test_idx


def apply_normalization(ds: PairDataset, train_idx: np.ndarray, cfg: dict) -> None:
    """Normalisiert ds.ee IN-PLACE, Modus ueber cfg['normalization'] (siehe normalize.py) -
    backbone-abhaengig konfigurierbar, nicht an dieser Stelle hartcodiert. Stats werden
    AUSSCHLIESSLICH auf ds.ee[train_idx] gefittet (nie auf dem Gesamtdatensatz, sonst Leakage)
    und - falls 'zscore' - als Artefakt gespeichert, damit Val/Test/Inferenz dieselbe
    Statistik verwenden."""
    mode = cfg.get("normalization", NONE)
    if mode not in VALID_MODES:
        raise ValueError(f"Unbekannter normalization-Modus: {mode!r} (erwartet {VALID_MODES})")

    if mode == NONE:
        return  # No-Op: z.B. AlphaEarth ist bereits L2-normalisiert (siehe normalize.py)

    stats = fit_zscore(ds.ee[train_idx])
    stats_path = cfg["data"].get("norm_stats_path")
    if stats_path:
        save_stats(stats, stats_path)
    ds.ee = apply_zscore(ds.ee, stats)


def build_pos_mask(text_id: torch.Tensor) -> torch.Tensor:
    """(n,) Ganzzahl-Gruppen (siehe PairDataset.text_id, aus pd.factorize auf dem echten
    Text-String - byte-identischer Vergleich, kein Aehnlichkeitsmass/Schwellwert) -> (n,n)
    Bool-Maske, True wo zwei Batch-Eintraege EXAKT denselben Text haben. Diagonale ist immer
    True. Grund: 54,8% der POIs teilen sich nur 46 Texte (der haeufigste allein 31.230 POIs) -
    ohne diese Maske wertet SigLIPLoss solche Duplikate faelschlich als harte Negative."""
    return text_id.unsqueeze(0) == text_id.unsqueeze(1)


def recall_hits(ee_z: torch.Tensor, text_z: torch.Tensor, pos_mask: torch.Tensor, k: int) -> torch.Tensor:
    """Wie recall_at_k, aber gibt den Pro-Query-Treffervektor (n,) bool zurueck statt des
    Batch-Mittels - noetig fuer die Aufspaltung nach n_pois und fuer die Ceiling-Normierung
    (Phase-7-Nachtrag), die beide pro Query rechnen, nicht pro Batch."""
    n = ee_z.shape[0]
    k = min(k, n)
    sim = ee_z @ text_z.T
    topk = sim.topk(k, dim=1).indices  # (n, k)
    return pos_mask.gather(1, topk).any(dim=1)  # (n,) - mind. ein Treffer unter Top-k?


def recall_at_k(ee_z: torch.Tensor, text_z: torch.Tensor, pos_mask: torch.Tensor, k: int) -> float:
    """In-Batch-Retrieval: fuer jede Zeile i, ist IRGENDEIN wahrer Positiv (pos_mask[i,:]) unter
    den Top-k aehnlichsten Text-Vektoren? Multi-Positive-bewusst (nicht nur die Diagonale) -
    bei geteilten Texten (siehe build_pos_mask) zaehlt jeder Treffer unter den Duplikaten."""
    return recall_hits(ee_z, text_z, pos_mask, k).float().mean().item()


def recall_ceiling(text_id: torch.Tensor, k: int) -> torch.Tensor:
    """Nutzervorgabe (Phase-7-Nachtrag): recall@k_diag ist bei Textduplikaten nach oben
    gedeckelt - der Encoder liefert fuer identische Text-Strings identische Vektoren, kann
    also unter den n_dup Kandidaten mit demselben Text wie die Query NICHT gezielt genau den
    wahren diagonalen Treffer bevorzugen (alle Aehnlichkeiten sind fuer diese Gruppe exakt
    gleich). Ceiling pro Query = Erwartungswert des maximal erreichbaren Trefferanteils unter
    zufaelliger Tie-Break-Reihenfolge innerhalb der Duplikatgruppe: min(1, k / n_dup). n_dup
    wird IM SELBEN Kandidatenpool gezaehlt wie recall_hits (derselbe Batch), sonst waere die
    Ceiling nicht zum gemessenen Recall vergleichbar."""
    eq = text_id.unsqueeze(0) == text_id.unsqueeze(1)  # (n,n), Diagonale immer True
    n_dup = eq.sum(dim=1).float()  # pro Query: Anzahl Kandidaten mit identischem Text (>=1)
    return torch.clamp(k / n_dup, max=1.0)


def class_recall_at_k(ee_z: torch.Tensor, text_z: torch.Tensor, primary_key: np.ndarray, k: int) -> np.ndarray:
    """Nutzervorgabe (Phase-7-Nachtrag): klassenbezogener Recall@k - Treffer, wenn IRGENDEIN
    Top-k-Kandidat dieselbe primaere OSM-Klasse traegt wie die Query selbst (deren wahres Ziel
    ist per Konstruktion die Diagonale, also identisch mit primary_key[i]) - lockerer als
    recall@k_diag (exakter Textabgleich). Gibt NaN fuer Queries mit primary_key=='' zurueck
    (nicht aus dem Text parsebar, siehe dataset.py::extract_primary_key) statt sie als
    Treffer/Nichttreffer zu werten - beim Mitteln explizit auszuschliessen."""
    n = ee_z.shape[0]
    k = min(k, n)
    sim = ee_z @ text_z.T
    topk = sim.topk(k, dim=1).indices.cpu().numpy()
    hits = np.full(n, np.nan)
    for i in range(n):
        if primary_key[i] == "":
            continue
        hits[i] = float(np.any(primary_key[topk[i]] == primary_key[i]))
    return hits


def _new_ceiling_acc() -> dict:
    return {"n": 0, "hit1": 0.0, "hit10": 0.0, "ceil1": 0.0, "ceil10": 0.0,
            "class10_sum": 0.0, "class10_n": 0}


def _summarize_ceiling_acc(a: dict) -> dict | None:
    if a["n"] == 0:
        return None
    r1, r10 = a["hit1"] / a["n"], a["hit10"] / a["n"]
    c1, c10 = a["ceil1"] / a["n"], a["ceil10"] / a["n"]
    return {
        "n": a["n"],
        "recall@1_diag": r1, "recall@10_diag": r10,
        "ceiling@1_diag": c1, "ceiling@10_diag": c10,
        "exploitation@1_diag": r1 / c1 if c1 > 0 else float("nan"),
        "exploitation@10_diag": r10 / c10 if c10 > 0 else float("nan"),
        "class_recall@10": a["class10_sum"] / a["class10_n"] if a["class10_n"] > 0 else float("nan"),
    }


@torch.no_grad()
def evaluate(model, criterion, dl, device, use_pos_mask: bool = True) -> dict | None:
    """Recall wird zweifach berichtet: recall@k gegen die (ggf. Multi-Positive-) pos_mask, UND
    recall@k_diag ausschliesslich gegen die Diagonale (echte 1:1-Paare). Grund: bei pos_mask
    zaehlen Duplikat-Texte als zusaetzliche Treffer, was recall@k systematisch anhebt, unabhaengig
    von der eigentlichen Modellguete - recall@k_diag ist dadurch der einzige Wert, der zwischen
    einem Lauf mit und einem Lauf ohne pos_mask fair vergleichbar ist.

    Phase-7-Nachtrag (Nutzervorgabe): recall@k_diag allein ist zwischen Laeufen mit
    unterschiedlicher Duplikatrate NICHT vergleichbar (die Pixel-Aggregation hat die
    Duplikatrate gegenueber dem POI-Ebene-Lauf veraendert) - zusaetzlich Ceiling (maximal
    erreichbarer Recall bei perfekter Tie-Break-Ausnutzung), Ausschoepfung (gemessen/Ceiling)
    und ein klassenbezogener Recall@10 (Treffer = gleiche primaere OSM-Klasse statt exaktem
    Text), je getrennt fuer n_pois==1 und n_pois>1 UND insgesamt ('all'). Gewichtete
    Aggregation ueber Summe/Anzahl (nicht Mittel-der-Batch-Mittel wie bei den obigen
    Alt-Metriken) - noetig, weil die single/multi-Teilmengen pro Batch unterschiedlich gross
    sind und ein naives Batch-Mittel das sonst verzerren wuerde."""
    if len(dl.dataset) == 0:
        return None
    model.eval()
    total_loss = 0.0
    total_r1, total_r10 = 0.0, 0.0
    total_r1_diag, total_r10_diag = 0.0, 0.0
    n_batches = 0
    acc = {"all": _new_ceiling_acc(), "single": _new_ceiling_acc(), "multi": _new_ceiling_acc()}

    for batch in dl:
        ee = batch["ee_embedding"].to(device)
        text = batch["text_embedding"].to(device)
        n = ee.shape[0]
        diag_mask = torch.eye(n, device=device, dtype=torch.bool)
        text_id = batch["text_id"].to(device)
        pos_mask = (build_pos_mask(text_id) if use_pos_mask and "text_id" in batch else diag_mask)
        ee_z, text_z = model(ee, text)
        total_loss += criterion(ee_z, text_z, pos_mask=pos_mask).item()
        total_r1 += recall_at_k(ee_z, text_z, pos_mask, k=1)
        total_r10 += recall_at_k(ee_z, text_z, pos_mask, k=10)

        hit1_diag = recall_hits(ee_z, text_z, diag_mask, k=1).cpu().numpy()
        hit10_diag = recall_hits(ee_z, text_z, diag_mask, k=10).cpu().numpy()
        total_r1_diag += hit1_diag.mean()
        total_r10_diag += hit10_diag.mean()

        ceil1 = recall_ceiling(text_id, k=1).cpu().numpy()
        ceil10 = recall_ceiling(text_id, k=10).cpu().numpy()
        primary_key = np.array(batch["primary_key"])
        class10_hits = class_recall_at_k(ee_z, text_z, primary_key, k=10)

        n_pois = batch["n_pois"].numpy()
        for name, m in (("all", np.ones(n, dtype=bool)), ("single", n_pois == 1), ("multi", n_pois > 1)):
            cnt = int(m.sum())
            if cnt == 0:
                continue
            a = acc[name]
            a["n"] += cnt
            a["hit1"] += hit1_diag[m].sum()
            a["hit10"] += hit10_diag[m].sum()
            a["ceil1"] += ceil1[m].sum()
            a["ceil10"] += ceil10[m].sum()
            valid = ~np.isnan(class10_hits[m])
            a["class10_sum"] += class10_hits[m][valid].sum()
            a["class10_n"] += int(valid.sum())

        n_batches += 1
    model.train()
    return {
        "loss": total_loss / n_batches,
        "recall@1": total_r1 / n_batches,
        "recall@10": total_r10 / n_batches,
        "recall@1_diag": total_r1_diag / n_batches,
        "recall@10_diag": total_r10_diag / n_batches,
        "ceiling": {name: _summarize_ceiling_acc(acc[name]) for name in acc},
    }


def train(cfg: dict, checkpoint_dir: Path = CHECKPOINT_DIR, log_to_file: bool = True) -> TwoTower:
    run_name = f"{cfg.get('backbone', 'backbone')}_{cfg.get('ablation', 'v1')}"

    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"train_{run_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_path, "a")
        sys.stdout = _Tee(sys.stdout, log_file)
        sys.stderr = _Tee(sys.stderr, log_file)
        print(f"=== train.py gestartet ({run_name}) ===")
        print(f"Log-Datei: {log_path}")

    device = torch.device(cfg["train"].get("device", "cpu"))
    print(f"Device: {device}")
    model = build_model(cfg).to(device)
    criterion = SigLIPLoss(**cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    ds = PairDataset(cfg["data"]["pixels_path"], cfg["data"]["text_embeddings_path"])
    train_ds, val_ds, train_idx, _test_idx = pixel_split(ds)
    apply_normalization(ds, train_idx, cfg)

    # num_workers=0 (Default): DGX Spark teilt sich CPU/GPU 128 GB Unified Memory, und
    # DataLoader-Worker forken den kompletten Prozessstatus (inkl. bereits allokierter
    # CUDA-/Unified-Memory-Buffer) - >0 Worker sind hier ein reales Speicherrisiko, kein reines
    # Performance-Detail. Nur bewusst ueber die Config erhoehen, nicht als Default.
    num_workers = cfg["train"].get("num_workers", 0)
    # KEIN WeightedRandomSampler mehr (Phase 5/7-Vorgabe): weight ist durchgehend 1.0, ein
    # gewichteter Sampler mit konstanten Gewichten waere reine Ziehung-mit-Zuruecklegen ohne
    # echten Effekt, nur teurer als normales Shuffle. shuffle=True ist trotzdem Pflicht, nicht
    # Kosmetik: pixels_{backbone}.parquet ist tileweise aufgebaut (aggregate_pixel_pairs.py) -
    # ohne Shuffle bestuenden Batches fast nur aus einer Kachel, was fuer den Batch-Contrastive-
    # Loss praktisch keine raeumlich diversen Negative uebriglaesst.
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                           num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=num_workers)

    ckpt_dir = checkpoint_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(train_ds)} Trainings-/{len(val_ds)} Val-Zeilen, "
          f"Checkpoints unter {ckpt_dir}")

    # use_pos_mask=False fuer Kontrolllaeufe (Loss-Skalenvergleich) - sonst identisch, die
    # Maske wird dann nie gebaut/uebergeben, SigLIPLoss faellt auf die Default-Diagonalmaske
    # zurueck (wie vor der pos_mask-Einfuehrung).
    use_pos_mask = cfg["train"].get("use_pos_mask", True)

    patience = cfg["train"].get("patience", 10)

    # Nutzervorgabe: Early Stopping ist per Config-Feld auf eine beliebige val_metrics-Groesse
    # umstellbar (Default weiterhin val_loss). Grund: bei SigLIP haengt der Loss an lernbarer
    # Temperatur/Bias, die sich waehrend des Trainings mitveraendern - er ist deshalb kein
    # verlaessliches Stoppkriterium, waehrend Retrieval-Metriken davon unabhaengig sind.
    # "val_loss" wird auf val_metrics["loss"] gemappt (minimieren), alle anderen Keys
    # (z.B. "recall@10_diag") direkt aus val_metrics gelesen und maximiert.
    early_stopping_metric = cfg["train"].get("early_stopping_metric", "val_loss")
    early_stopping_key = "loss" if early_stopping_metric == "val_loss" else early_stopping_metric
    maximize_metric = early_stopping_metric != "val_loss"
    best_metric = float("-inf") if maximize_metric else float("inf")
    epochs_no_improve = 0

    # Nutzervorgabe: harte Stop-Bedingungen ausserhalb des regulaeren Early-Stoppings (das nur
    # auf val_loss reagiert) - Temperatur, die wegzulaufen droht, oder Retrieval, das sich ueber
    # mehrere Epochen konsistent verschlechtert, sollen den Lauf sofort abbrechen, nicht erst
    # nachtraeglich im Log auffallen.
    temp_guard = cfg["train"].get("temp_guard", 25.0)
    recall_fall_patience = cfg["train"].get("recall_fall_patience", 5)
    recall_fall_streak = 0
    prev_recall10_diag = None

    for epoch in range(cfg["train"]["epochs"]):
        t_epoch = time.time()
        epoch_losses = []
        for batch in train_dl:
            ee = batch["ee_embedding"].to(device)
            text = batch["text_embedding"].to(device)
            pos_mask = (build_pos_mask(batch["text_id"].to(device))
                        if use_pos_mask and "text_id" in batch else None)

            ee_z, text_z = model(ee, text)
            loss = criterion(ee_z, text_z, pos_mask=pos_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss_mean = float(np.mean(epoch_losses))
        train_loss_std = float(np.std(epoch_losses))

        val_metrics = evaluate(model, criterion, val_dl, device, use_pos_mask=use_pos_mask)
        epoch_s = time.time() - t_epoch
        temp = criterion.log_t.exp().item()
        bias = criterion.b.item()
        lr = optimizer.param_groups[0]["lr"]
        if val_metrics is not None:
            val_str = (f"val_loss={val_metrics['loss']:.4f} "
                       f"val_recall@1={val_metrics['recall@1']:.4f} "
                       f"val_recall@10={val_metrics['recall@10']:.4f} "
                       f"val_recall@1_diag={val_metrics['recall@1_diag']:.4f} "
                       f"val_recall@10_diag={val_metrics['recall@10_diag']:.4f}")
        else:
            val_str = "val_loss=n/a"
        print(f"epoch {epoch + 1}/{cfg['train']['epochs']}: "
              f"train_loss_mean={train_loss_mean:.4f} train_loss_std={train_loss_std:.4f} "
              f"{val_str} lr={lr:.2e} temp={temp:.4f} bias={bias:.4f} dauer={epoch_s:.1f}s")

        # Phase-7-Nachtrag (Nutzervorgabe): Ceiling/Ausschoepfung/Klassenrecall in jede Zeile,
        # getrennt fuer alle/single/multi-POI-Pixel - Recall allein ist zwischen Laeufen mit
        # unterschiedlicher Duplikatrate nicht vergleichbar.
        if val_metrics is not None:
            for bucket in ("all", "single", "multi"):
                c = val_metrics["ceiling"][bucket]
                if c is None:
                    print(f"  ceiling[{bucket}]: keine Pixel in dieser Teilmenge")
                    continue
                print(f"  ceiling[{bucket}] n={c['n']}: "
                      f"recall@1_diag={c['recall@1_diag']:.4f}/ceil={c['ceiling@1_diag']:.4f} "
                      f"(Ausschoepfung={c['exploitation@1_diag']:.4f}) "
                      f"recall@10_diag={c['recall@10_diag']:.4f}/ceil={c['ceiling@10_diag']:.4f} "
                      f"(Ausschoepfung={c['exploitation@10_diag']:.4f}) "
                      f"class_recall@10={c['class_recall@10']:.4f}")

        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "criterion_state": criterion.state_dict(),
            "cfg": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        # Bugfix (Phase-7-Nachtrag): der Best-Checkpoint-Block MUSS vor jedem Guard-'break'
        # laufen. Vorher stand er danach - ein Guard-Abbruch (temp_guard/recall_fall_patience)
        # verliess die Schleife, bevor die aktuelle Epoche ueberhaupt als 'best' in Frage kam,
        # selbst wenn sie tatsaechlich die bislang beste Metrik hatte (beobachtet: Epoche 7 im
        # ae_v1-Lauf hatte den bisher hoechsten recall@10_diag, aber best.pt blieb auf Epoche 6
        # stehen, weil der temp_guard-'break' zuerst kam).
        if val_metrics is not None:
            current_metric = val_metrics[early_stopping_key]
            improved = current_metric > best_metric if maximize_metric else current_metric < best_metric
            if improved:
                best_metric = current_metric
                epochs_no_improve = 0
                torch.save(ckpt, ckpt_dir / "best.pt")
            else:
                epochs_no_improve += 1

        if temp > temp_guard:
            print(f"STOP (Nutzervorgabe): Temperatur {temp:.4f} > {temp_guard} - Lauf wird abgebrochen.")
            break

        if val_metrics is not None:
            r10d = val_metrics["recall@10_diag"]
            recall_fall_streak = recall_fall_streak + 1 if (prev_recall10_diag is not None and r10d < prev_recall10_diag) else 0
            prev_recall10_diag = r10d
            if recall_fall_streak >= recall_fall_patience:
                print(f"STOP (Nutzervorgabe): val_recall@10_diag faellt seit {recall_fall_streak} "
                      f"aufeinanderfolgenden Epochen (aktuell {r10d:.4f}) - Lauf wird abgebrochen.")
                break

        if val_metrics is not None and not improved and epochs_no_improve >= patience:
            print(f"Early Stopping: {early_stopping_metric} seit {patience} Epochen nicht "
                  f"verbessert (bestes {early_stopping_metric}={best_metric:.4f}).")
            break

    return model


def smoke_test() -> None:
    """CPU-Smoke-Test mit Dummy-Tensoren: prueft nur, dass Forward/Loss/Backward durchlaufen -
    KEINE echten Daten."""
    torch.manual_seed(0)
    batch_size, ee_dim, text_dim, latent_dim = 8, 128, 1024, 256

    model = TwoTower(ee_dim=ee_dim, text_dim=text_dim, latent_dim=latent_dim)
    criterion = SigLIPLoss()

    ee = torch.randn(batch_size, ee_dim)
    text = torch.randn(batch_size, text_dim)

    ee_z, text_z = model(ee, text)
    assert ee_z.shape == (batch_size, latent_dim)
    assert torch.allclose(ee_z.norm(dim=-1), torch.ones(batch_size), atol=1e-5)

    loss = criterion(ee_z, text_z)
    assert loss.ndim == 0

    loss.backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "Kein Gradient bei den Projektionskoepfen angekommen"

    print(f"Smoke-Test OK: loss={loss.item():.4f}, {len(grads)} Parameter-Tensoren mit Gradient "
          f"(Forward/Loss/Backward laufen durch).")


def _smoke_test_pos_mask() -> None:
    """Nutzervorgabe: Batch mit absichtlich doppelten Texten -> pos_mask muss die
    entsprechenden Positionen als positiv markieren, UND der Loss mit korrekter Maske muss
    niedriger ausfallen als mit der Default-Diagonalmaske (die identische Texte faelschlich
    als harte Negative behandelt)."""
    torch.manual_seed(0)
    n, latent_dim = 8, 16

    # poi 0&3 teilen sich Text A, poi 2&5 teilen sich Text B, Rest hat eindeutige Texte
    text_id = torch.tensor([0, 1, 2, 0, 4, 2, 6, 7])
    pos_mask = build_pos_mask(text_id)

    assert bool(pos_mask[0, 3]) and bool(pos_mask[3, 0]), "Duplikat-Paar (0,3) muss positiv markiert sein"
    assert bool(pos_mask[2, 5]) and bool(pos_mask[5, 2]), "Duplikat-Paar (2,5) muss positiv markiert sein"
    assert not bool(pos_mask[0, 1]), "unterschiedliche Texte duerfen nicht positiv markiert werden"
    assert bool(pos_mask.diagonal().all()), "Diagonale muss immer positiv sein"
    assert int(pos_mask.sum()) == n + 4, "erwartet: 8 Diagonale + 2x2 symmetrische Duplikat-Paare"

    # ee_z == text_z konstruiert (bevor die Duplikate ueberschrieben werden): die 'wahren'
    # Paare (Diagonale) haben dadurch garantiert maximale Kosinus-Aehnlichkeit (=1), damit der
    # Effekt der Maskenkorrektur eindeutig und nicht vom Zufall der Initialisierung abhaengt.
    ee_z = F.normalize(torch.randn(n, latent_dim), dim=-1)
    text_z = ee_z.clone()
    text_z[3] = text_z[0]  # poi 3 teilt sich den Text-Vektor mit poi 0 (identischer Text)
    text_z[5] = text_z[2]  # poi 5 teilt sich den Text-Vektor mit poi 2

    criterion = SigLIPLoss(init_t=10.0, init_b=0.0)
    loss_with_mask = criterion(ee_z, text_z, pos_mask=pos_mask).item()
    loss_default = criterion(ee_z, text_z).item()  # Default: nur Diagonale positiv

    assert loss_with_mask < loss_default, (
        f"Loss mit korrekter pos_mask ({loss_with_mask:.4f}) sollte niedriger sein als mit der "
        f"Default-Diagonalmaske ({loss_default:.4f}), die identische Texte faelschlich bestraft"
    )

    print(f"Smoke-Test OK: pos_mask markiert doppelte Texte korrekt als positiv "
          f"({int(pos_mask.sum())} positive Eintraege statt {n} bei Diagonale-only), "
          f"Loss mit Maske ({loss_with_mask:.4f}) < Loss ohne Maske ({loss_default:.4f}).")


def overfit_sanity_check(n: int = 200, epochs: int = 300, lr: float = 1e-2, seed: int = 0) -> None:
    """Abnahme-Kriterium Task 3.2: Loop auf 200 fixen (synthetischen) Paaren ueberfitten -
    Loss muss klar sinken, Retrieval-Recall@1 auf denselben 200 Paaren nahe 1.0 gehen.
    Synthetische Tensoren statt echter Daten, da Qwen3-Embedding-4B hier nicht laeuft
    (compute_text_embeddings.py ist bewusst nur ein Geruest, siehe dortiger Docstring) -
    fuer eine reine Memorisierungs-/Gradientenfluss-Pruefung der Projektionskoepfe genuegt das:
    jedes der n EE/Text-Paare ist ein eindeutiger fixer Vektor, den die Koepfe einander
    zuordnen muessen."""
    torch.manual_seed(seed)
    ee_dim, text_dim, latent_dim = 128, 1024, 256

    ee = torch.randn(n, ee_dim)
    text = torch.randn(n, text_dim)

    model = TwoTower(ee_dim=ee_dim, text_dim=text_dim, latent_dim=latent_dim)
    criterion = SigLIPLoss()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr)

    first_loss = None
    for epoch in range(epochs):
        ee_z, text_z = model(ee, text)
        loss = criterion(ee_z, text_z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 0:
            first_loss = loss.item()
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            print(f"  overfit-epoch {epoch + 1}/{epochs}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        ee_z, text_z = model(ee, text)
        sim = (ee_z @ text_z.T).numpy()
    recall_at_1 = float((sim.argmax(axis=1) == np.arange(n)).mean())
    last_loss = loss.item()

    print(f"\nSanity-Overfit ({n} Paare): loss {first_loss:.4f} -> {last_loss:.4f}, "
          f"Recall@1={recall_at_1:.4f}")
    assert last_loss < first_loss, "Loss ist nicht gesunken"
    assert recall_at_1 > 0.95, f"Recall@1={recall_at_1:.4f} liegt nicht nahe 1.0"
    print("Abnahme-Kriterium erfuellt: Loss sinkt klar, Recall@1 nahe 1.0.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-Tower SigLIP Training")
    parser.add_argument("--config", type=str, default=None, help="Pfad zu configs/*.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Nur Dummy-Forward/Backward pruefen")
    parser.add_argument("--overfit-check", action="store_true",
                         help="Sanity-Overfit auf 200 synthetischen Paaren (Task-3.2-Abnahme)")
    args = parser.parse_args()

    if args.overfit_check:
        overfit_sanity_check()
        return

    if args.smoke_test or args.config is None:
        smoke_test()
        _smoke_test_pos_mask()
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
