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
from splits import load_split_assignment, split_indices

CHECKPOINT_DIR = Path("checkpoints")


def build_model(cfg: dict) -> TwoTower:
    return TwoTower(
        ee_dim=cfg["ee_dim"],
        text_dim=cfg.get("text_dim", 1024),
        latent_dim=cfg.get("latent_dim", 256),
        hidden_dim=cfg.get("hidden_dim"),
        dropout=cfg.get("dropout", 0.1),
    )


def spatial_split(ds: PairDataset, splits_path) -> tuple:
    """Raeumlich geblockter Split (F1, siehe build_splits.py/splits.py) statt zufaelligem
    Zeilensplit - Gitterzellen wurden vorab einmalig zufaellig auf train/val/test verteilt,
    POIs erben den Split ihrer Zelle. split_indices() prueft bereits intern, dass die
    poi_id-Mengen der Splits disjunkt sind (bricht sonst mit AssertionError ab)."""
    assignment = load_split_assignment(splits_path)
    idx = split_indices(ds.poi_id, assignment)
    return Subset(ds, idx["train"]), Subset(ds, idx["val"]), idx["train"], idx["test"]


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


def make_subset_coverage_sampler(ds: PairDataset, train_idx: np.ndarray):
    """WeightedRandomSampler ueber coverage / n_pixels_poi NUR der Trainings-Teilmenge - Gewichte
    muessen relativ zu den lokalen Subset-Indizes (0..len(train_idx)-1) stehen, sonst zieht
    der Sampler falsch gewichtete Beispiele."""
    weights = torch.from_numpy(ds.sample_weight[train_idx].copy())
    return torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(train_idx), replacement=True)


def build_pos_mask(text_id: torch.Tensor) -> torch.Tensor:
    """(n,) Ganzzahl-Gruppen (siehe PairDataset.text_id, aus pd.factorize auf dem echten
    Text-String - byte-identischer Vergleich, kein Aehnlichkeitsmass/Schwellwert) -> (n,n)
    Bool-Maske, True wo zwei Batch-Eintraege EXAKT denselben Text haben. Diagonale ist immer
    True. Grund: 54,8% der POIs teilen sich nur 46 Texte (der haeufigste allein 31.230 POIs) -
    ohne diese Maske wertet SigLIPLoss solche Duplikate faelschlich als harte Negative."""
    return text_id.unsqueeze(0) == text_id.unsqueeze(1)


def recall_at_k(ee_z: torch.Tensor, text_z: torch.Tensor, pos_mask: torch.Tensor, k: int) -> float:
    """In-Batch-Retrieval: fuer jede Zeile i, ist IRGENDEIN wahrer Positiv (pos_mask[i,:]) unter
    den Top-k aehnlichsten Text-Vektoren? Multi-Positive-bewusst (nicht nur die Diagonale) -
    bei geteilten Texten (siehe build_pos_mask) zaehlt jeder Treffer unter den Duplikaten."""
    n = ee_z.shape[0]
    k = min(k, n)
    sim = ee_z @ text_z.T
    topk = sim.topk(k, dim=1).indices  # (n, k)
    hit = pos_mask.gather(1, topk).any(dim=1)  # (n,) - mind. ein Treffer unter Top-k?
    return hit.float().mean().item()


@torch.no_grad()
def evaluate(model, criterion, dl, device, use_pos_mask: bool = True) -> dict | None:
    """Recall wird zweifach berichtet: recall@k gegen die (ggf. Multi-Positive-) pos_mask, UND
    recall@k_diag ausschliesslich gegen die Diagonale (echte 1:1-Paare). Grund: bei pos_mask
    zaehlen Duplikat-Texte als zusaetzliche Treffer, was recall@k systematisch anhebt, unabhaengig
    von der eigentlichen Modellguete - recall@k_diag ist dadurch der einzige Wert, der zwischen
    einem Lauf mit und einem Lauf ohne pos_mask fair vergleichbar ist."""
    if len(dl.dataset) == 0:
        return None
    model.eval()
    total_loss = 0.0
    total_r1, total_r10 = 0.0, 0.0
    total_r1_diag, total_r10_diag = 0.0, 0.0
    n_batches = 0
    for batch in dl:
        ee = batch["ee_embedding"].to(device)
        text = batch["text_embedding"].to(device)
        n = ee.shape[0]
        diag_mask = torch.eye(n, device=device, dtype=torch.bool)
        pos_mask = (build_pos_mask(batch["text_id"].to(device)) if use_pos_mask and "text_id" in batch
                    else diag_mask)
        ee_z, text_z = model(ee, text)
        total_loss += criterion(ee_z, text_z, pos_mask=pos_mask).item()
        total_r1 += recall_at_k(ee_z, text_z, pos_mask, k=1)
        total_r10 += recall_at_k(ee_z, text_z, pos_mask, k=10)
        total_r1_diag += recall_at_k(ee_z, text_z, diag_mask, k=1)
        total_r10_diag += recall_at_k(ee_z, text_z, diag_mask, k=10)
        n_batches += 1
    model.train()
    return {
        "loss": total_loss / n_batches,
        "recall@1": total_r1 / n_batches,
        "recall@10": total_r10 / n_batches,
        "recall@1_diag": total_r1_diag / n_batches,
        "recall@10_diag": total_r10_diag / n_batches,
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

    ds = PairDataset(cfg["data"]["pairs_path"], cfg["data"]["text_embeddings_path"],
                      poi_text_path=cfg["data"].get("poi_text_path"))
    train_ds, val_ds, train_idx, _test_idx = spatial_split(ds, cfg["data"]["splits_path"])
    apply_normalization(ds, train_idx, cfg)

    # num_workers=0 (Default): DGX Spark teilt sich CPU/GPU 128 GB Unified Memory, und
    # DataLoader-Worker forken den kompletten Prozessstatus (inkl. bereits allokierter
    # CUDA-/Unified-Memory-Buffer) - >0 Worker sind hier ein reales Speicherrisiko, kein reines
    # Performance-Detail. Nur bewusst ueber die Config erhoehen, nicht als Default.
    num_workers = cfg["train"].get("num_workers", 0)
    sampler = make_subset_coverage_sampler(ds, train_idx) if cfg["train"].get("coverage_sampling", True) else None
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler,
                           shuffle=(sampler is None), num_workers=num_workers)
    # shuffle=True ist hier Pflicht, nicht Kosmetik: pairs_{backbone}.parquet ist nach poi_id
    # gruppiert (im Schnitt ~17-18 aufeinanderfolgende Zeilen je POI, siehe spatial_join.py) -
    # ohne Shuffle bestuenden Val-Batches fast nur aus einer Handvoll POIs (viele Pixel je POI),
    # was fuer den Batch-Contrastive-Loss praktisch keine echten Negative uebriglaesst und
    # val_loss massiv verzerrt (beobachtet: ~40x hoeher als train_loss im Dry-Run).
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

        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "criterion_state": criterion.state_dict(),
            "cfg": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

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

        if val_metrics is not None:
            current_metric = val_metrics[early_stopping_key]
            improved = current_metric > best_metric if maximize_metric else current_metric < best_metric
            if improved:
                best_metric = current_metric
                epochs_no_improve = 0
                torch.save(ckpt, ckpt_dir / "best.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
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
