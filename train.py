"""
train.py

Config-getriebener Trainings-Loop fuer das Two-Tower-SigLIP-Setup. Ein Skript fuer alle
Backbone x Ablation-Kombinationen (settled decision, CLAUDE.md) - siehe configs/*.yaml.
Nur die Projektionskoepfe sind trainierbar, Encoder liefern bereits fertige Embeddings.

Ohne --config laeuft nur der CPU-Smoke-Test mit Dummy-Tensoren (Forward/Loss/Backward),
da echte Text-Embeddings noch nicht existieren (compute_text_embeddings.py ist nur ein Geruest).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

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


def split_train_val(ds: PairDataset, val_frac: float, seed: int = 0):
    """Zufaelliger Index-Split; coverage-Sampler (siehe unten) braucht die lokalen
    Trainings-Indizes, um pro Split die richtigen Gewichte zu ziehen."""
    n = len(ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac) if val_frac > 0 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return Subset(ds, train_idx), Subset(ds, val_idx), train_idx


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


@torch.no_grad()
def evaluate(model, criterion, dl, device) -> float | None:
    if len(dl.dataset) == 0:
        return None
    model.eval()
    total, n_batches = 0.0, 0
    for batch in dl:
        ee = batch["ee_embedding"].to(device)
        text = batch["text_embedding"].to(device)
        ee_z, text_z = model(ee, text)
        total += criterion(ee_z, text_z).item()
        n_batches += 1
    model.train()
    return total / n_batches


def train(cfg: dict, checkpoint_dir: Path = CHECKPOINT_DIR) -> TwoTower:
    device = torch.device(cfg["train"].get("device", "cpu"))
    model = build_model(cfg).to(device)
    criterion = SigLIPLoss(**cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    ds = PairDataset(cfg["data"]["pairs_path"], cfg["data"]["text_embeddings_path"])
    train_ds, val_ds, train_idx = split_train_val(ds, cfg["train"].get("val_frac", 0.1))
    apply_normalization(ds, train_idx, cfg)

    sampler = make_subset_coverage_sampler(ds, train_idx) if cfg["train"].get("coverage_sampling", True) else None
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler, shuffle=(sampler is None))
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    run_name = f"{cfg.get('backbone', 'backbone')}_{cfg.get('ablation', 'v1')}"
    ckpt_dir = checkpoint_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        last_loss = None
        for batch in train_dl:
            ee = batch["ee_embedding"].to(device)
            text = batch["text_embedding"].to(device)

            ee_z, text_z = model(ee, text)
            loss = criterion(ee_z, text_z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        val_loss = evaluate(model, criterion, val_dl, device)
        temp = criterion.log_t.exp().item()
        bias = criterion.b.item()
        lr = optimizer.param_groups[0]["lr"]
        val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        print(f"epoch {epoch + 1}/{cfg['train']['epochs']}: "
              f"loss={last_loss:.4f} val_loss={val_str} lr={lr:.2e} temp={temp:.4f} bias={bias:.4f}")

        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "criterion_state": criterion.state_dict(),
            "cfg": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")

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
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
