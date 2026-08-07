"""
train.py

Config-getriebener Trainings-Loop fuer das Two-Tower-SigLIP-Setup. Ein Skript fuer alle
Backbone x Ablation-Kombinationen (settled decision, CLAUDE.md) - siehe configs/*.yaml.
Nur die Projektionskoepfe sind trainierbar, Encoder liefern bereits fertige Embeddings.

Ohne --config laeuft nur der CPU-Smoke-Test mit Dummy-Tensoren (Forward/Loss/Backward),
da echte Text-Embeddings noch nicht existieren (compute_text_embeddings.py ist nur ein Geruest).
"""
import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import PairDataset, make_coverage_sampler
from losses import SigLIPLoss
from model import TwoTower


def build_model(cfg: dict) -> TwoTower:
    return TwoTower(
        ee_dim=cfg["ee_dim"],
        text_dim=cfg.get("text_dim", 1024),
        latent_dim=cfg.get("latent_dim", 256),
        hidden_dim=cfg.get("hidden_dim"),
        dropout=cfg.get("dropout", 0.1),
    )


def train(cfg: dict) -> None:
    device = torch.device(cfg["train"].get("device", "cpu"))
    model = build_model(cfg).to(device)
    criterion = SigLIPLoss(**cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    ds = PairDataset(cfg["data"]["pairs_path"], cfg["data"]["text_embeddings_path"])
    sampler = make_coverage_sampler(ds) if cfg["train"].get("coverage_sampling", True) else None
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], sampler=sampler, shuffle=(sampler is None))

    for epoch in range(cfg["train"]["epochs"]):
        last_loss = None
        for batch in dl:
            ee = batch["ee_embedding"].to(device)
            text = batch["text_embedding"].to(device)
            coverage = batch["coverage"].to(device)

            ee_z, text_z = model(ee, text)
            loss = criterion(ee_z, text_z, weights=coverage)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        print(f"epoch {epoch + 1}/{cfg['train']['epochs']}: loss={last_loss:.4f}")


def smoke_test() -> None:
    """CPU-Smoke-Test mit Dummy-Tensoren: prueft nur, dass Forward/Loss/Backward durchlaufen -
    KEINE echten Daten."""
    torch.manual_seed(0)
    batch_size, ee_dim, text_dim, latent_dim = 8, 128, 1024, 256

    model = TwoTower(ee_dim=ee_dim, text_dim=text_dim, latent_dim=latent_dim)
    criterion = SigLIPLoss()

    ee = torch.randn(batch_size, ee_dim)
    text = torch.randn(batch_size, text_dim)
    coverage = torch.rand(batch_size)

    ee_z, text_z = model(ee, text)
    assert ee_z.shape == (batch_size, latent_dim)
    assert torch.allclose(ee_z.norm(dim=-1), torch.ones(batch_size), atol=1e-5)

    loss = criterion(ee_z, text_z, weights=coverage)
    assert loss.ndim == 0

    loss.backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "Kein Gradient bei den Projektionskoepfen angekommen"

    print(f"Smoke-Test OK: loss={loss.item():.4f}, {len(grads)} Parameter-Tensoren mit Gradient "
          f"(Forward/Loss/Backward laufen durch).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-Tower SigLIP Training")
    parser.add_argument("--config", type=str, default=None, help="Pfad zu configs/*.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Nur Dummy-Forward/Backward pruefen")
    args = parser.parse_args()

    if args.smoke_test or args.config is None:
        smoke_test()
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
