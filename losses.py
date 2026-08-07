"""
losses.py

SigLIP-Loss (Zhai et al. 2023) statt InfoNCE (settled decision, CLAUDE.md): paarweiser
Sigmoid-Verlust ueber die volle Batch-Similarity-Matrix, mit lernbarer Temperatur t und Bias b.
Multi-Positive-faehig ueber eine explizite pos_mask (z.B. mehrere Prompt-Templates pro POI).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    def __init__(self, init_t: float = 10.0, init_b: float = -10.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(math.log(init_t)))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, ee_z: torch.Tensor, text_z: torch.Tensor,
                weights: torch.Tensor | None = None,
                pos_mask: torch.Tensor | None = None) -> torch.Tensor:
        n = ee_z.shape[0]
        logits = ee_z @ text_z.T * self.log_t.exp() + self.b  # (n, n)

        if pos_mask is None:
            pos_mask = torch.eye(n, device=ee_z.device, dtype=torch.bool)
        labels = torch.where(pos_mask, 1.0, -1.0)

        # coverage-Gewicht (settled decision: coverage in [0,1] als Gewicht) wirkt nur auf die
        # positiven Paare - ein Pixel mit geringer POI-Ueberdeckung liefert ein schwaecheres
        # Signal. Annahme, da CLAUDE.md nicht spezifiziert, WORAUF genau das Gewicht wirkt.
        pair_weight = torch.ones_like(logits)
        if weights is not None:
            pair_weight = torch.where(pos_mask, weights.unsqueeze(1).expand(n, n), pair_weight)

        loss = -F.logsigmoid(labels * logits) * pair_weight
        return loss.sum() / n


def _smoke_test():
    torch.manual_seed(0)
    batch_size, latent_dim = 8, 256

    ee_z = F.normalize(torch.randn(batch_size, latent_dim), dim=-1)
    text_z = F.normalize(torch.randn(batch_size, latent_dim), dim=-1)
    coverage = torch.rand(batch_size)

    criterion = SigLIPLoss()
    loss = criterion(ee_z, text_z, weights=coverage)
    assert loss.ndim == 0, "Loss muss ein Skalar sein"
    assert torch.isfinite(loss)

    loss.backward()
    assert criterion.log_t.grad is not None and criterion.b.grad is not None

    print(f"Smoke-Test OK: SigLIP-Loss={loss.item():.4f} (Skalar), Gradienten fuer t/b vorhanden.")


if __name__ == "__main__":
    _smoke_test()
