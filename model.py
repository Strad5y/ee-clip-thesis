"""
model.py

Projektionskoepfe fuer das Two-Tower-CLIP-Setup (CLAUDE.md, settled decisions):
Encoder sind FROZEN und liegen ausserhalb dieses Moduls (TESSERA/AlphaEarth-Embeddings kommen aus
spatial_join.py, Text-Embeddings aus compute_text_embeddings.py) - hier werden nur die
trainierbaren MLP-Projektionskoepfe auf den gemeinsamen latenten Raum d=256 definiert.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear -> L2-normalize (settled decision, CLAUDE.md)."""

    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TwoTower(nn.Module):
    """Ein Backbone (TESSERA oder AlphaEarth) x Text - pro Backbone ein eigenes Modell
    (settled decision: separate Two-Tower-Modelle pro Backbone, kein Weight-Sharing)."""

    def __init__(self, ee_dim: int, text_dim: int = 1024, latent_dim: int = 256,
                 hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.ee_head = ProjectionHead(ee_dim, latent_dim, hidden_dim, dropout)
        self.text_head = ProjectionHead(text_dim, latent_dim, hidden_dim, dropout)

    def forward(self, ee_embedding: torch.Tensor, text_embedding: torch.Tensor):
        return self.ee_head(ee_embedding), self.text_head(text_embedding)


def _smoke_test():
    torch.manual_seed(0)
    batch_size, ee_dim, text_dim, latent_dim = 8, 128, 1024, 256

    head = ProjectionHead(ee_dim, latent_dim)
    x = torch.randn(batch_size, ee_dim)
    z = head(x)
    assert z.shape == (batch_size, latent_dim), z.shape
    assert torch.allclose(z.norm(dim=-1), torch.ones(batch_size), atol=1e-5)

    model = TwoTower(ee_dim=ee_dim, text_dim=text_dim, latent_dim=latent_dim)
    ee = torch.randn(batch_size, ee_dim)
    text = torch.randn(batch_size, text_dim)
    ee_z, text_z = model(ee, text)
    assert ee_z.shape == (batch_size, latent_dim) == text_z.shape
    assert torch.allclose(ee_z.norm(dim=-1), torch.ones(batch_size), atol=1e-5)
    assert torch.allclose(text_z.norm(dim=-1), torch.ones(batch_size), atol=1e-5)

    print(f"Smoke-Test OK: ProjectionHead {ee_dim}->{latent_dim}, TwoTower-Output L2-normalisiert.")


if __name__ == "__main__":
    _smoke_test()
