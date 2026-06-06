"""Denoiser stack with Syndrome Feature Modulation (SFM)."""
import torch
from torch import nn


class SFMLayer(nn.Module):
    """FiLM-style scale+shift produced from the syndrome feature c."""

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feature_dim)
        self.shift = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, feature_dim)
        c : (batch, cond_dim)
        """
        gamma = self.scale(c)  # (batch, feature_dim)
        beta = self.shift(c)   # (batch, feature_dim)
        return gamma * x + beta


class GatedResidualBlock(nn.Module):
    """Linear layer + SFM + gated residual pathway."""

    def __init__(self, feature_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.sfm = SFMLayer(feature_dim, cond_dim)
        self.gate = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.act(h)
        h = self.sfm(h, c)
        h = self.dropout(h)
        g = torch.sigmoid(self.gate(x))
        return (1 - g) * x + g * h


class DenoiserStack(nn.Module):
    """Stack of gated residual SFM layers."""

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedResidualBlock(feature_dim, cond_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, c)
        return x
