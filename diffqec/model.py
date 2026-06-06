"""Top-level DiffQEC model."""
import math
import torch
import torch.nn.functional as F
from torch import nn

from .syndrome_processor import SyndromeProcessor
from .denoiser import DenoiserStack


class DiffQEC(nn.Module):
    """Discrete denoising-diffusion decoder for surface codes.

    Parameters
    ----------
    L : int
        Number of logical observables.
    syndrome_shape : tuple
        (rounds, D) or (rounds, H, W) — expected syndrome dimensions.
        For the flattened case D is used.
    hidden : int
        Embedding / hidden dimension.
    num_denoiser_layers : int
        Depth of the SFM denoiser stack.
    dropout : float
    """

    def __init__(
        self,
        L: int,
        syndrome_shape: tuple,
        hidden: int = 64,
        num_denoiser_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.L = L
        self.syndrome_shape = syndrome_shape
        if len(syndrome_shape) == 2:
            rounds, D = syndrome_shape
            self.syndrome_proc = SyndromeProcessor(in_spatial=D, hidden=hidden, out_feature=hidden)
        else:
            raise ValueError("syndrome_shape must be (rounds, D)")

        # Embedding for noisy state x_t
        self.x_embed = nn.Linear(L, hidden)
        # Timestep embedding (sinusoidal)
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Syndrome feature already projected to hidden by processor

        self.denoiser = DenoiserStack(
            feature_dim=hidden,
            cond_dim=hidden,
            num_layers=num_denoiser_layers,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(hidden, L * 2)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        syndrome: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xt       : (batch, L) float {0,1}
        t        : (batch,) int  — diffusion timestep
        syndrome : (batch, rounds, D) float

        Returns
        -------
        logits : (batch, L, 2) float — logits for each bit's two classes
        """
        b = xt.size(0)
        e_x = self.x_embed(xt)                     # (batch, hidden)
        e_t = self.t_embed(t.float())              # (batch, hidden)
        c = self.syndrome_proc(syndrome)           # (batch, hidden)
        h = e_x + e_t + c                          # (batch, hidden)
        h = self.denoiser(h, c)                    # (batch, hidden)
        logits = self.out_proj(h).view(b, self.L, 2)
        return logits


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


