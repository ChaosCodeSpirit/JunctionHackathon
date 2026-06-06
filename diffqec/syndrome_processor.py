"""Syndrome processor: spatial conv + temporal GRU."""
import torch
from torch import nn


class SyndromeProcessor(nn.Module):
    """Encodes a syndrome history into a fixed-size feature vector c.

    Accepts flattened spatial syndromes (batch, rounds, D) and applies
    a 1-D convolution over the spatial dimension per round, followed by
    per-round pooling and a GRU over rounds.
    """

    def __init__(
        self,
        in_spatial: int,
        hidden: int = 64,
        out_feature: int = 64,
        num_conv_layers: int = 2,
    ):
        super().__init__()
        self.in_spatial = in_spatial
        self.hidden = hidden
        self.out_feature = out_feature

        layers = []
        in_ch = 1
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
            in_ch = hidden
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.spatial_conv = nn.Sequential(*layers)

        self.round_proj = nn.Linear(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out_proj = nn.Linear(hidden, out_feature)

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        syndrome : (batch, rounds, D) float — detection events per round.

        Returns
        -------
        c : (batch, out_feature) float
        """
        b, r, d = syndrome.shape
        # (batch*rounds, 1, D) for 1D conv over spatial dimension
        x = syndrome.view(b * r, 1, d)
        x = self.spatial_conv(x)           # (batch*rounds, hidden, 1)
        x = x.squeeze(-1)                  # (batch*rounds, hidden)
        x = x.view(b, r, self.hidden)
        x = self.round_proj(x)
        x, _ = self.gru(x)                 # (batch, rounds, hidden)
        # Use final hidden state
        c = x[:, -1, :]                    # (batch, hidden)
        c = self.out_proj(c)               # (batch, out_feature)
        return c
