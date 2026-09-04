"""Gated residual block of the score network.

One block of the residual stack, following the CSDI architecture
(Tashiro et al., 2021): diffusion-step conditioning, a temporal transformer layer with
positional encoding, a gated activation unit (tanh filter modulated by a
sigmoid gate), and a final convolution that splits the output into a
residual path (fed to the next block) and a skip path (aggregated at the
network output).
"""

import math

import torch
import torch.nn as nn

from .config import CHANNELS, DIFFUSION_DIM, NUM_HEADS
from .temporal_transformer import TemporalTransformer


class ResidualBlock(nn.Module):
    """Residual block: conditioning -> attention -> gated unit -> residual/skip split.

    Input  x        : (B, C, L)
    Returns (x_next, skip), both (B, C, L).
    """

    def __init__(self, channels=CHANNELS, diffusion_dim=DIFFUSION_DIM,
                 num_heads=NUM_HEADS, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.diffusion_projection = nn.Linear(diffusion_dim, channels)
        self.transformer = TemporalTransformer(channels, num_heads)
        self.mid_projection    = nn.Conv1d(channels, 2 * channels, kernel_size, padding=padding)
        self.output_projection = nn.Conv1d(channels, 2 * channels, kernel_size, padding=padding)

    def forward(self, x, diff_emb, pos_emb, time_emb):
        # x: (B,C,L) | diff_emb: (B,diffusion_dim) | pos_emb: (1,C,L) | time_emb: (1,2C,L)

        # (1) diffusion-step conditioning  ->  first  (+)  at (C, L)
        d = self.diffusion_projection(diff_emb).unsqueeze(-1)   # (B, C, 1)
        h = x + d                                               # (B, C, L)

        # (2) positional encoding + temporal transformer
        h = h + pos_emb                                         # (B, C, L)
        h = self.transformer(h)                                 # (B, C, L)

        # (3) mid projection (C -> 2C)
        h = self.mid_projection(h)                              # (B, 2C, L)

        # (4) time-embedding conditioning  ->  second (+)  at (2C, L)
        h = h + time_emb                                        # (B, 2C, L)

        # (5) gated activation:  sigmoid(gate) * tanh(filter)
        gate, filt = h.chunk(2, dim=1)                          # each (B, C, L)
        h = torch.sigmoid(gate) * torch.tanh(filt)              # (B, C, L)

        # (6) output projection (C -> 2C), then split into residual + skip
        h = self.output_projection(h)                           # (B, 2C, L)
        residual, skip = h.chunk(2, dim=1)                      # each (B, C, L)

        # (7) residual rejoins block input (scaled by 1/sqrt(2)); skip is aggregated at the output
        return (x + residual) / math.sqrt(2.0), skip
