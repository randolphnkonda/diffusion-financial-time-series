"""Score network s_theta(x_t, t) for the VE-SDE.

A WaveNet/DiffWave-style residual stack with a temporal transformer in each
block (the CSDI residual block), adapted to univariate return windows.

Pipeline:

    Input (1,L)
      -> Conv1d (1 -> C)                                  input_projection
      -> [ x N gated residual blocks ]                    ResidualBlock
           (1) + diffusion-step embedding   @ (C, L)
           (2) + positional embedding -> Temporal Transformer
           (3) Conv1d (C -> 2C)                           mid_projection
           (4) + time embedding             @ (2C, L)
           (5) sigmoid(gate) * tanh(filter) @ (2C -> C)   gated activation
           (6) Conv1d (C -> 2C)                           output_projection
           (7) split -> residual (to next block) + skip (to output aggregation)
      -> sum of skips / sqrt(N)
      -> Conv1d (C -> C) -> Conv1d (C -> 1)
    Output (1,L)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CHANNELS, DIFFUSION_DIM, NUM_HEADS, NUM_RES_BLOCKS, device
from .utils import DiffusionEmbedding, PositionalEncoding, TimeEmbedding
from .residual_block import ResidualBlock


# ---------------------------------------------------------------------------
# Score network
# ---------------------------------------------------------------------------
class ScoreNet(nn.Module):
    """CSDI-style score network adapted for univariate time series."""

    def __init__(self, channels=CHANNELS, diffusion_dim=DIFFUSION_DIM,
                 num_heads=NUM_HEADS, num_blocks=NUM_RES_BLOCKS, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.num_blocks = num_blocks

        # (1, L) -> (C, L)
        self.input_projection = nn.Conv1d(1, channels, kernel_size, padding=padding)

        self.diffusion_embedding  = DiffusionEmbedding(diffusion_dim)
        self.positional_embedding = PositionalEncoding(channels)
        self.time_embedding       = TimeEmbedding(2 * channels)

        self.blocks = nn.ModuleList([
            ResidualBlock(channels, diffusion_dim, num_heads, kernel_size)
            for _ in range(num_blocks)
        ])

        # skip aggregation head: (C, L) -> (1, L)
        self.skip_projection   = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.output_projection = nn.Conv1d(channels, 1, kernel_size, padding=padding)
        nn.init.zeros_(self.output_projection.weight)   # common score-net init
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x, t):
        """
        x : (B, L) noisy series
        t : (B,)   continuous diffusion time (e.g. ~U[0, 1]); conditions the net
        returns: (B, L) estimated score
        """
        B, L = x.shape
        dev = x.device

        x = x.unsqueeze(1)                              # (B, 1, L)
        x = F.relu(self.input_projection(x))            # (B, C, L)

        diff_emb = self.diffusion_embedding(t)          # (B, diffusion_dim)
        pos_emb  = self.positional_embedding(L, dev)    # (1, C, L)
        time_emb = self.time_embedding(L, dev)          # (1, 2C, L)

        skip_total = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x, diff_emb, pos_emb, time_emb)
            skip_total = skip_total + skip

        out = skip_total / math.sqrt(self.num_blocks)   # skip aggregation, scaled by 1/sqrt(n)
        out = F.relu(self.skip_projection(out))         # (B, C, L)
        out = self.output_projection(out)               # (B, 1, L)
        return out.squeeze(1)                           # (B, L)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = ScoreNet().to(device)
    B, L = 4, 128
    x = torch.randn(B, L, device=device)
    t = torch.rand(B, device=device)               # continuous diffusion time ~U[0,1]
    y = net(x, t)
    assert y.shape == (B, L), y.shape
    n_params = sum(p.numel() for p in net.parameters())
    print(f"output shape: {tuple(y.shape)}  |  parameters: {n_params:,}")