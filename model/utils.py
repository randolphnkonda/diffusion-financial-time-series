"""Embedding modules for the score network.

- GaussianFourierProjection / DiffusionEmbedding: embed the continuous
  diffusion time t (the noise-level conditioning of the score network).
- PositionalEncoding: learnable position embedding added before the
  temporal transformer.
- TimeEmbedding: sinusoidal + learnable embedding of sequence position,
  added after the mid projection (the second conditioning injection).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CHANNELS, DIFFUSION_DIM, MAX_LEN


# ---------------------------------------------------------------------------
# Sinusoidal helper (shared by the sequence-position embeddings)
# ---------------------------------------------------------------------------
def sinusoidal_embedding(positions, dim):
    """Integer-position sinusoidal features (base-10000), as in vanilla
    Transformers. Used for the *sequence-position* embeddings (time/positional)
    along L. NOT suitable for continuous diffusion time -- see
    GaussianFourierProjection for that.

    positions: (...,) tensor -> (..., dim) sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=positions.device).float() / half
    )                                              # (half,)
    args = positions.float().unsqueeze(-1) * freqs  # (..., half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:                                # pad if dim is odd
        emb = F.pad(emb, (0, 1))
    return emb                                      # (..., dim)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class GaussianFourierProjection(nn.Module):
    """Continuous-time noise conditioning (NCSN++ / Song et al.).

    Fixed random Gaussian frequencies sized for t in ~[0, 1]. Unlike the
    base-10000 integer table, this retains resolution across a small continuous
    range, so nearby diffusion times map to distinguishable embeddings.
    """

    def __init__(self, dim=DIFFUSION_DIM, scale=16.0):
        super().__init__()
        # random frequencies, fixed (not trained)
        self.register_buffer("W", torch.randn(dim // 2) * scale)

    def forward(self, t):                            # t: (B,) continuous in ~[0,1]
        proj = t[:, None] * self.W[None, :] * 2 * math.pi   # (B, dim/2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, dim)


class DiffusionEmbedding(nn.Module):
    """Continuous diffusion time t -> DIFFUSION_DIM vector (shared across blocks).

    Condition on t (the uniform draw), not raw sigma(t): for a VE-SDE sigma is
    monotonic in t, so t is the better-conditioned input. If you must condition
    on the noise level, pass normalized log-sigma instead of sigma.
    """

    def __init__(self, dim=DIFFUSION_DIM, scale=16.0):
        super().__init__()
        self.dim = dim
        self.fourier = GaussianFourierProjection(dim, scale)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)

    def forward(self, t):                            # t: (B,) continuous
        emb = self.fourier(t)                        # (B, dim)
        emb = F.silu(self.proj1(emb))
        emb = F.silu(self.proj2(emb))
        return emb                                   # (B, dim)


class PositionalEncoding(nn.Module):
    """Learnable position embedding, added BEFORE the transformer -> (1, C, L)."""

    def __init__(self, channels=CHANNELS, max_len=MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(max_len, channels)

    def forward(self, length, device):
        pos = torch.arange(length, device=device)
        emb = self.embed(pos)                        # (L, C)
        return emb.transpose(0, 1).unsqueeze(0)      # (1, C, L)


class TimeEmbedding(nn.Module):
    """Absolute temporal positions: sinusoidal + learnable -> (1, 2C, L)."""

    def __init__(self, dim=2 * CHANNELS, max_len=MAX_LEN):
        super().__init__()
        self.dim = dim
        self.learnable = nn.Embedding(max_len, dim)

    def forward(self, length, device):
        pos = torch.arange(length, device=device)
        sinus = sinusoidal_embedding(pos, self.dim)  # (L, 2C)
        learn = self.learnable(pos)                  # (L, 2C)
        emb = sinus + learn                          # (L, 2C)
        return emb.transpose(0, 1).unsqueeze(0)      # (1, 2C, L)

