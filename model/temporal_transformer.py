"""Temporal transformer layer used inside each residual block.

Multi-head self-attention over the time axis of a (B, C, L) feature map,
treating the channel dimension C as the model dimension. Pre-norm blocks
with a position-wise feed-forward network, as in Vaswani et al. (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CHANNELS, NUM_HEADS, NUM_TF_LAYERS


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, d_model, head_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):                            # (B, T, d_model)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        return wei @ v                               # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Multiple heads in parallel."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([Head(d_model, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)      # mix across heads

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TemporalTransformer(nn.Module):
    """Input/Output (B, C, L). Treats C as the model dim and attends over L."""

    def __init__(self, d_model=CHANNELS, num_heads=NUM_HEADS, num_layers=NUM_TF_LAYERS):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(d_model, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x):                            # (B, C, L)
        x = x.transpose(1, 2)                        # (B, L, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x.transpose(1, 2)                     # (B, C, L)
