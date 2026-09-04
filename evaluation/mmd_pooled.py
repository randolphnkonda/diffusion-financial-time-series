"""Supplementary MMD diagnostic against the pooled real data.

The chunk battery compares generated samples against one real chunk at a
time and averages. This module instead compares a generated set against all
real windows pooled together, and returns the three expectation terms of
the MMD^2 estimator (xx, yy, xy) separately, so that a large MMD can be
attributed to within-generated self-similarity (the yy term) or to the
cross term.

Implementation notes:
    1. The real-real kernel over the full pool is large, so the terms are
       accumulated block-wise: same estimator, same numbers, bounded memory.
    2. Bandwidths are estimated from the pooled real data exactly as in
       `mmd_battery`, so pooled and per-chunk values share the kernel.

Usage:
    out = mmd_pooled_check(chunks, generated)
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from .utils import rbf_kernel, estimate_median_bandwidth
    _FALLBACK = False
except ImportError:                                    # standalone use
    print("Failed to import rbf_kernel, estimate_median_bandwidth from utils")
    _FALLBACK = True

    def rbf_kernel(a, b, bandwidths):
        """Fallback multi-scale RBF: SUM of exp(-d^2 / (2 s^2)) over
        bandwidths. WARNING: if your project rbf_kernel averages instead
        of summing, absolute values differ by a constant factor from the
        battery tables (rankings unaffected). Prefer the project import.
        """
        d2 = torch.cdist(a, b).pow(2)
        k = torch.zeros_like(d2)
        for s in bandwidths:
            k = k + torch.exp(-d2 / (2.0 * s * s))
        return k

    def estimate_median_bandwidth(x, y, max_samples=2000, seed=42):
        g = torch.Generator().manual_seed(seed)
        pool = torch.cat([x, y], dim=0)
        if len(pool) > max_samples:
            idx = torch.randperm(len(pool), generator=g)[:max_samples]
            pool = pool[idx]
        d = torch.cdist(pool, pool)
        return d[d > 0].median().item()


def _to_tensor(x):
    return x.detach().cpu().float() if torch.is_tensor(x) \
        else torch.as_tensor(np.asarray(x, dtype=np.float32))


def mmd2_terms_blocked(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: Sequence[float],
    block: int = 2000,
) -> Tuple[float, float, float, float]:
    """Unbiased, UNCLAMPED MMD^2 with the three expectation terms,
    accumulated block-wise so pooled-scale inputs fit in memory.

    Returns (mmd2, xx_term, yy_term, xy_term); identical estimator to
    maximum_mean_discrepancy(..., return_squared=True).
    """
    x, y = _to_tensor(x), _to_tensor(y)
    n, m = x.shape[0], y.shape[0]

    def offdiag_mean(a):
        """sum of k(a_i, a_j) over i != j, divided by n(n-1)."""
        na = a.shape[0]
        total = 0.0
        diag = 0.0
        for i in range(0, na, block):
            ai = a[i:i + block]
            for j in range(0, na, block):
                aj = a[j:j + block]
                k = rbf_kernel(ai, aj, bandwidths)
                total += k.sum().item()
                if i == j:
                    diag += k.diagonal().sum().item()
        return (total - diag) / (na * (na - 1))

    def cross_mean(a, b):
        total = 0.0
        for i in range(0, a.shape[0], block):
            ai = a[i:i + block]
            for j in range(0, b.shape[0], block):
                total += rbf_kernel(ai, b[j:j + block], bandwidths) \
                    .sum().item()
        return total / (a.shape[0] * b.shape[0])

    xx = offdiag_mean(x)
    yy = offdiag_mean(y)
    xy = cross_mean(x, y)
    return xx + yy - 2.0 * xy, xx, yy, xy


def mmd_pooled_check(
    chunks,
    generated,
    seed: int = 42,
    bandwidth_scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    block: int = 2000,
):
    """Pooled-real MMD ladder, printed like mmd_battery.

    Rungs:
        pooled floor    interleaved chunk halves (chunks[0::2] vs
                        chunks[1::2]) so BOTH halves span the full
                        history -- a mixture-vs-mixture null. (A
                        chronological first-half/second-half split would
                        reintroduce the era effect being tested.)
        real-generated  all pooled real windows vs the generated set
        real-normal     pooled real vs N(0, I) shaped like generated
        real-uniform    pooled real vs variance-matched uniform

    Returns dict {rung: (mmd2, xx, yy, xy)}.
    """
    rng = np.random.default_rng(seed)
    chunks = [_to_tensor(c) for c in chunks]
    gen = _to_tensor(generated)
    pool = torch.cat(chunks, dim=0)

    sigma = estimate_median_bandwidth(pool, pool, max_samples=2000,
                                      seed=seed)
    bandwidths = [sigma * s for s in bandwidth_scales]
    tag = " (FALLBACK rbf: absolute scale may differ)" if _FALLBACK else ""
    print(f"[mmd-pooled] sigma = {sigma:.4f}; "
          f"bandwidths = {[f'{b:.3f}' for b in bandwidths]}{tag}")

    half_a = torch.cat(chunks[0::2], dim=0)
    half_b = torch.cat(chunks[1::2], dim=0)

    def noise_like(kind):
        if kind == "normal":
            return torch.randn(gen.shape)
        bound = 3.0 ** 0.5
        return torch.empty(gen.shape).uniform_(-bound, bound)

    results = {}
    rows = [
        ("pooled floor",   half_a, half_b),
        ("real-generated", pool,   gen),
        ("real-normal",    pool,   noise_like("normal")),
        ("real-uniform",   pool,   noise_like("uniform")),
    ]

    print(f"\n{'comparison':<16s} {'MMD2':>12s} {'xx':>10s} "
          f"{'yy':>10s} {'xy':>10s}")
    print("-" * 62)
    for label, a, b in rows:
        mmd2, xx, yy, xy = mmd2_terms_blocked(a, b, bandwidths, block)
        results[label] = (mmd2, xx, yy, xy)
        print(f"{label:<16s} {mmd2:>12.8f} {xx:>10.6f} "
              f"{yy:>10.6f} {xy:>10.6f}")

    print("\nRead-out: if a D1 model's real-generated MMD2 here is LOWER "
          "than its D2/D3\ncounterparts' (opposite of the per-chunk "
          "table), the inversion is a protocol\nartefact. Compare yy "
          "across models to test the diversity account.")
    return results