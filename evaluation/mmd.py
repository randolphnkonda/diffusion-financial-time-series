"""Maximum mean discrepancy (MMD) evaluation.

`maximum_mean_discrepancy` computes the unbiased empirical MMD^2 estimator
of Gretton et al. (2012) with a multi-scale Gaussian kernel. `mmd_battery` evaluates it over
the shared disjoint chunks: a real-vs-real floor from disjoint chunk pairs,
real-vs-noise references, and real-vs-generated, each reported with a
z-score against the floor.

MMD^2 is reported unclamped: the unbiased estimator fluctuates around zero
under the null, so a slightly negative floor is expected. The estimator is
unbiased at any sample size, so the full generated set is compared against
each chunk. Bandwidths are set once from the pooled real data by the median
heuristic and shared by all comparisons.
"""

from typing import Optional, Sequence

import numpy as np
import torch

from .utils import estimate_median_bandwidth, rbf_kernel


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------
def maximum_mean_discrepancy(
    real: torch.Tensor,
    generated: torch.Tensor,
    bandwidths: Optional[Sequence[float]] = None,
    return_squared: bool = False,
    seed: int = 42,
) -> torch.Tensor:
    """Unbiased empirical MMD between two sample sets with a multi-scale RBF kernel.

    Args:
        real: Real samples with shape [N, D].
        generated: Generated samples with shape [M, D].
        bandwidths: RBF kernel bandwidths. When omitted, bandwidths are selected
          around the median-heuristic value.
        return_squared: Return MMD^2 (unclamped) instead of MMD.
        seed: Seed used for median-bandwidth estimation.

    Returns:
        Scalar MMD or MMD² tensor.
    """
    if real.ndim != 2 or generated.ndim != 2:
        raise ValueError("Inputs must have shape [N, D] and [M, D].")

    if real.shape[1] != generated.shape[1]:
        raise ValueError("Feature dimensions must match.")

    if real.shape[0] < 2 or generated.shape[0] < 2:
        raise ValueError("Each sample must contain at least two points.")

    real = real.float()
    generated = generated.to(
        device=real.device,
        dtype=real.dtype,
    )

    x, y = real, generated

    if bandwidths is None:
        sigma = estimate_median_bandwidth(
            x,
            y,
            max_samples=2000,
            seed=seed,
        )
        bandwidths = [
            sigma / 4.0,
            sigma / 2.0,
            sigma,
            sigma * 2.0,
            sigma * 4.0,
        ]

    k_xx = rbf_kernel(x, x, bandwidths)
    k_yy = rbf_kernel(y, y, bandwidths)
    k_xy = rbf_kernel(x, y, bandwidths)

    n = x.shape[0]
    m = y.shape[0]

    xx_term = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
    yy_term = (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
    xy_term = k_xy.mean()

    mmd_squared = xx_term + yy_term - 2.0 * xy_term

    if return_squared:
        return mmd_squared

    return torch.sqrt(mmd_squared.clamp_min(0.0))


# ---------------------------------------------------------------------------
# Chunk-based evaluation battery
# ---------------------------------------------------------------------------
def _to_tensor(x):
    return x.detach().cpu().float() if torch.is_tensor(x) \
        else torch.as_tensor(np.asarray(x, dtype=np.float32))


def mmd_battery(chunks, generated, seed=42, bandwidth_scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """MMD^2 over the shared disjoint chunks.

    Comparisons: real-vs-real floor (disjoint chunk pairs), real-vs-generated,
    real-vs-standard-normal and real-vs-uniform (unit variance), each per
    chunk. Returns {comparison: np.array of unclamped MMD^2 values} and
    prints a summary table with z-scores against the floor.
    """
    rng = np.random.default_rng(seed)
    chunks = [_to_tensor(c) for c in chunks]
    gen = _to_tensor(generated)
    shape = tuple(chunks[0].shape)

    # ---- bandwidths: once, from pooled real data ---------------------------
    real_pool = torch.cat(chunks, dim=0)
    sigma = estimate_median_bandwidth(real_pool, real_pool,
                                      max_samples=2000, seed=seed)
    bandwidths = [sigma * s for s in bandwidth_scales]
    print(f"[mmd] median-heuristic sigma = {sigma:.4f}; "
          f"bandwidths = {[f'{b:.3f}' for b in bandwidths]}")

    def mmd2(a, b):
        return maximum_mean_discrepancy(
            a, b, bandwidths=bandwidths, return_squared=True
        ).item()                                  # unclamped MMD^2

    def noise_like(kind):
        if kind == "normal":
            return torch.randn(shape)
        bound = 3.0 ** 0.5
        return torch.empty(shape).uniform_(-bound, bound)

    results = {}

    n_pairs = len(chunks) // 2
    results["real-real"] = np.array(
        [mmd2(chunks[2 * k], chunks[2 * k + 1]) for k in range(n_pairs)])

    results["real-generated"] = np.array([mmd2(c, gen) for c in chunks])
    results["real-normal"] = np.array(
        [mmd2(c, noise_like("normal")) for c in chunks])
    results["real-uniform"] = np.array(
        [mmd2(c, noise_like("uniform")) for c in chunks])

    # ---- table --------------------------------------------------------------
    floor = results["real-real"]
    f_std = floor.std(ddof=1) if n_pairs > 1 else np.nan
    print(f"\n{'comparison':<18s} {'MMD2 mean':>12s} {'MMD2 std':>10s} "
          f"{'z':>8s}   n")
    print("-" * 58)
    for label, vals in results.items():
        z = (vals.mean() - floor.mean()) / (f_std + 1e-12)
        print(f"{label:<18s} {vals.mean():>12.8f} "
              f"{vals.std(ddof=1):>10.8f} {z:>+8.2f}   {len(vals)}")
    print("\nNote: negative floor values are expected (unbiased estimator "
          "under the null).")
    return results