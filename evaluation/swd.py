"""Sliced Wasserstein distance evaluation.

`sliced_wasserstein_distance` projects both sample sets onto random unit
directions, computes the one-dimensional Wasserstein-p distance between the
sorted projections, and aggregates as SW_p = (E_theta[W_p^p])^(1/p).
Projection directions are drawn on the CPU from a seeded generator and then
moved to the compute device.

`swd_battery` evaluates SW_1 and SW_2 over the shared disjoint chunks with a
real-vs-real floor, real-vs-noise references and real-vs-generated. The
finite-sample estimate is positively biased with a bias that depends on the
sample size, so every comparison is between equal-sized sets; the generated
set and the noise references are subsampled to the chunk size.
"""

from typing import Optional

import numpy as np
import torch


# ===========================================================================
# Core distance
# ===========================================================================
def sliced_wasserstein_distance(
    real_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    num_projections: int = 1000,
    p: int = 2,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """p-Sliced Wasserstein distance between two equal-sized sample sets.

    real_samples: [N, D], generated_samples: [N, D].
    """
    if real_samples.ndim != 2 or generated_samples.ndim != 2:
        raise ValueError("Both inputs must have shape [num_samples, dimension].")
    if real_samples.shape[1] != generated_samples.shape[1]:
        raise ValueError("Both inputs must have the same feature dimension.")
    if real_samples.shape[0] != generated_samples.shape[0]:
        raise ValueError(
            "Equal sample counts required (finite-sample bias depends on n): "
            f"{real_samples.shape[0]} vs {generated_samples.shape[0]}. "
            "Subsample the larger set."
        )
    if real_samples.shape[0] < 2:
        raise ValueError("Each distribution must contain at least two samples.")
    if num_projections <= 0:
        raise ValueError("num_projections must be positive.")
    if p < 1:
        raise ValueError("p must be at least 1.")

    dtype = real_samples.dtype
    real = real_samples.to(device=device, dtype=dtype)
    gen = generated_samples.to(device=device, dtype=dtype)
    dimension = real.shape[1]

    # Directions on CPU (robust across backends), then moved to device.
    cpu_gen = None
    if seed is not None:
        cpu_gen = torch.Generator()          # CPU generator
        cpu_gen.manual_seed(seed)
    directions = torch.randn(num_projections, dimension,
                             dtype=dtype, generator=cpu_gen)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    directions = directions.to(device)

    real_proj = real @ directions.T          # [N, num_projections]
    gen_proj = gen @ directions.T            # [N, num_projections]

    real_sorted = torch.sort(real_proj, dim=0).values
    gen_sorted = torch.sort(gen_proj, dim=0).values

    projected_costs = torch.mean(torch.abs(real_sorted - gen_sorted) ** p, dim=0)
    return projected_costs.mean().pow(1.0 / p)


def compute_SW(real, generated, projections=2000, seed=42, device="cpu"):
    """Convenience: (SW1, SW2) with shared directions seed."""
    kw = dict(num_projections=projections, seed=seed, device=device)
    return (sliced_wasserstein_distance(real, generated, p=1, **kw),
            sliced_wasserstein_distance(real, generated, p=2, **kw))


# ===========================================================================
# Battery: chunk-based ladder with error bars
# ===========================================================================
def swd_battery(chunks, generated, device="cpu", projections=2000, seed=42):
    """SWD ladder using the shared disjoint chunks.

    Rungs:
        real vs real      -- disjoint chunk pairs (floor)
        real vs generated -- each chunk vs a chunk-sized subsample of generated
        real vs normal    -- each chunk vs standard normal (standardized space)
        real vs uniform   -- each chunk vs variance-matched U(-sqrt(3), sqrt(3))

    All comparisons are chunk-size vs chunk-size (equal n on every rung).
    Returns dict of {rung: {"sw1": array, "sw2": array}} plus prints the table.
    """
    rng = np.random.default_rng(seed)
    n_per = len(chunks[0])
    gen_cpu = generated.detach().cpu()

    def gen_subsample():
        idx = rng.choice(len(gen_cpu), size=n_per, replace=False)
        return gen_cpu[torch.as_tensor(idx)]

    def noise_like(chunk, kind):
        if kind == "normal":
            return torch.randn(chunk.shape)
        bound = 3.0 ** 0.5                      # Var(U(-a,a)) = a^2/3 = 1
        return torch.empty(chunk.shape).uniform_(-bound, bound)

    results = {}

    # ---- floor: disjoint pairs -------------------------------------------
    n_pairs = len(chunks) // 2
    rr = [compute_SW(chunks[2 * k], chunks[2 * k + 1],
                     projections, seed, device) for k in range(n_pairs)]
    results["real-real"] = {"sw1": np.array([a.item() for a, _ in rr]),
                            "sw2": np.array([b.item() for _, b in rr])}

    # ---- other rungs: each chunk vs candidate -----------------------------
    for label, make in [
        ("real-generated", lambda c: gen_subsample()),
        ("real-normal",    lambda c: noise_like(c, "normal")),
        ("real-uniform",   lambda c: noise_like(c, "uniform")),
    ]:
        vals = [compute_SW(c, make(c), projections, seed, device)
                for c in chunks]
        results[label] = {"sw1": np.array([a.item() for a, _ in vals]),
                          "sw2": np.array([b.item() for _, b in vals])}

    # ---- table -------------------------------------------------------------
    floor1 = results["real-real"]["sw1"]
    floor2 = results["real-real"]["sw2"]
    print(f"{'comparison':<18s} {'SW1 mean':>9s} {'SW1 std':>8s} {'z1':>7s} "
          f"{'SW2 mean':>9s} {'SW2 std':>8s} {'z2':>7s}   n")
    print("-" * 78)
    for label, r in results.items():
        z1 = (r["sw1"].mean() - floor1.mean()) / (floor1.std(ddof=1) + 1e-12)
        z2 = (r["sw2"].mean() - floor2.mean()) / (floor2.std(ddof=1) + 1e-12)
        print(f"{label:<18s} {r['sw1'].mean():>9.5f} "
              f"{r['sw1'].std(ddof=1):>8.5f} {z1:>+7.2f} "
              f"{r['sw2'].mean():>9.5f} {r['sw2'].std(ddof=1):>8.5f} "
              f"{z2:>+7.2f}   {len(r['sw1'])}")
    return results