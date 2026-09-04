"""Stylized-facts statistics and real-data benchmark (Cont, 2001).

1. `compute_stylized_facts(data, lags)`: per-window statistics for one
   (n_windows, L) set -- moments, tail quantiles, ACF of returns, |returns|
   and squared returns, leverage correlation corr(r_t, |r_{t+k}|), and the
   Jarque-Bera and ARCH-LM tests reported as 5% rejection rates across
   windows (p-values are never averaged).

2. `RealBenchmark`: computed once on the disjoint real chunks and cached to
   JSON. Stores the across-chunk mean and standard deviation of every
   (statistic, lag), so every model is compared against identical reference
   values.

3. `compare_to_benchmark(generated, benchmark)`: one table per model,
       statistic | lag | real_mean | real_std | generated | z,
   with z = (generated - real_mean) / real_std.

4. `plot_stylized_curves(chunks, generated, ...)`: the three diagnostic
   curves (ACF of returns, ACF of |returns|, leverage) over lags 1-50 with
   the real mean and a +/-2 sd band across chunks.

All inputs are (n, L) arrays on the same standardised scale. Error bars are
meaningful only for disjoint chunks (overlapping windows give correlated
statistics and understate the spread).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

try:
    from statsmodels.stats.diagnostic import het_arch
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

from scipy.stats import jarque_bera


# ===========================================================================
# Core per-set statistics (vectorized over windows)
# ===========================================================================
def _to_np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x,
                      dtype=np.float64)


def _acf_at_lag(x, lag):
    """Per-window ACF at one lag. x: (n, L) demeaned -> (n,)."""
    var = (x ** 2).mean(axis=1) + 1e-12
    cov = (x[:, :-lag] * x[:, lag:]).mean(axis=1)
    return cov / var


def _leverage_at_lag(x, lag):
    """Per-window corr(r_t, |r_{t+lag}|). x: (n, L) raw -> (n,)."""
    a = np.abs(x)
    xm = x.mean(axis=1, keepdims=True)
    am = a.mean(axis=1, keepdims=True)
    xs = x.std(axis=1) + 1e-12
    as_ = a.std(axis=1) + 1e-12
    cov = ((x[:, :-lag] - xm) * (a[:, lag:] - am)).mean(axis=1)
    return cov / (xs * as_)


def compute_stylized_facts(data, lags=(1, 5, 10, 20),
                           arch_subsample=200, seed=0):
    """Statistics for one (n_windows, L) set.

    Returns dict: {stat_name: value} where lag-dependent stats are keyed
    as f"{name}@lag{k}". Test-based stats are 5% rejection rates.
    """
    x = _to_np(data)
    n, L = x.shape
    rng = np.random.default_rng(seed)

    out = {}

    # ---- lag-free: moments & tails (per window, then averaged) -----------
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-12
    z = (x - mu) / sd
    out["mean"] = x.mean()
    out["std"] = sd.mean()
    out["skewness"] = (z ** 3).mean(axis=1).mean()
    out["excess_kurtosis"] = ((z ** 4).mean(axis=1) - 3.0).mean()
    q01 = np.quantile(x, 0.01, axis=1)
    q99 = np.quantile(x, 0.99, axis=1)
    out["left_tail_1pct"] = q01.mean()
    out["right_tail_99pct"] = q99.mean()
    out["tail_ratio"] = (np.abs(q01) / (np.abs(q99) + 1e-12)).mean()

    # ---- lag-dependent: ACFs & leverage -----------------------------------
    xd = x - mu
    xabs = np.abs(x)
    xabs_d = xabs - xabs.mean(axis=1, keepdims=True)
    x2 = x ** 2
    x2_d = x2 - x2.mean(axis=1, keepdims=True)

    for k in lags:
        if k >= L:
            continue
        out[f"acf_return@lag{k}"] = _acf_at_lag(xd, k).mean()
        out[f"acf_abs_return@lag{k}"] = _acf_at_lag(xabs_d, k).mean()
        out[f"acf_sq_return@lag{k}"] = _acf_at_lag(x2_d, k).mean()
        out[f"leverage@lag{k}"] = _leverage_at_lag(x, k).mean()

    # ---- tests as rejection rates ------------------------------------------
    jb_rej = [jarque_bera(x[i]).pvalue < 0.05 for i in range(n)]
    out["jb_reject_5pct"] = float(np.mean(jb_rej))

    if _HAS_STATSMODELS:
        idx = (rng.choice(n, size=min(arch_subsample, n), replace=False)
               if n > arch_subsample else np.arange(n))
        rej = []
        for i in idx:
            try:
                p = het_arch(x[i], nlags=min(max(lags), L // 4))[1]
                rej.append(p < 0.05)
            except Exception:
                pass
        out["arch_lm_reject_5pct"] = float(np.mean(rej)) if rej else np.nan
    else:
        out["arch_lm_reject_5pct"] = np.nan

    return out


# ===========================================================================
# Real benchmark: compute once, reuse for every model
# ===========================================================================
class RealBenchmark:
    """Across-chunk mean/std of every statistic, computed on disjoint chunks."""

    def __init__(self, mean, std, n_chunks, lags, chunk_shape):
        self.mean = mean            # dict stat -> float
        self.std = std              # dict stat -> float
        self.n_chunks = n_chunks
        self.lags = list(lags)
        self.chunk_shape = list(chunk_shape)

    @classmethod
    def from_chunks(cls, chunks, lags=(1, 5, 10, 20), **kw):
        rows = [compute_stylized_facts(c, lags=lags, **kw) for c in chunks]
        df = pd.DataFrame(rows)
        return cls(
            mean=df.mean().to_dict(),
            std=df.std(ddof=1).to_dict(),
            n_chunks=len(chunks),
            lags=lags,
            chunk_shape=tuple(_to_np(chunks[0]).shape),
        )

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        payload = {
            "mean": self.mean, "std": self.std,
            "n_chunks": self.n_chunks, "lags": self.lags,
            "chunk_shape": self.chunk_shape,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Benchmark saved to {Path(path).resolve()}")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            p = json.load(f)
        return cls(p["mean"], p["std"], p["n_chunks"], p["lags"],
                   p["chunk_shape"])


# ===========================================================================
# Per-model comparison
# ===========================================================================
def compare_to_benchmark(generated, benchmark, model_name="model", **kw):
    """One comparison table: statistic | real_mean | real_std | generated | z.

    NOTE on window-count mismatch: real_std is the spread of chunk-level
    statistics at the benchmark's chunk size. If `generated` has a different
    number of windows, z remains a useful effect-size gauge but is not an
    exact test statistic.
    """
    gen_shape = tuple(_to_np(generated).shape)
    if gen_shape[1] != benchmark.chunk_shape[1]:
        raise ValueError(
            f"Window length mismatch: generated L={gen_shape[1]} vs "
            f"benchmark L={benchmark.chunk_shape[1]}."
        )

    gen_stats = compute_stylized_facts(generated, lags=benchmark.lags, **kw)

    rows = []
    for stat, gval in gen_stats.items():
        rm = benchmark.mean.get(stat, np.nan)
        rs = benchmark.std.get(stat, np.nan)
        z = (gval - rm) / (rs + 1e-12) if np.isfinite(rs) else np.nan
        # split "name@lagK" into columns
        name, _, lag = stat.partition("@lag")
        rows.append({
            "statistic": name,
            "lag": int(lag) if lag else None,
            "real_mean": rm, "real_std": rs,
            model_name: gval, "z": z,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["statistic", "lag"], na_position="first",
                        ignore_index=True)
    return df


def multi_model_summary(tables):
    """Combine {model_name: comparison_df} into one wide |z| table for a
    quick which-model-is-closest view."""
    out = None
    for name, df in tables.items():
        cols = df[["statistic", "lag", "z"]].rename(columns={"z": name})
        out = cols if out is None else out.merge(cols, on=["statistic", "lag"])
    return out


# ===========================================================================
# Diagnostic curves (real band vs generated)
# ===========================================================================
def _curve_band(chunks, fn, max_lag):
    per = np.stack([fn(_to_np(c), max_lag) for c in chunks])
    return per.mean(0), per.std(0, ddof=1)


def _acf_curve(x, max_lag, transform=None):
    if transform is not None:
        x = transform(x)
    x = x - x.mean(axis=1, keepdims=True)
    return np.array([1.0] + [_acf_at_lag(x, k).mean()
                             for k in range(1, max_lag + 1)])


def _lev_curve(x, max_lag):
    return np.array([_leverage_at_lag(x, k).mean()
                     for k in range(1, max_lag + 1)])


def plot_stylized_curves(chunks, generated, max_lag=50, model_name="generated",
                         save_path=None, show=True):
    acf_r_m, acf_r_s = _curve_band(chunks, lambda x, m: _acf_curve(x, m), max_lag)
    acf_a_m, acf_a_s = _curve_band(
        chunks, lambda x, m: _acf_curve(x, m, transform=np.abs), max_lag)
    lev_m, lev_s = _curve_band(chunks, _lev_curve, max_lag)

    g = _to_np(generated)
    acf_r_g = _acf_curve(g, max_lag)
    acf_a_g = _acf_curve(g, max_lag, transform=np.abs)
    lev_g = _lev_curve(g, max_lag)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
    lags1 = np.arange(1, max_lag + 1)

    def panel(ax, x, rm, rs, gv, title):
        ax.fill_between(x, rm - 2 * rs, rm + 2 * rs, color="steelblue",
                        alpha=0.25, label="real ±2sd")
        ax.plot(x, rm, color="steelblue", lw=1.5, label="real mean")
        ax.plot(x, gv, color="crimson", lw=1.5, label=model_name)
        ax.axhline(0, color="gray", lw=0.7, alpha=0.6)
        ax.set_title(title); ax.set_xlabel("lag"); ax.grid(alpha=0.3)

    panel(axes[0], lags1, acf_r_m[1:], acf_r_s[1:], acf_r_g[1:],
          "ACF of returns")
    panel(axes[1], lags1, acf_a_m[1:], acf_a_s[1:], acf_a_g[1:],
          "ACF of |returns| (vol clustering)")
    panel(axes[2], lags1, lev_m, lev_s, lev_g,
          "Leverage: corr(r_t, |r_{t+k}|)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Stylized facts: real vs {model_name}", y=1.02)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)

