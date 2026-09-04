"""Marginal and per-window distribution diagnostics.

Both entry points accept `generated` either as a single (n, L) tensor/array
(with `gen_label`) or as a dict {label: (n, L) set} of up to five models.
Real data are drawn as a filled histogram and each model as a step outline.

1. plot_marginals(real, generated, ...)
   Four panels on the pooled values (all windows flattened):
     (a) density overlay             -- overall shape and dispersion
     (b) log-density                 -- low-probability (tail) regions
     (c) QQ plot of each model vs real
     (d) empirical survival P(|r| >= x) on log-log axes -- tail decay
   With `standardize_tails=True`, each series is z-scored before panels
   (c) and (d) so that shape and tail behaviour are compared free of scale.

2. plot_window_stats(chunks, generated, ...)
   Distributions across windows of the per-window standard deviation,
   excess kurtosis and lag-1 autocorrelation of absolute returns.

Inputs are on the same standardised scale.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REAL_C = "silver"
MODEL_COLORS = ["crimson", "darkorange", "teal", "purple", "olive"]


def _to_np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x,
                      dtype=np.float64)


def _as_models(generated, gen_label):
    """Normalize `generated` to an ordered {label: np.ndarray} dict."""
    if isinstance(generated, dict):
        models = {k: _to_np(v) for k, v in generated.items()}
    else:
        models = {gen_label: _to_np(generated)}
    if not models:
        raise ValueError("generated must contain at least one model.")
    if len(models) > len(MODEL_COLORS):
        raise ValueError(f"At most {len(MODEL_COLORS)} models per figure.")
    return models


# ===========================================================================
# 1. Pooled marginals and tails
# ===========================================================================
def plot_marginals(real, generated, gen_label="generated",
                   save_path=None, show=True, bins=120, title="auto",
                   standardize_tails=False):
    """standardize_tails : when True, z-score EACH series (real and every
    model, by its own pooled mean/std) before the QQ and survival panels,
    so those two panels compare SHAPE and tail decay free of scale/location
    mismatch. The density panels (a, b) always use the raw values, where a
    scale mismatch is itself the finding. Panel titles note the mode."""
    r = _to_np(real).ravel()
    models = {lb: g.ravel() for lb, g in _as_models(generated, gen_label).items()}

    def _z(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    if standardize_tails:
        r_t = _z(r)
        models_t = {lb: _z(g) for lb, g in models.items()}
        tail_note = " (standardized)"
    else:
        r_t, models_t = r, models
        tail_note = ""

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), dpi=150)

    everything = np.concatenate([r] + list(models.values()))
    lo, hi = np.quantile(everything, [0.0005, 0.9995])
    grid = np.linspace(lo, hi, bins)

    # (a) density overlay
    ax = axes[0]
    ax.hist(r, bins=grid, density=True, alpha=0.4, color=REAL_C, label="real")
    for (lb, g), c in zip(models.items(), MODEL_COLORS):
        ax.hist(g, bins=grid, density=True, histtype="step",
                lw=1.6, color=c, label=lb)
    ax.set_title("Density", fontsize=17)
    ax.set_xlabel("return", fontsize=15)
    ax.set_ylabel("density", fontsize=15)
    ax.legend(frameon=False, fontsize=11)

    # (b) log-density
    ax = axes[1]
    ax.hist(r, bins=grid, density=True, alpha=0.4, color=REAL_C)
    for (lb, g), c in zip(models.items(), MODEL_COLORS):
        ax.hist(g, bins=grid, density=True, histtype="step", lw=1.6, color=c)
    ax.set_yscale("log")
    ax.set_title("Log-density (tails)", fontsize=17)
    ax.set_xlabel("return", fontsize=15)
    ax.set_ylabel("density (log scale)", fontsize=15)

    # (c) QQ plot: each model's quantiles vs real quantiles
    ax = axes[2]
    q = np.linspace(0.001, 0.999, 400)
    rq = np.quantile(r_t, q)
    lim = abs(rq).max()
    for (lb, g), c in zip(models_t.items(), MODEL_COLORS):
        gq = np.quantile(g, q)
        lim = max(lim, abs(gq).max())
        ax.plot(rq, gq, color=c, lw=1.5, label=lb)
    lim *= 1.05
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, ls="--")
    ax.set_xlabel("real quantiles", fontsize=15)
    ax.set_ylabel("model quantiles", fontsize=15)
    ax.set_title("QQ plot vs real" + tail_note, fontsize=17)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # (d) empirical survival of |.|, log-log: inclusive P(|X| >= x), one
    # point per unique value (keeps the sample maximum, handles ties)
    ax = axes[3]
    series = [("real", r_t, REAL_C)] + [
        (lb, g, c) for (lb, g), c in zip(models_t.items(), MODEL_COLORS)]
    for lb, x, c in series:
        a = np.sort(np.abs(x))
        values, first_indices = np.unique(a, return_index=True)
        sf = (len(a) - first_indices) / len(a)      # P(|X| >= x)
        keep = values > 0                           # log axis needs x > 0
        ax.loglog(values[keep], sf[keep], color=c, lw=1.5, label=lb)
    var = "z" if standardize_tails else "r"
    ax.set_xlabel(f"|{var}|", fontsize=15)
    ax.set_ylabel(f"P(|{var}| \u2265 x)", fontsize=15)
    ax.set_title("Tail survival (log-log)" + tail_note, fontsize=17)
    ax.legend(frameon=False, fontsize=13)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=13)
    if title is not None:
        fig.suptitle(("Marginal distribution: real vs "
                      + " / ".join(models.keys())) if title == "auto"
                     else title, y=1.03, fontsize=18)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


# ===========================================================================
# 2. Per-window statistic distributions
# ===========================================================================
def _window_stats(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1)
    z = (x - mu) / (sd[:, None] + 1e-12)
    kurt = (z ** 4).mean(axis=1) - 3.0
    a = np.abs(x)
    ad = a - a.mean(axis=1, keepdims=True)
    var = (ad ** 2).mean(axis=1) + 1e-12
    acf1 = (ad[:, :-1] * ad[:, 1:]).mean(axis=1) / var
    return {"std": sd, "excess kurtosis": kurt, "|r|-ACF(1)": acf1}


def plot_window_stats(chunks, generated, gen_label="generated",
                      save_path=None, show=True, bins=40, title="auto",
                      show_means=True):
    """Per-window stat distributions: real (all chunks pooled) vs each model.

    show_means : draw a dashed vertical line at each distribution's mean
        (real and every model, in their colours). False for a cleaner plot.
    """
    real = np.concatenate([_to_np(c) for c in chunks], axis=0) \
        if isinstance(chunks, (list, tuple)) else _to_np(chunks)
    models = _as_models(generated, gen_label)

    rs = _window_stats(real)
    ms = {lb: _window_stats(g) for lb, g in models.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    for ax, key in zip(axes, rs):
        both = np.concatenate([rs[key]] + [s[key] for s in ms.values()])
        lo, hi = np.quantile(both, [0.005, 0.995])
        grid = np.linspace(lo, hi, bins)
        ax.hist(rs[key], bins=grid, density=True, alpha=0.4,
                color=REAL_C, label="real")
        if show_means:
            ax.axvline(rs[key].mean(), color=REAL_C, lw=1.2, ls="--")
        for (lb, s), c in zip(ms.items(), MODEL_COLORS):
            ax.hist(s[key], bins=grid, density=True, histtype="step",
                    lw=1.6, color=c, label=lb)
            if show_means:
                ax.axvline(s[key].mean(), color=c, lw=1.2, ls="--")
        ax.set_title(f"per-window {key}", fontsize=17)
        ax.set_xlabel(key, fontsize=15)
        ax.set_ylabel("density", fontsize=15)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=13)
    axes[0].legend(frameon=False, fontsize=13)
    if title is not None:
        fig.suptitle(("Per-window statistics: real vs "
                      + " / ".join(models.keys())) if title == "auto"
                     else title, y=1.03, fontsize=18)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)