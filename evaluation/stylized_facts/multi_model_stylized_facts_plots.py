"""Stylized-facts curves for several generated sets against the real band.

Used for capacity, sampler and data-volume comparisons across models. Curves are computed with the same helpers as
`plot_stylized_curves`, so the numbers match the single-model figures.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .stylized_facts import _to_np, _acf_curve, _lev_curve, _curve_band

REAL_C = "black"
MODEL_COLORS = ["darkorange", "cyan", "brown"]


def plot_model_comparison_curves(chunks, models, max_lag=50,
                                 include_acf=True, panels=None,
                                 mark_lags=(10, 20),
                                 title="auto", save_path=None, show=True):
    """Real band (mean +/- 2 sd across chunks) vs each generated set.

    chunks : list of (n, L) real chunks (the shared evaluation chunks)
    models : dict {label: (n, L) generated set}; 1+ entries, order preserved
    panels : which panels to draw, in order. Any subset of
        ("acf", "acf_abs", "lev"), e.g. ("acf",) for the raw-return ACF
        alone, or ("acf_abs", "lev") for clustering + leverage. When None
        (default), falls back to include_acf: all three panels if True,
        ("acf_abs", "lev") if False.
    include_acf : shorthand for the default panel set, only consulted when panels is None
    mark_lags : lags to mark with dotted vertical lines (empty/None to skip)
    title : figure suptitle. "auto" (default) builds one from the model
        labels; None suppresses the title entirely (e.g. for figures whose
        caption lives in LaTeX); any other string is used verbatim.
    """
    if not models:
        raise ValueError("models dict must contain at least one entry.")
    if len(models) > len(MODEL_COLORS):
        raise ValueError(f"At most {len(MODEL_COLORS)} models per figure.")

    # ---- real bands (computed once) ---------------------------------------
    acf_r_m, acf_r_s = _curve_band(chunks, lambda x, m: _acf_curve(x, m), max_lag)
    acf_a_m, acf_a_s = _curve_band(
        chunks, lambda x, m: _acf_curve(x, m, transform=np.abs), max_lag)
    lev_m, lev_s = _curve_band(chunks, _lev_curve, max_lag)

    # ---- model curves ------------------------------------------------------
    curves = {}
    for label, gen in models.items():
        g = _to_np(gen)
        curves[label] = {
            "acf": _acf_curve(g, max_lag),
            "acf_abs": _acf_curve(g, max_lag, transform=np.abs),
            "lev": _lev_curve(g, max_lag),
        }

    # ---- figure ------------------------------------------------------------
    if panels is None:
        panels = ("acf", "acf_abs", "lev") if include_acf else ("acf_abs", "lev")
    PANEL_SPEC = {
        "acf": (acf_r_m[1:], acf_r_s[1:], "ACF of returns", "autocorrelation"),
        "acf_abs": (acf_a_m[1:], acf_a_s[1:],
                    "ACF of |returns| (vol clustering)", "autocorrelation"),
        "lev": (lev_m, lev_s,
                r"Leverage: $\mathrm{corr}(r_t,\, |r_{t+k}|)$", "correlation"),
    }
    unknown = [p for p in panels if p not in PANEL_SPEC]
    if unknown or not panels:
        raise ValueError(f"panels must be a non-empty subset of "
                         f"{tuple(PANEL_SPEC)}; got {panels}")

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels + 0.5, 4.5),
                             dpi=150)
    axes = np.atleast_1d(axes)
    lags1 = np.arange(1, max_lag + 1)

    def panel(ax, rm, rs, key, ptitle, ylabel=None):
        ax.fill_between(lags1, rm - 2 * rs, rm + 2 * rs, color=REAL_C,
                        alpha=0.22, label="real \u00b12sd")
        ax.plot(lags1, rm, color=REAL_C, lw=1.6, label="real mean")
        for (label, c), color in zip(curves.items(), MODEL_COLORS):
            ax.plot(lags1, c[key][-max_lag:] if key != "lev" else c[key],
                    color=color, lw=1.4, label=label)
        for ml in (mark_lags or []):
            if 1 <= ml <= max_lag:
                ax.axvline(ml, color="gray", lw=0.9, ls=":", alpha=0.8)
        ax.axhline(0, color="gray", lw=0.7, alpha=0.6)
        ax.set_title(ptitle)
        ax.set_xlabel("lag")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    for idx, name in enumerate(panels):
        rm, rs, ptitle, natural_ylabel = PANEL_SPEC[name]
        show_ylabel = (idx == 0) or (name == "lev")
        panel(axes[idx], rm, rs, name, ptitle,
              ylabel=natural_ylabel if show_ylabel else None)

    axes[0].legend(frameon=False, fontsize=8)
    if title is not None:
        fig.suptitle(("Stylized facts: real vs " + " / ".join(models.keys()))
                     if title == "auto" else title, y=1.02)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)