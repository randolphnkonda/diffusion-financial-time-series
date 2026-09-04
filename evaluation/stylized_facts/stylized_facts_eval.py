"""Stylized-facts evaluation entry points used by the notebooks.

Consumes the shared chunks from `evaluation.utils.make_eval_chunks`.

    1. bench, real_table = build_real_benchmark(chunks, lags=..., save_path=...)
       Computes and displays the real-data table (mean +/- std across chunks).
       Cached to JSON so it is computed once per chunk size.

    2. results = evaluate_stylized_facts(name, generated, chunks, bench)
       Displays the generated-vs-real table (with a z-score per statistic
       and lag) and saves the diagnostic curves figure.

The across-chunk standard deviation, and hence every z-score, is specific to
the chunk size; models compared with one another should be evaluated at a
common chunk size.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .stylized_facts import (
    RealBenchmark,
    compare_to_benchmark,
    plot_stylized_curves,
)

try:
    from IPython.display import display, Markdown
    def _show(title, df):
        display(Markdown(f"### {title}"))
        display(df)
except ImportError:                       # plain-script fallback
    def _show(title, df):
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))


# Statistics shown in tables by default (internal key -> display name).
# Pass key_stats=None to any display function to show everything.
KEY_STATS = {
    "mean": "mean",
    "std": "std",
    "skewness": "skewness",
    "excess_kurtosis": "excess_kurtosis",
    "acf_return": "acf_return",
    "acf_abs_return": "acf_abs_return",
    "acf_sq_return": "acf_squared_return",
    "leverage": "leverage_effect",
}


def _filter_stats(df, key_stats):
    """Keep only key_stats rows (in that order) and apply display names."""
    if key_stats is None:
        return df
    df = df[df["statistic"].isin(key_stats)].copy()
    order = {k: i for i, k in enumerate(key_stats)}
    df["_o"] = df["statistic"].map(order)
    df = df.sort_values(["_o", "lag"], na_position="first",
                        ignore_index=True).drop(columns="_o")
    df["statistic"] = df["statistic"].map(key_stats)
    return df


# ===========================================================================
# Real benchmark: compute + display once per chunk size
# ===========================================================================
def build_real_benchmark(chunks, lags=(1, 5, 10, 20), save_path=None,
                         force=False, key_stats=KEY_STATS):
    """Compute (or load cached) benchmark and display the real-only table.

    key_stats: dict of internal->display names to show (default KEY_STATS);
    pass None to display every computed statistic. The benchmark object
    always stores everything regardless.
    """
    if save_path and Path(save_path).exists() and not force:
        bench = RealBenchmark.load(save_path)
        if bench.chunk_shape[0] != len(chunks[0]):
            raise ValueError(
                f"Cached benchmark chunk size {bench.chunk_shape[0]} != "
                f"current {len(chunks[0])}. Use a different save_path per "
                f"chunk size, or pass force=True to recompute."
            )
        print(f"[benchmark] loaded from {save_path}")
    else:
        bench = RealBenchmark.from_chunks(chunks, lags=lags)
        if save_path:
            bench.save(save_path)

    rows = []
    for stat, m in bench.mean.items():
        name, _, lag = stat.partition("@lag")
        rows.append({"statistic": name,
                     "lag": int(lag) if lag else None,
                     "real_mean": m,
                     "real_std": bench.std.get(stat, np.nan)})
    real_table = (pd.DataFrame(rows)
                  .sort_values(["statistic", "lag"], na_position="first",
                               ignore_index=True))
    real_table = _filter_stats(real_table, key_stats)
    _show(f"Real benchmark ({bench.n_chunks} chunks × "
          f"{bench.chunk_shape[0]} windows)", real_table.round(4))
    return bench, real_table


# ===========================================================================
# Per-model stylized-facts evaluation
# ===========================================================================
def evaluate_stylized_facts(name, generated, chunks, bench,
                            max_lag_curves=50, fig_dir="figures",
                            show_curves=True, key_stats=KEY_STATS):
    """Gen-vs-real stylized facts for one model. Returns table + figure path.

    key_stats: statistics to display (default KEY_STATS); None shows all.
    The returned stylized_table is the displayed (filtered) one; the full
    unfiltered table is returned under 'stylized_table_full'.
    """
    gen_cpu = generated.detach().cpu() if torch.is_tensor(generated) \
        else torch.as_tensor(np.asarray(generated))

    table_full = compare_to_benchmark(gen_cpu, bench, model_name=name)
    table = _filter_stats(table_full, key_stats)
    _show(f"{name}: stylized facts vs real", table.round(4))

    fig_path = Path(fig_dir) / f"{name}_stylized_curves.png"
    plot_stylized_curves(chunks, gen_cpu, max_lag=max_lag_curves,
                         model_name=name, save_path=fig_path,
                         show=show_curves)

    return {"name": name, "stylized_table": table,
            "stylized_table_full": table_full, "curves_path": fig_path}