"""Stylized-fact diagnostic plots on a raw log-return series.

Used to inspect heavy tails, volatility clustering and the leverage effect
of the real series and to guide the choice of window length.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare_chunks(log_returns, config):
    """Split a series into `num_series` consecutive chunks of `series_length`."""
    r = pd.Series(log_returns).dropna()

    series_length = config.get("series_length")
    num_series = config.get("num_series")

    if series_length is None or num_series is None:
        return [r]

    chunks = []

    for i in range(num_series):
        start = i * series_length
        end = start + series_length

        chunk = r.iloc[start:end]

        if len(chunk) == series_length:
            chunks.append(chunk)

    return chunks


def autocorrelation(series, max_lag):
    """Sample autocorrelation of `series` for lags 1..max_lag."""
    s = pd.Series(series).dropna()
    return [s.autocorr(lag=lag) for lag in range(1, max_lag + 1)]


def leverage_effect_values(log_returns, max_lag):
    """
    Lead-lag leverage correlation:
    L(k) = Corr(r_t, r²_(t+k))
    Negative values indicate leverage effect.
    """

    r = pd.Series(log_returns).dropna().values

    leverage = []

    for lag in range(1, max_lag + 1):
        x = r[:-lag]
        y = r[lag:] ** 2

        corr = np.corrcoef(x, y)[0, 1]
        leverage.append(corr)

    return np.array(leverage)


def plot_heavy_tail(log_returns, config):
    chunks = prepare_chunks(log_returns, config)

    alpha = config.get("alpha", 4.35)
    figsize = config.get("figsize", (10, 6))

    plt.figure(figsize=figsize)

    for i, r in enumerate(chunks):
        x = np.sort(np.abs(r))
        x = x[x > 0]

        ccdf = 1.0 - np.arange(1, len(x) + 1) / len(x)
        label = "Full series" if len(chunks) == 1 else f"Chunk {i + 1}, n={len(r)}"
        
        plt.loglog(
            x,
            ccdf,
            marker=".",
            linestyle="none",
            alpha=0.6,
            label=label
        )

    full_r = pd.Series(log_returns).dropna()
    full_abs = np.abs(full_r[full_r != 0])

    x_ref = np.linspace(full_abs.min(), full_abs.max(), 500)

    # Reference Pareto tail slope: P(|r| > x) ~ x^-alpha
    y_ref = x_ref ** (-alpha)

    # Scale reference line to empirical CCDF level
    y_ref = y_ref / y_ref.max()

    plt.loglog(
        x_ref,
        y_ref,
        linestyle="--",
        label=f"Reference tail α={alpha}"
    )

    plt.title("Heavy-Tail Distribution")
    plt.xlabel("|log return|")
    plt.ylabel("P(|r| > x)")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_volatility_clustering(log_returns, config):
    chunks = prepare_chunks(log_returns, config)

    max_lag = config.get("max_lag", 30)
    figsize = config.get("figsize", (10, 6))
    volatility_measure = config.get("volatility_measure", "absolute")

    plt.figure(figsize=figsize)

    for i, r in enumerate(chunks):

        if volatility_measure == "squared":
            vol_proxy = r ** 2
            ylabel = "ACF(r²)"
        else:
            vol_proxy = np.abs(r)
            ylabel = "ACF(|r|)"

        acf_values = autocorrelation(vol_proxy, max_lag)

        label = "Full series" if len(chunks) == 1 else f"Chunk {i + 1}, n={len(r)}"

        plt.scatter(
            range(1, max_lag + 1),
            acf_values,
            alpha=0.7,
            label=label
        )

        plt.plot(
            range(1, max_lag + 1),
            acf_values,
            alpha=0.4
        )

    plt.axhline(0, linestyle="--")
    plt.title("Volatility Clustering")
    plt.xlabel("Lag k")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_leverage_effect(log_returns, config):
    chunks = prepare_chunks(log_returns, config)

    max_lag = config.get("max_lag", 30)
    figsize = config.get("figsize", (10, 6))

    plt.figure(figsize=figsize)

    for i, r in enumerate(chunks):
        lev = leverage_effect_values(r, max_lag)

        label = "Full series" if len(chunks) == 1 else f"Chunk {i + 1}, n={len(r)}"

        plt.scatter(
            range(1, max_lag + 1),
            lev,
            alpha=0.7,
            label=label
        )

        plt.plot(
            range(1, max_lag + 1),
            lev,
            alpha=0.4
        )

    plt.axhline(0, linestyle="--")
    plt.title("Leverage Effect: Lead-Lag Correlation")
    plt.xlabel("Lag k")
    plt.ylabel("Corr(rₜ, r²ₜ₊ₖ)")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_all_stylized_facts(log_returns, config):
    plot_heavy_tail(log_returns, config)
    plot_volatility_clustering(log_returns, config)
    plot_leverage_effect(log_returns, config)