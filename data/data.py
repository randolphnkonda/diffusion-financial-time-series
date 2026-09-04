"""Data acquisition and descriptive diagnostics for the log-return series.

`get_data` is a validated pipeline: the ticker and date range are checked
before any download, the download is verified to contain usable rows, the
price frame is cleaned (sorted, de-duplicated, missing and non-positive
closes removed) with a per-step report, and only then are log returns
derived. Clean input passes through unchanged.
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf


# ---------------------------------------------------------------------------
# Acquisition pipeline: validate -> download -> clean -> derive returns
# ---------------------------------------------------------------------------
def _validate_config(config):
    """Check ticker and date range before touching the network.

    Returns (ticker, start, end) with the dates as pandas Timestamps.
    Raises ValueError with a specific message for each failure mode.
    """
    ticker = config.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(
            f"config['ticker'] must be a non-empty string, got {ticker!r}."
        )
    ticker = ticker.strip()

    start_raw, end_raw = config.get("start"), config.get("end")
    try:
        start = pd.to_datetime(start_raw)
        end = pd.to_datetime(end_raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Unparseable date(s): start={start_raw!r}, end={end_raw!r}."
        ) from exc
    if pd.isna(start) or pd.isna(end):
        raise ValueError(
            f"config['start'] and config['end'] are required, got "
            f"start={start_raw!r}, end={end_raw!r}."
        )
    if start >= end:
        raise ValueError(f"start ({start.date()}) must be before end ({end.date()}).")
    if start > pd.Timestamp.today():
        raise ValueError(f"start ({start.date()}) lies in the future.")
    return ticker, start, end


def _download_prices(ticker, start, end):
    """Download daily OHLCV data, normalising yfinance's output format.

    Raises RuntimeError on network/API failure and ValueError when the
    request succeeds but returns no usable rows (typically a bad ticker or
    an empty date range).
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as exc:
        raise RuntimeError(
            f"Download failed for {ticker!r} ({start.date()} to {end.date()}): {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker {ticker!r} between {start.date()} "
            f"and {end.date()}. Check that the ticker exists on Yahoo Finance "
            f"and that the range overlaps its listing period."
        )

    # yfinance returns MultiIndex columns (field, ticker) in some versions;
    # flatten to plain field names for a single-ticker download.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in ("Open", "High", "Low", "Close", "Volume")
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"Download for {ticker!r} is missing expected column(s) {missing}; "
            f"got columns {list(df.columns)}."
        )
    return df[["Open", "High", "Low", "Close", "Volume"]]


def clean_price_data(df):
    """Clean a raw daily price frame before any return is computed.

    Steps (each counted in the returned report):
        1. drop rows whose index is not a valid timestamp, then sort by date
        2. drop duplicate dates (first occurrence kept)
        3. coerce Close to numeric (unparseable entries become NaN)
        4. drop rows with missing Close
        5. drop rows with non-positive Close (log returns undefined)

    Returns (clean_df, report) where report maps step -> rows removed.
    """
    report = {}
    out = df.copy()

    n = len(out)
    out = out[pd.notna(pd.to_datetime(out.index, errors="coerce"))]
    out = out.sort_index()
    report["invalid_index"] = n - len(out)

    n = len(out)
    out = out[~out.index.duplicated(keep="first")]
    report["duplicate_dates"] = n - len(out)

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")

    n = len(out)
    out = out.dropna(subset=["Close"])
    report["missing_close"] = n - len(out)

    n = len(out)
    out = out[out["Close"] > 0]
    report["nonpositive_close"] = n - len(out)

    return out, report


def get_data(config, save_csv=True, min_observations=2, verbose=True):
    """Validated pipeline from ticker to daily log returns.

    validate -> download -> clean -> derive log returns. Returns a DataFrame
    with columns Close and log_return, indexed by date. The cleaned OHLCV
    download is written to ./tables/<ticker>_daily_data.csv when `save_csv`.

    Raises ValueError for a bad ticker, bad dates, or a series left with
    fewer than `min_observations` returns after cleaning, and RuntimeError
    for download failures. A per-step cleaning report is printed when any
    row was removed (always shown with `verbose=True`).
    """
    ticker, start, end = _validate_config(config)
    raw = _download_prices(ticker, start, end)
    df, report = clean_price_data(raw)

    if save_csv:
        os.makedirs("./tables", exist_ok=True)
        df.to_csv("./tables/" + ticker + "_daily_data.csv")

    s_price = df[["Close"]].copy()
    s_price["log_return"] = np.log(s_price["Close"] / s_price["Close"].shift(1))
    s_price = s_price.replace([np.inf, -np.inf], np.nan).dropna()

    if len(s_price) < min_observations:
        raise ValueError(
            f"Only {len(s_price)} log return(s) remain for {ticker!r} after "
            f"cleaning (minimum {min_observations}). Cleaning report: {report}."
        )

    removed = sum(report.values())
    if verbose or removed:
        msg = f"[data] {ticker}: {len(raw)} rows downloaded"
        if removed:
            steps = ", ".join(f"{k}={v}" for k, v in report.items() if v)
            msg += f", {removed} removed ({steps})"
        msg += f", {len(s_price)} log returns."
        print(msg)

    return s_price


def get_summary(log_returns):
    """Print and return summary statistics of a log-return series."""
    summary = {
        "Mean": log_returns.mean(),
        "Std": log_returns.std(),
        "Skewness": skew(log_returns),
        "Kurtosis": kurtosis(log_returns, fisher=False),  # Normal = 3
        "JB Statistic": jarque_bera(log_returns).statistic,
        "JB p-value": jarque_bera(log_returns).pvalue,
        "5% Quantile": log_returns.quantile(0.05),
        "95% Quantile": log_returns.quantile(0.95),
        "Annual Return": log_returns.mean() * 252,
        "Annual Volatility": log_returns.std() * np.sqrt(252),
        "Sharpe Ratio": (
            log_returns.mean() * 252
            / (log_returns.std() * np.sqrt(252))
        )
    }
    print(pd.Series(summary))

    return summary


def return_diagnostics(log_returns, lags=20):
    """Stylized-fact diagnostics for one log-return series.

    Returns (summary statistics, leverage correlation by lag, the series).
    """

    r = log_returns
    r2 = r ** 2
    abs_r = r.abs()
    out = {}

    # -------------------------
    # 1. Heavy tails
    # -------------------------

    out["mean"] = r.mean()
    out["std"] = r.std()
    out["skewness"] = skew(r)
    out["excess_kurtosis"] = kurtosis(r, fisher=True)   # normal = 0
    out["kurtosis"] = kurtosis(r, fisher=False)         # normal = 3
    out["tail_ratio_1pct_99pct"] = abs(r.quantile(0.01)) / abs(r.quantile(0.99))
    out["left_tail_1pct"] = r.quantile(0.01)
    out["right_tail_99pct"] = r.quantile(0.99)

    # -------------------------
    # 2. Volatility clustering
    # -------------------------

    out["acf_return_lag1"] = r.autocorr(lag=1)
    out["acf_abs_return_lag1"] = abs_r.autocorr(lag=1)
    out["acf_squared_return_lag1"] = r2.autocorr(lag=1)
    arch_test = het_arch(r, nlags=lags)
    out["arch_lm_stat"] = arch_test[0]
    out["arch_lm_pvalue"] = arch_test[1]

    # -------------------------
    # 3. Leverage effect
    # negative returns followed by higher future volatility
    # -------------------------

    out["corr_return_next_squared_return"] = r.shift(1).corr(r2)
    leverage_by_lag = {}
    
    for lag in range(1, lags + 1):
        leverage_by_lag[f"leverage_lag_{lag}"] = r.shift(lag).corr(r2)

    return pd.Series(out), pd.Series(leverage_by_lag), r


def diagnostics_for_tensor(sample_tensor, lags=20):
    """Run `return_diagnostics` on each row of a (N, T) tensor (first 10 rows).

    Returns (diagnostics DataFrame, leverage-by-lag DataFrame).
    """
    sample_tensor = sample_tensor[:10]
    all_diagnostics = []
    all_leverage = []

    for i in range(sample_tensor.shape[0]):
        r = sample_tensor[i].detach().cpu().numpy()
        r = pd.Series(r)

        diagnostics, leverage, _ = return_diagnostics(r, lags=lags)

        diagnostics.name = f"sample_{i}"
        leverage.name = f"sample_{i}"

        all_diagnostics.append(diagnostics)
        all_leverage.append(leverage)

    diagnostics_df = pd.DataFrame(all_diagnostics)
    leverage_df = pd.DataFrame(all_leverage)

    return diagnostics_df, leverage_df