"""General plotting utilities: sample paths, price and log-return series, loss curve."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Sample paths
# ---------------------------------------------------------------------------
def plot_sample_returns(
    samples,
    save_fig=False,
    title=None,
    show_titles=True
):
    """
    Plot a single return series or up to 12 sample paths.

    Parameters
    ----------
    samples :
        pandas DataFrame, pandas Series, numpy array,
        torch Tensor, or collection of sample paths.

    save_fig : bool, default=False
        Whether to save the figure.

    title : str, optional
        Required if save_fig=True.
        Figure will be saved as:
        figures/<title>_returns.png

    show_titles : bool, default=True
        Whether to display figure and subplot titles.
    """

    if save_fig and not title:
        raise ValueError("title must be provided when save_fig=True")

    def save_if_requested():
        if save_fig:
            os.makedirs("figures", exist_ok=True)

            plt.savefig(
                f"figures/{title}_returns.png",
                dpi=300,
                bbox_inches="tight"
            )

    # ------------------------
    # Single pandas DataFrame
    # ------------------------
    if isinstance(samples, pd.DataFrame):
        plt.figure(figsize=(10, 4))
        plt.plot(samples)

        if show_titles:
            plt.title(title if title else "Return Series")

        plt.grid(True)
        plt.tight_layout()

        save_if_requested()
        plt.show()
        return

    # ------------------------
    # Single pandas Series
    # ------------------------
    if isinstance(samples, pd.Series):
        plt.figure(figsize=(10, 4))
        plt.plot(samples.values)

        if show_titles:
            plt.title(title if title else "Return Series")

        plt.grid(True)
        plt.tight_layout()

        save_if_requested()
        plt.show()
        return

    # ------------------------
    # Single numpy array
    # ------------------------
    if isinstance(samples, np.ndarray) and samples.ndim == 1:
        plt.figure(figsize=(10, 4))
        plt.plot(samples)

        if show_titles:
            plt.title(title if title else "Return Series")

        plt.grid(True)
        plt.tight_layout()

        save_if_requested()
        plt.show()
        return

    # ------------------------
    # Single torch tensor
    # ------------------------
    if isinstance(samples, torch.Tensor) and samples.ndim == 1:
        plt.figure(figsize=(10, 4))
        plt.plot(samples.detach().cpu().numpy())

        if show_titles:
            plt.title(title if title else "Return Series")

        plt.grid(True)
        plt.tight_layout()

        save_if_requested()
        plt.show()
        return

    # ------------------------
    # Multiple samples
    # ------------------------
    num_samples = min(12, len(samples))
    num_rows = -(-num_samples // 3)

    fig, axes = plt.subplots(
        num_rows,
        3,
        figsize=(15, 2.6 * num_rows)
    )

    axes = np.atleast_1d(axes).flatten()

    for i in range(num_samples):
        sample = samples[i]

        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()

        elif isinstance(sample, pd.Series):
            sample = sample.values

        axes[i].plot(sample)

        if show_titles:
            axes[i].set_title(f"Sample {i + 1}")

        axes[i].grid(True)

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    if show_titles and title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    save_if_requested()
    plt.show()

# ---------------------------------------------------------------------------
# Price and log-return series
# ---------------------------------------------------------------------------

def plot_price(stock_data, ticker):
    """
    Plot and save the closing price series for a given ticker.
    """
    plt.figure(figsize=(10, 5))

    plt.plot(
        stock_data.index,
        stock_data["Close"],
        linewidth=1,
        label=ticker
    )

    plt.xlabel("Date")
    plt.ylabel("Share Price")
    plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend()

    plt.tight_layout()

    # Create figures directory if it does not exist
    os.makedirs("figures", exist_ok=True)

    # Save figure
    plt.savefig(
        f"figures/{ticker}_price.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_log_returns(log_returns, ticker):
    """
    Plot and save log returns for a given ticker.
    """
    log_returns = np.asarray(log_returns).flatten()

    plt.figure(figsize=(12, 4))

    plt.plot(
        log_returns,
        linewidth=0.8,
        label=ticker
    )

    plt.axhline(0, linestyle="--", linewidth=0.8)

    plt.xlabel("Time")
    plt.ylabel("Log Return")

    plt.tight_layout()

    # Create figures directory if necessary
    os.makedirs("figures", exist_ok=True)

    # Save figure
    plt.savefig(
        f"figures/{ticker}_logreturns.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_leverage_curve(leverage_curve):
    leverage_curve.plot(marker="o")
    plt.axhline(0, linestyle="--")
    plt.title("Leverage Effect: Corr(r_t-k, r_t²)")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.show()


def plot_loss_curve(losses, save_path, title="Training Loss",
                    smooth_window=10, log_scale=True, show=True):
    """
    Plot and save a training loss curve.

    losses        : list/array of per-epoch losses (as returned by train_net)
    save_path     : where to save, e.g. "figures/loss_curve.png"
    smooth_window : moving-average window (set 0/None to disable)
    log_scale     : log y-axis (recommended for score-matching losses)
    show          : call plt.show() after saving
    """
    losses = np.asarray(losses, dtype=float)
    epochs = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # raw curve, faint
    ax.plot(epochs, losses, color="steelblue", alpha=0.35,
            linewidth=1.0, label="per epoch")

    # smoothed overlay
    if smooth_window and len(losses) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode="valid")
        smooth_x = epochs[smooth_window - 1:]
        ax.plot(smooth_x, smoothed, color="crimson", linewidth=2.0,
                label=f"moving avg ({smooth_window})")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss" + (" (log scale)" if log_scale else ""))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    # annotate final loss
    ax.annotate(f"final: {losses[-1]:.5f}",
                xy=(epochs[-1], losses[-1]),
                xytext=(-70, 15), textcoords="offset points",
                fontsize=9, color="dimgray")

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)   # create dirs if missing
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved loss curve to {save_path.resolve()}")

    if show:
        plt.show()
    plt.close(fig)