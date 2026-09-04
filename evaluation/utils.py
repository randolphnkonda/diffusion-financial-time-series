"""Shared evaluation utilities.

- make_eval_chunks: the disjoint, seed-fixed real chunks consumed by every
  evaluation (stylized facts, C2ST, sliced Wasserstein, MMD, FFAD).
- Kernel helpers for MMD (pairwise distances, median-heuristic bandwidth,
  multi-scale RBF kernel).
- Fourier-feature and Gaussian-statistics helpers for FFAD.
"""

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Gaussian statistics of feature vectors (FFAD)
# ---------------------------------------------------------------------------
def calculate_statistics(features: np.ndarray, eps: float = 1e-6):
    """Calculate the mean and covariance of feature vectors."""
    features = np.asarray(features, dtype=np.float64)

    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")

    if len(features) < 2:
        raise ValueError(
            "At least two samples are required to calculate covariance."
        )

    mean = np.mean(features, axis=0)
    covariance = np.cov(features, rowvar=False)

    # np.cov returns a scalar for one-dimensional latent representations.
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0]) * eps

    return mean, covariance


def symmetric_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Compute the square root of a symmetric positive-semidefinite matrix."""
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

# ---------------------------------------------------------------------------
# Fourier features (FFAD)
# ---------------------------------------------------------------------------
def prepare_fourier_features(
    data, use_phase: bool = True, normalize_fft: bool = True
) -> np.ndarray:
    """Convert time-series samples into flattened Fourier-domain features.

    Expected input shapes:
        Univariate:   (n_samples, sequence_length)
        Multivariate: (n_samples, sequence_length, n_channels)

    Returns:
        Array of shape (n_samples, n_fourier_features).
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim not in (2, 3):
        raise ValueError(
            "data must have shape (samples, time) or "
            "(samples, time, channels)."
        )

    # Fourier transform along the time dimension.
    fft_values = np.fft.rfft(data, axis=1)
    magnitude = np.abs(fft_values)

    if normalize_fft:
        magnitude = np.log1p(magnitude)

    if use_phase:
        phase = np.angle(fft_values)

        # Represent phase continuously to avoid the -π/π discontinuity.
        features = np.concatenate(
            [magnitude, np.sin(phase), np.cos(phase)], axis=-1
        )
    else:
        features = magnitude

    return features.reshape(len(data), -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def make_chunks(series, series_length, stride=None, num_series=None):
    """
    Create chunks from a return series.

    If series_length is None, return the full series as one chunk.

    If num_series is None, return all possible chunks based on stride.
    If num_series is given, return only the first num_series chunks.
    """

    r = pd.Series(series).dropna()

    if series_length is None:
        return [r]

    if stride is None:
        stride = series_length

    chunks = []

    start = 0

    while start + series_length <= len(r):
        chunk = r.iloc[start:start + series_length]

        chunks.append(chunk)

        if num_series is not None and len(chunks) >= num_series:
            break

        start += stride

    return chunks


def make_eval_chunks(X_train, n_per_set=None, n_chunks=None, seed=42):
    """Disjoint real chunks from a seeded permutation of the real windows.

    All evaluations consume the same chunks, so every metric describes
    identical real data; the across-chunk spread of a statistic is its
    sampling variability at that chunk size.

    Exactly one of `n_per_set` / `n_chunks` may be given; if neither, the
    default is n_chunks=6. If `n_per_set` is too large for the data it is
    reduced to the largest size that still yields MIN_CHUNKS chunks (with a
    warning). Deterministic in `seed`.

    Returns (chunks, info): chunks are CPU tensors of shape (n_per_set, L).
    """
    MIN_CHUNKS = 4          # fewer than this and the std estimate is decorative
    MIN_PER_SET = 50        # below this, per-chunk statistics are too noisy
    
    N = len(X_train)

    if n_per_set is not None and n_chunks is not None:
        raise ValueError("Give n_per_set OR n_chunks, not both.")

    if n_per_set is None:
        n_chunks = n_chunks or 6
        n_per_set = N // n_chunks
    else:
        if N // n_per_set < MIN_CHUNKS:
            shrunk = N // MIN_CHUNKS
            print(f"[chunks] WARNING: n_per_set={n_per_set} allows only "
                  f"{N // n_per_set} disjoint chunks from {N} windows; "
                  f"shrinking to n_per_set={shrunk} ({MIN_CHUNKS} chunks).")
            n_per_set = shrunk
        n_chunks = N // n_per_set

    if n_per_set < MIN_PER_SET:
        raise ValueError(
            f"Chunk size {n_per_set} < {MIN_PER_SET}: per-chunk statistics "
            f"would be too noisy (N={N}, n_chunks={n_chunks}). "
            f"Reduce n_chunks."
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    X_cpu = X_train.detach().cpu()
    chunks = [X_cpu[perm[k * n_per_set:(k + 1) * n_per_set]]
              for k in range(n_chunks)]

    leftover = N - n_chunks * n_per_set
    info = {"seed": seed, "n_per_set": n_per_set, "n_chunks": n_chunks,
            "window_length": X_cpu.shape[1], "n_real": N,
            "leftover": leftover}
    print(f"[chunks] {n_chunks} disjoint chunks of {n_per_set} windows "
          f"(N={N}, L={info['window_length']}, seed={seed}, "
          f"{leftover} windows unused)")
    return chunks, info


# ---------------------------------------------------------------------------
# Kernel helpers (MMD)
# ---------------------------------------------------------------------------
def squared_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Euclidean distances between all rows of x and y.
    Returns: Matrix of shape [len(x), len(y)]
    """
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T

    distances = x_norm + y_norm - 2.0 * x @ y.T

    # Numerical round-off can produce tiny negative values.
    return distances.clamp_min(0.0)

def estimate_median_bandwidth(x: torch.Tensor, y:torch.Tensor, max_samples: int = 200,
                              seed: int = 42) -> float:
    """
    Estimate an RBF bandwidth using the median heuristic.

    A subset is used so that bandwidth estimation does not require constructing an excessively
    large distance matrix.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must have shape [num_samples, dimension].")
    
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension.")

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    combined = torch.cat([x, y], dim=0)

    if combined.shape[0] > max_samples:
        indices = torch.randperm(combined.shape[0], generator=generator,
                                 device=x.device)[:max_samples]
        combined = combined[indices]

    distances_sq = squared_pairwise_distances(combined, combined)

    # Remove zero diagonal distances and duplicate-point distances.
    positive_distances_sq = distances_sq[distances_sq > 0]

    if positive_distances_sq.numel() == 0:
        return 1.0
    
    median_distance_sq = positive_distances_sq.median()

    # RBF kernel: exp(-||x - y||^2 / (2 sigma^2))
    sigma = torch.sqrt(median_distance_sq / 2.0).clamp_min(1e-8)

    return sigma.item()

def rbf_kernel(x, y, bandwidths):
    """Multi-scale RBF kernel: mean over `bandwidths` of exp(-d^2 / (2 sigma^2))."""
    distances_sq = squared_pairwise_distances(x, y)

    kernels = []

    for sigma in bandwidths:
        sigma = float(sigma)

        if sigma <= 0:
            raise ValueError("Bandwidth must be positive.")

        kernels.append(
            torch.exp(
                -distances_sq / (2.0 * sigma ** 2)
            )
        )

    return torch.stack(kernels).mean(dim=0)