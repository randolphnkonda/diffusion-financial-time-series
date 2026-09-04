"""Sliding-window dataset of standardised daily log returns."""

import numpy as np
import torch
from torch.utils.data import Dataset


class DailyLogReturnsData(Dataset):
    """Windows of length `window_size` taken every `stride` steps from a return series.

    If `normalize` is True the series is z-scored with its own mean and
    standard deviation, which are kept in `self.mean` / `self.std` so that
    generated samples can be mapped back to raw return units.
    Each item has shape (1, window_size).
    """

    def __init__(self, log_returns, window_size=64, stride=1, normalize=True):

        self.mean = log_returns.mean()
        self.std = log_returns.std()

        if normalize:
            log_returns = (log_returns - self.mean) / self.std

        values = log_returns.values.astype(np.float32)

        windows = []
        for start in range(0, len(values) - window_size + 1, stride):
            end = start + window_size
            windows.append(values[start:end])

        self.X = torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]