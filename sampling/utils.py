"""Noise schedule shared by all samplers.

Must match the schedule used in training (model/train_network.py).
"""


def sigma(t, sigma_min=0.01, sigma_max=10.0):
    """sigma(t) = sigma_min * (sigma_max / sigma_min) ** t  for t in [0, 1]."""
    return sigma_min * ((sigma_max / sigma_min)**t)
