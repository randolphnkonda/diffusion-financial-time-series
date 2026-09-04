"""Annealed Langevin dynamics (ALD) sampler.

Runs Langevin MCMC at each noise level of the schedule, from sigma_max down
to sigma_min, with a step size proportional to sigma_t^2. Provided as a
reference stochastic sampler; the predictor-corrector sampler is the one
used for the experiments.

Returns (samples, snapshots, snapshot_sigmas).
"""

import torch

from .utils import sigma


def ALD_sampler(net, config):
    """config keys: noise_lvls, t_min (> 0), t_max, steps, num_samples, dim, eps, log_every."""
    device = next(net.parameters()).device

    noise_lvls = config.get("noise_lvls", 100)
    t_min      = config.get("t_min", 1e-3)   # must be > 0 so that sigma(t_min) > 0
    t_max      = config.get("t_max", 1.0)
    N          = config.get("steps", 100)
    log_every  = config.get("log_every", 20)
    samples    = config.get("num_samples", 2000)
    dim        = config.get("dim")
    eps        = config.get("eps", 1e-5)     # Langevin step-size scale

    shape = (samples, dim)

    timesteps = torch.linspace(t_min, t_max, noise_lvls, device=device)
    sigma_min = sigma(timesteps[0])
    sigma_max = sigma(timesteps[-1])
    assert sigma_min > 0, "sigma(t_min) must be > 0, else alpha_t blows up"

    X_t = torch.randn(shape, device=device) * sigma_max

    snapshots, snapshot_sigma = [], []
    net.eval()

    for noise_level in range(noise_lvls - 1, -1, -1):
        t = timesteps[noise_level]
        sigma_t = sigma(t)

        alpha_t = eps * (sigma_t / sigma_min) ** 2
        t_batch = t.expand(samples)              # (B,) continuous time — matches training

        is_last_level = (noise_level == 0)
        for i in range(N):
            with torch.no_grad():
                score = net(X_t, t_batch)        # pass t (B,), NOT sigma_cond (B,1)

            X_t = X_t + 0.5 * alpha_t * score
            if not (is_last_level and i == N - 1):       # no noise on the final step
                X_t = X_t + torch.sqrt(alpha_t) * torch.randn(shape, device=device)

        if noise_level == 0 or noise_level % log_every == 0:
            print(f"Noise level: {noise_level}, sigma = {sigma_t.item():.5f}")
            snapshots.append(X_t.clone())
            snapshot_sigma.append(sigma_t.item())

    return X_t, snapshots, snapshot_sigma
