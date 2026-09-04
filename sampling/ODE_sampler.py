"""Probability flow ODE sampler for the VE-SDE with Heun's method
(Song et al., 2021; Heun steps as in Karras et al., 2022).

Deterministic counterpart of the reverse SDE:

    dx/dt = -1/2 * d(sigma^2)/dt * s_theta(x, t)

Discretised on the same sigma grid as the PC sampler, the Euler step is half
the predictor drift with the noise term removed:

    x <- x + 1/2 * (sigma_{i+1}^2 - sigma_i^2) * s_theta(x, t_{i+1})

Heun's method re-evaluates the score at the predicted point and averages the
two slopes (second-order accuracy), so far fewer noise levels are needed than
for the stochastic sampler (K = 30, NFE = 60 in the experiments, against
K = 100 and NFE = 400 for PC). An optional final denoising step at sigma_min
is applied when `denoise=True`.

Returns (samples, snapshots, noise_levels).
"""

import torch

from .utils import sigma


def ODE_sampler(net, config, seed=42):
    """config keys: noise_levels, heun, denoise, num_samples, dim, t_min, t_max, log_every."""
    device = next(net.parameters()).device

    t_min = config.get("t_min", 1e-3)
    t_max = config.get("t_max", 1.0)
    N = config.get("noise_levels", 40)
    log_every = config.get("log_every", 10)
    samples = config.get("num_samples", 2000)
    dim = config["dim"]
    use_heun = config.get("heun", True)
    denoise = config.get("denoise", False)

    shape = (samples, dim)

    timesteps = torch.linspace(t_min, t_max, N, device=device)
    sigmas = sigma(timesteps)

    if not torch.all(sigmas[1:] >= sigmas[:-1]):
        raise ValueError("sigma(t) must increase with t.")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    sigma_max = sigmas[-1]
    X_t = torch.randn(shape, device=device, generator=generator) * sigma_max   # prior N(0, sigma_max^2 I)

    snapshots, noise_levels = [], []

    net.eval()

    with torch.inference_mode():
        for i in range(N - 2, -1, -1):
            t_i, t_ip1 = timesteps[i], timesteps[i + 1]
            sigma_i, sigma_ip1 = sigmas[i], sigmas[i + 1]

            delta_sigma2 = sigma_ip1.square() - sigma_i.square()

            # ---------------------------------------------------------
            # Probability-flow ODE predictor
            # dx / d(sigma^2) = -0.5 * score
            # ---------------------------------------------------------
            score_1 = net(X_t, t_ip1.expand(samples))
            drift_1 = 0.5 * delta_sigma2 * score_1
            X_euler = X_t + drift_1

            # ---------------------------------------------------------
            # Heun correction
            # ---------------------------------------------------------
            if use_heun:
                score_2 = net(X_euler, t_i.expand(samples))
                drift_2 = 0.5 * delta_sigma2 * score_2
                X_t = X_t + 0.5 * (drift_1 + drift_2)
            else:
                X_t = X_euler

            # ---------------------------------------------------------
            # Diagnostics
            # ---------------------------------------------------------
            if i == 0 or i == N - 2 or i % log_every == 0:
                print(
                    f"Timestep: {i}, sigma = {sigma_i.item():.5f}, "
                    f"mean = {X_t.mean().item():.4f}, "
                    f"std = {X_t.std().item():.4f}"
                )
                snapshots.append(X_t.detach().cpu().clone())
                noise_levels.append(f"{sigma_i.item():.4f}")

        # -------------------------------------------------------------
        # Optional final denoising
        # -------------------------------------------------------------
        if denoise:
            score = net(X_t, timesteps[0].expand(samples))
            X_t = X_t + sigmas[0].square() * score

    return X_t, snapshots, noise_levels
