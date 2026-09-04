"""Predictor-corrector (PC) sampler for the reverse VE-SDE (Song et al., 2021).

Predictor: reverse-diffusion (Euler-Maruyama) step of the reverse SDE.
Corrector: Langevin dynamics with an SNR-based step size.

Returns the generated samples, shape (num_samples, dim).
"""

import torch

from .utils import sigma


def PC_sampler(net, config, seed=42):
    """config keys: noise_levels, steps (corrector steps), snr, num_samples, dim, t_min, t_max, log_every."""
    device = next(net.parameters()).device

    t_min = config.get("t_min", 0.0)
    t_max = config.get("t_max", 1.0)

    N = config.get("noise_levels", 100)
    c_steps = config.get("steps", 1)

    samples = config.get("num_samples", 2000)
    dim = config["dim"]

    snr = config.get("snr", 0.16)

    shape = (samples, dim)
    log_every = config.get("log_every", 20)

    # ---------------------------------------------------------
    # Noise schedule
    # ---------------------------------------------------------
    timesteps = torch.linspace(
        t_min,
        t_max,
        N,
        device=device,
    )

    sigmas = sigma(timesteps)

    if not torch.all(sigmas[1:] >= sigmas[:-1]):
        raise ValueError(
            "sigma(t) must increase with t."
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    # ---------------------------------------------------------
    # VE prior
    # ---------------------------------------------------------
    sigma_T = sigmas[-1]

    X_t = (
        torch.randn(shape, device=device)
        * sigma_T
    )

    net.eval()

    with torch.inference_mode():

        for i in range(N - 2, -1, -1):

            t_i = timesteps[i]
            t_ip1 = timesteps[i + 1]

            sigma_i = sigmas[i]
            sigma_ip1 = sigmas[i + 1]

            delta_sigma2 = (
                sigma_ip1.square()
                - sigma_i.square()
            )

            # =================================================
            # Predictor: VE reverse diffusion
            # =================================================

            t_batch = t_ip1.expand(samples)

            score = net(X_t, t_batch)

            X_mean = (
                X_t
                + delta_sigma2 * score
            )

            # Final step is deterministic denoising
            if i > 0:
                X_t = (
                    X_mean
                    + torch.sqrt(delta_sigma2)
                    * torch.randn_like(X_t)
                )
            else:
                X_t = X_mean

            # =================================================
            # Corrector: Langevin dynamics
            # =================================================

            if i > 0 and c_steps > 0:

                t_batch = t_i.expand(samples)

                for _ in range(c_steps):

                    score = net(X_t, t_batch)
                    noise = torch.randn_like(X_t)

                    grad_norm = (
                        score
                        .reshape(samples, -1)
                        .norm(dim=-1)
                        .mean()
                    )

                    noise_norm = (
                        noise
                        .reshape(samples, -1)
                        .norm(dim=-1)
                        .mean()
                    )

                    step_size = 2 * (
                        snr
                        * noise_norm
                        / grad_norm.clamp_min(1e-12)
                    ).square()

                    X_t = (
                        X_t
                        + step_size * score
                        + torch.sqrt(2 * step_size)
                        * noise
                    )
            if i == 1 or i == N - 2 or i % log_every == 0:
                            print(
                                f"Timestep: {i}, "
                                f"sigma = {sigma_i.item():.5f}, "
                                f"mean = {X_t.mean().item():.4f}, "
                                f"std = {X_t.std().item():.4f}"
                            )
    return X_t

