"""FFAD (Fréchet Fourier-transform Auto-encoder Distance).

Requires:
 - Fourier transform of input data from time to frequency domain
 - Trained autoencoder with a reconstruction loss on the input dataset in frequency space
 - Featue embeddings from the auto encoder of the real and generated samples
 - Compute Fréchet Distance between the real and generated feature embeddings

Fourier transform is used because the frequency domain of the time series reveals hidden 
temporal dynamics (such as; multi-period seasonality, systemic volatility, phase shift) 
and structural properties (such as; signal-to-noise ration, global sequence topology) 
present in the series.
"""

from scipy.fft import fft
import torch
import torch.nn as nn
import numpy as np
from .utils import symmetric_matrix_sqrt, calculate_statistics, prepare_fourier_features


# ---------------------------------------------------------------------------
# FFAD Latent Vectors
# ---------------------------------------------------------------------------
def get_ffad_latent_vectors(
    data, model, device: str = "cpu", batch_size: int = 512
) -> np.ndarray:
    """Apply the trained autoencoder encoder to Fourier-domain features."""
    model = model.to(device)
    model.eval()

    data_tensor = torch.as_tensor(data, dtype=torch.float32)
    latent_batches = []

    with torch.no_grad():
        for start in range(0, len(data_tensor), batch_size):
            batch = data_tensor[start : start + batch_size].to(device)
            latent = model.encoder(batch)
            latent_batches.append(latent.cpu())

    return torch.cat(latent_batches, dim=0).numpy()


# ---------------------------------------------------------------------------
# Compute Frechet Distance
# ---------------------------------------------------------------------------
def frechet_distance(
                        mean_real: np.ndarray,
                        covariance_real: np.ndarray,
                        mean_generated: np.ndarray,
                        covariance_generated: np.ndarray,
                    ) -> float:
    """Numerically stable Fréchet distance between two Gaussian distributions."""
    mean_real = np.asarray(mean_real, dtype=np.float64)
    mean_generated = np.asarray(mean_generated, dtype=np.float64)
    covariance_real = np.asarray(covariance_real, dtype=np.float64)
    covariance_generated = np.asarray(covariance_generated, dtype=np.float64)

    mean_difference = mean_real - mean_generated

    # Stable form: Tr(C1 + C2 - 2 * sqrt(sqrt(C1) C2 sqrt(C1)))
    sqrt_covariance_real = symmetric_matrix_sqrt(covariance_real)
    covariance_product = (
        sqrt_covariance_real @ covariance_generated @ sqrt_covariance_real
    )
    covariance_mean = symmetric_matrix_sqrt(covariance_product)

    distance = (
        mean_difference @ mean_difference
        + np.trace(covariance_real)
        + np.trace(covariance_generated)
        - 2.0 * np.trace(covariance_mean)
    )

    # Remove tiny negative values caused by numerical precision.
    return float(max(distance, 0.0))


# ---------------------------------------------------------------------------
# Evaluate FFAD
# ---------------------------------------------------------------------------
def evaluate_ffad(
                        real_data,
                        generated_data,
                        autoencoder,
                        device: str = "cpu",
                        batch_size: int = 512,
                        use_phase: bool = True,
                        normalize_fft: bool = True,
                ):
    """Calculate FFAD between real and generated time-series datasets.

    The autoencoder must be trained on Fourier features produced with the
    same `use_phase` and `normalize_fft` settings.

    Returns:
        Dictionary containing the FFAD score and intermediate information.
    """
    real_data = np.asarray(real_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)

    if real_data.ndim != generated_data.ndim:
        raise ValueError(
            "Real and generated data must have the same number of dimensions."
        )

    if real_data.shape[1:] != generated_data.shape[1:]:
        raise ValueError(
            "Real and generated samples must have matching time-series shapes. "
            f"Received {real_data.shape[1:]} and {generated_data.shape[1:]}."
        )

    real_fourier = prepare_fourier_features(
        real_data, use_phase=use_phase, normalize_fft=normalize_fft
    )
    generated_fourier = prepare_fourier_features(
        generated_data, use_phase=use_phase, normalize_fft=normalize_fft
    )

    expected_input_dim = autoencoder.encoder[0].in_features
    if real_fourier.shape[1] != expected_input_dim:
        raise ValueError(
            "The autoencoder input dimension does not match the Fourier "
            "feature dimension. "
            f"Model expects {expected_input_dim}, but FFAD produced "
            f"{real_fourier.shape[1]} features."
        )

    real_latent = get_ffad_latent_vectors(
        real_fourier, autoencoder, device=device, batch_size=batch_size
    )
    generated_latent = get_ffad_latent_vectors(
        generated_fourier, autoencoder, device=device, batch_size=batch_size
    )

    mean_real, covariance_real = calculate_statistics(real_latent)
    mean_generated, covariance_generated = calculate_statistics(
        generated_latent
    )

    ffad_score = frechet_distance(
        mean_real, covariance_real, mean_generated, covariance_generated
    )

    return {
        "ffad": ffad_score,
        "real_latent": real_latent,
        "generated_latent": generated_latent,
        "real_latent_mean": mean_real,
        "generated_latent_mean": mean_generated,
        "real_latent_covariance": covariance_real,
        "generated_latent_covariance": covariance_generated,
    }

