"""Autoencoder network, training loop, latent and reconstructed vectors representation.
"""
import torch
import numpy as np
import torch.nn as nn

# ---------------------------------------------------------------------------
# Autoencoder Network
# ---------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded

# ---------------------------------------------------------------------------
# Latent Vectors
# ---------------------------------------------------------------------------
def get_latent_vectors(data, model: AutoEncoder, device: str ="cpu") -> torch.Tensor:
    """Extract the latent representation from the autoencoder."""
    model = model.to(device)
    model.eval()

    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        data_tensor = data.float()
    else:
        data_tensor = torch.tensor(data, dtype=torch.float32)
    
    data_tensor = data_tensor.to(device)

    with torch.no_grad():
        latent_vectors = model.encoder(data_tensor)

    return latent_vectors.cpu()

# ---------------------------------------------------------------------------
# Reconstructed Vectors
# ---------------------------------------------------------------------------
def get_reconstructed_vectors(data, model, device="cpu"):
    """Return reconstructed vectors and the overall mean squared error."""

    model = model.to(device)
    model.eval()

    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        data_tensor = data.float()
    else:
        data_tensor = torch.tensor(data, dtype=torch.float32)

    data_tensor = data_tensor.to(device)

    with torch.no_grad():
        reconstructed_vectors = model(data_tensor)
        reconstruction_error = torch.mean(
            (reconstructed_vectors - data_tensor) ** 2
        )

    return (
        reconstructed_vectors.cpu(),
        reconstruction_error.item()
    )
