"""Training loop for the FFAD autoencoder (reconstruction loss)."""

import numpy as np
import torch

from .autoencoder import AutoEncoder


def train_ae(data, net: AutoEncoder, train_config: dict):
    """Train an autoencoder with the mean squared reconstruction loss.

    train_config keys: device, lr, epochs, batch_size, log_every.
    Returns the per-epoch training loss.
    """
    device = train_config.get("device", "cpu")

    net = net.to(device)
    net.train()

    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        data_tensor = data.float()
    else:
        data_tensor = torch.tensor(data, dtype=torch.float32)
    
    data_tensor = data_tensor.to(device)

    n = len(data)
    if n == 0:
        raise ValueError("The training dataset is empty")

    batch_size = train_config.get("batch_size", 64)
    epochs = train_config.get("epochs", 20)
    lr = train_config.get("lr", 0.02)
    log_every = train_config.get("log_every", 20)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    training_loss = []
    
    for epoch in range(epochs):
        indices = torch.randperm(n, device=device)
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            batch_indices = indices[start: start + batch_size]
            batch = data_tensor[batch_indices]

            optimizer.zero_grad()
            
            reconstructed = net(batch)
            loss = criterion(reconstructed, batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
        
        epoch_loss /= n
        training_loss.append(epoch_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"Epoch = {epoch + 1}/{epochs}, Reconstruction Loss = {epoch_loss:.6f}")

    return training_loss