"""Training of the score network by denoising score matching.

Implements the exponential VE noise schedule, the sigma^2-weighted denoising
score matching loss (Vincent, 2011; Song et al., 2021), and the training loop.
"""

import torch


# ---------------------------------------------------------------------------
# Exponential noise schedule
# ---------------------------------------------------------------------------
def sigma(t, sigma_min=0.01, sigma_max=10.0):
    """sigma(t) = sigma_min * (sigma_max / sigma_min) ** t  for t in [0, 1]."""
    return sigma_min * ((sigma_max / sigma_min)**t)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
def loss_function(net, batch):
    """Denoising score matching loss for the VE-SDE with sigma^2 weighting."""
    B = batch.shape[0]
    t = torch.rand(B, device=batch.device)        # (B,) continuous time

    eps = torch.randn_like(batch)                  # (B, L)
    sigma_t = sigma(t).unsqueeze(-1)               # (B, 1), for broadcasting

    X_t = batch + sigma_t * eps                    # (B, L)

    score = net(X_t, t)                            # network is conditioned on t, shape (B,)

    # target score is -eps/sigma; with sigma^2 weighting this reduces to:
    return (score * sigma_t + eps).pow(2).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_net(X0, net, train_config, model_config, seed=0):
    """Train `net` on the windows `X0` (n_windows, L) and return the loss history.

    train_config keys: iter, batch_size, lr, log_every, drop_lr, save_path.
    A checkpoint (weights, optimiser state, configs, loss history) is written
    to `save_path` at the end of every epoch, overwriting the same file.
    """
    batch_size = train_config.get("batch_size")
    lr = train_config.get("lr")
    n_iters = train_config.get("iter")
    log_every = train_config.get("log_every", 100)
    drop_lr = train_config.get("drop_lr", 80)
    
    device = model_config.get("DEVICE")
    net = net.to(device)
    X0 = X0.to(device)

    print(f"Using device: {device}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    losses = []
    n = len(X0)

    print(f"Number of trainable parameters: { sum(p.numel() for p in net.parameters() if p.requires_grad) }")
    for epoch in range(1, n_iters + 1):
        indices = torch.randperm(n, device=device)
        
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = X0[batch_indices]

            opt.zero_grad(set_to_none=True)

            loss = loss_function(net, batch)

            loss.backward()
            opt.step()

            epoch_loss += loss.item() * len(batch)
        
        epoch_loss /= n
        losses.append(epoch_loss)

        if epoch == 1 or epoch % log_every == 0:
                print(f"Iteration: {epoch}, Loss = {epoch_loss:.5f}")
        
        if epoch == drop_lr:
            for parameter_group in opt.param_groups:
                parameter_group["lr"] /= 10.0

            lr = opt.param_groups[0]["lr"]
            print(f"Dropped learning rate to: {lr}")
    
        # Checkpoint after every epoch (same file, overwritten)
        save_path = train_config.get("save_path", "../checkpoints/scorenet6")
        torch.save({
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch,
            "losses": losses,
            "train_config": train_config,
            "model_config": model_config,
        }, save_path)

    print(f"Saved weights to {save_path}")

    return losses
