"""Configuration File for training the Score Network."""

data_config = dict(
    TICKER      = "^GSPC", # SNP 500
    START       = "1957-01-01",
    END         = "2026-06-16",
    WINDOW_SIZE = 256,
    STRIDE      = 1,
)

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Network Parameters
model_config = dict(
    CHANNELS       = 64,                              # conv channels C (vanilla CSDI = 64)
    DIFFUSION_DIM  = 128,                              # diffusion-step embedding width
    NUM_HEADS      = 8,                                # MUST divide CHANNELS, since transformer d_model = C
    NUM_TF_LAYERS  = 1,                                # transformer layers inside one temporal block
    NUM_RES_BLOCKS = 2,                                # the "x n" gated residual blocks (n = 4 in the paper)
    MAX_LEN        = data_config.get("WINDOW_SIZE"),   # sliding-window length L (upper bound for pos tables)
    KERNEL_SIZE    = 3,
    DEVICE         = device
)

