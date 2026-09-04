"""Architectural defaults for the model package.

These constants are the DEFAULT constructor arguments of the classes in
``model/`` (ScoreNet, ResidualBlock, TemporalTransformer, embeddings). They
are not the run configuration: the notebooks pass explicit values from the
top-level ``config.py`` (``model_config``), and sampling/evaluation rebuild
the network from the configuration stored in each checkpoint. Edit the
top-level ``config.py`` to change a training run; edit this file only to
change the package's built-in defaults.
"""

import torch

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CHANNELS       = 64    # conv channels C (vanilla CSDI = 64)
DIFFUSION_DIM  = 128   # diffusion-step embedding width
NUM_HEADS      = 8     # MUST divide CHANNELS, since transformer d_model = C
NUM_TF_LAYERS  = 1     # transformer layers inside one temporal block
NUM_RES_BLOCKS = 4     # the "x n" gated residual blocks (n = 4 in the paper)
MAX_LEN        = 256   # sliding-window length L (upper bound for pos tables)
KERNEL_SIZE    = 3

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
