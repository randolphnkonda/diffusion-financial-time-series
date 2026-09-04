"""GRU time series discriminator network."""

import torch
import torch.nn as nn

class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, hidden = self.gru(x)

        # Last GRU layer hidden state
        hidden = hidden[-1]
        
        output = self.classifier(hidden)
        return output