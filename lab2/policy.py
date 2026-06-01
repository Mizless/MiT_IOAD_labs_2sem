
import torch
import torch.nn as nn

from lab2.config import ACTION_SPACE, INPUT_DIM

class Policy(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden1: int = 512,
        hidden2: int = 512,
        hidden3: int = 256,
        output_dim: int = ACTION_SPACE,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden3, output_dim)
        self.critic = nn.Linear(hidden3, 1)

    def forward(self, x):
        hidden = self.shared(x)
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return logits, value
