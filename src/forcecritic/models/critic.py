"""Independent twin return/cost critics; observation features are fixed ACT0 features."""
from torch import nn
import torch


class TwinCritic(nn.Module):
    def __init__(self, condition_dim, width=256):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(condition_dim + 16, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 2)) for _ in range(2)])

    def forward(self, condition, action):
        x = torch.cat((condition.detach(), action.flatten(1)), -1)
        return torch.stack([head(x) for head in self.heads], 1)  # B, twin, R/C


def condition_features(frozen_act0, obs, remaining, controller):
    with torch.no_grad():
        visual = frozen_act0.visual(obs['rgb']).mean(1)
        return torch.cat((visual, obs['state'], obs['history'].flatten(1),
                          remaining.reshape(-1, 1), controller), -1).detach()
