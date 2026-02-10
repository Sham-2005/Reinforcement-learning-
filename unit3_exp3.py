import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

state_dim = 5  # robot_x, robot_y, item_x, item_y, obstacle_dist
action_dim = 4  # up, down, left, right

policy = PPOPolicy(state_dim, action_dim)

state = torch.randn(1, state_dim)
action_probs = policy(state)
dist = Categorical(action_probs)
action = dist.sample()

print("Action chosen:", action.item())
