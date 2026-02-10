import gym
import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# Example state: [x, y, distance_to_target]
state_dim = 3
action_dim = 2  # velocity_x, velocity_y

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)

state = torch.randn(1, state_dim)
action = actor(state)
q_value = critic(state, action)

print("Action:", action)
print("Q-value:", q_value)
