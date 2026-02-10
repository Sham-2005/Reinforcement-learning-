import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Environment
GRID_SIZE = 5
ACTIONS = 4  # up, down, left, right

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, ACTIONS),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

def move(pos, action):
    x, y = pos
    if action == 0: x -= 1
    if action == 1: x += 1
    if action == 2: y -= 1
    if action == 3: y += 1
    return max(0, min(4, x)), max(0, min(4, y))

for episode in range(500):
    pos = (0, 0)
    log_probs = []
    rewards = []

    for step in range(30):
        state = torch.tensor(pos, dtype=torch.float32)
        probs = policy(state)
        action = torch.distributions.Categorical(probs).sample()
        log_probs.append(torch.log(probs[action]))

        pos = move(pos, action.item())
        reward = 1  # reward for cleaning
        rewards.append(reward)

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)

    loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training completed for Cleaning Robot.")
