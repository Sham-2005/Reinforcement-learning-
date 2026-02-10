import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for episode in range(300):
    state = torch.tensor([0.0, 0.0])  # traffic light, position
    total_reward = 0

    for step in range(20):
        probs, value = model(state)
        action = torch.distributions.Categorical(probs).sample()

        reward = -1
        next_state = torch.tensor([1.0, 1.0]) if action.item() == 1 else state

        _, next_value = model(next_state)
        advantage = reward + 0.99 * next_value - value

        actor_loss = -torch.log(probs[action]) * advantage.detach()
        critic_loss = advantage.pow(2)

        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

print("Self-driving car training completed.")
