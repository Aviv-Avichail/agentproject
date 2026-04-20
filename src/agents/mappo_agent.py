import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    def __init__(self, global_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_obs):
        return self.net(global_obs)


class MAPPO_CTDE:
    def __init__(self, obs_dims, act_dims, lr=3e-4, gamma=0.99, clip_eps=0.2, ppo_epochs=4):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.agents = list(obs_dims.keys())

        self.actors = nn.ModuleDict({
            agent: Actor(obs_dims[agent], act_dims[agent]) for agent in self.agents
        })
        self.actor_optimizers = {
            agent: optim.Adam(self.actors[agent].parameters(), lr=lr) for agent in self.agents
        }

        global_obs_dim = sum(obs_dims.values())
        self.critic = CentralizedCritic(global_obs_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []

    def get_actions(self, obs_dict):
        actions, log_probs = {}, {}
        for agent in self.agents:
            if agent in obs_dict:
                obs_tensor = torch.FloatTensor(obs_dict[agent]).unsqueeze(0)
                with torch.no_grad():
                    dist = self.actors[agent](obs_tensor)
                    action = dist.sample()
                    log_probs[agent] = dist.log_prob(action).item()
                actions[agent] = action.item()
        return actions, log_probs

    def store(self, global_obs, obs_dict, actions, log_probs, rewards, done):
        self.memory.append((global_obs, obs_dict, actions, log_probs, rewards, done))

    def update(self):
        if len(self.memory) == 0: return

        global_obs_list = [m[0] for m in self.memory]
        rewards_list = {a: [m[4].get(a, 0) for m in self.memory] for a in self.agents}
        dones_list = [m[5] for m in self.memory]

        returns = {a: [] for a in self.agents}
        for a in self.agents:
            R = 0
            for r, d in zip(reversed(rewards_list[a]), reversed(dones_list)):
                if d: R = 0
                R = r + self.gamma * R
                returns[a].insert(0, R)

        global_obs_tensor = torch.FloatTensor(np.array(global_obs_list))

        for _ in range(self.ppo_epochs):
            values = self.critic(global_obs_tensor).squeeze()
            mean_returns = torch.FloatTensor(
                [np.mean([returns[a][i] for a in self.agents]) for i in range(len(self.memory))])

            critic_loss = nn.MSELoss()(values, mean_returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            advantages = (mean_returns - values.detach()).unsqueeze(1)

            for agent in self.agents:
                obs_tensor = torch.FloatTensor(np.array([m[1][agent] for m in self.memory if agent in m[1]]))
                old_log_probs = torch.FloatTensor([m[3][agent] for m in self.memory if agent in m[3]]).unsqueeze(1)
                actions_tensor = torch.FloatTensor([m[2][agent] for m in self.memory if agent in m[2]]).unsqueeze(1)

                if len(obs_tensor) == 0: continue

                dist = self.actors[agent](obs_tensor)
                new_log_probs = dist.log_prob(actions_tensor.squeeze(-1)).unsqueeze(1)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizers[agent].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[agent].step()

        self.memory.clear()

    def save(self, directory="models", prefix="l2maid"):
        import os
        os.makedirs(directory, exist_ok=True)
        torch.save(self.critic.state_dict(), f"{directory}/{prefix}_critic.pth")
        for agent in self.agents:
            torch.save(self.actors[agent].state_dict(), f"{directory}/{prefix}_actor_{agent}.pth")