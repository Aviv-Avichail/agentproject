import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.95, epsilon_decay=0.997):
        self.q_net = QNetwork(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        self.act_dim = act_dim

    def act(self, obs, explore=True):
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.act_dim)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return torch.argmax(q_values).item()

    def store(self, obs, act, rew, next_obs, done):
        self.memory.append((obs, act, rew, next_obs, done))

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size: return

        batch = random.sample(self.memory, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)

        obs = torch.FloatTensor(np.array(obs))
        acts = torch.LongTensor(acts).unsqueeze(1)
        rews = torch.FloatTensor(rews).unsqueeze(1)
        next_obs = torch.FloatTensor(np.array(next_obs))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_net(obs).gather(1, acts)
        with torch.no_grad():
            max_next_q = self.q_net(next_obs).max(1)[0].unsqueeze(1)
            target_q = rews + (self.gamma * max_next_q * (1 - dones))

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.q_net.state_dict(), filename)

    def load(self, filename):
        if os.path.exists(filename):
            self.q_net.load_state_dict(torch.load(filename))
            self.q_net.eval()