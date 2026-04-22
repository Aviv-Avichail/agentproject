import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        # 3-layer MLP, closer to the paper
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)


class CentralizedCritic(nn.Module):
    def __init__(self, global_obs_dim, hidden_dim=128):
        super().__init__()
        # 3-layer MLP, closer to the paper
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_obs):
        return self.net(global_obs)


class MAPPO_CTDE:
    def __init__(
        self,
        obs_dims,
        act_dims,
        actor_lr=5e-4,
        critic_lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ppo_epochs=10,
        entropy_coef=0.01,
        max_grad_norm=0.5
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.agents = list(obs_dims.keys())

        self.actors = nn.ModuleDict({
            agent: Actor(obs_dims[agent], act_dims[agent]) for agent in self.agents
        })
        self.actor_optimizers = {
            agent: optim.Adam(self.actors[agent].parameters(), lr=actor_lr)
            for agent in self.agents
        }

        global_obs_dim = sum(obs_dims.values())
        self.critic = CentralizedCritic(global_obs_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = []

    def get_actions(self, obs_dict):
        actions = {}
        log_probs = {}

        for agent in self.agents:
            if agent not in obs_dict:
                continue

            obs_tensor = torch.as_tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = self.actors[agent](obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            actions[agent] = int(action.item())
            log_probs[agent] = float(log_prob.item())

        return actions, log_probs

    def store(self, global_obs, obs_dict, actions, log_probs, rewards, done):
        self.memory.append({
            "global_obs": np.array(global_obs, dtype=np.float32),
            "obs_dict": {k: np.array(v, dtype=np.float32) for k, v in obs_dict.items()},
            "actions": dict(actions),
            "log_probs": dict(log_probs),
            "rewards": dict(rewards),
            "done": bool(done)
        })

    def _shared_reward_sequence(self):
        shared_rewards = []
        for transition in self.memory:
            if transition["rewards"]:
                shared_rewards.append(float(np.mean(list(transition["rewards"].values()))))
            else:
                shared_rewards.append(0.0)
        return np.array(shared_rewards, dtype=np.float32)

    def _compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self):
        if not self.memory:
            return

        global_obs_tensor = torch.as_tensor(
            np.stack([m["global_obs"] for m in self.memory]),
            dtype=torch.float32
        )
        dones = np.array([float(m["done"]) for m in self.memory], dtype=np.float32)
        rewards = self._shared_reward_sequence()

        with torch.no_grad():
            old_values = self.critic(global_obs_tensor).squeeze(-1).cpu().numpy()

        advantages, returns = self._compute_gae(rewards, old_values, dones)

        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)

        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            # Critic update
            values = self.critic(global_obs_tensor).squeeze(-1)
            critic_loss = nn.MSELoss()(values, returns_tensor)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # Actor updates
            for agent in self.agents:
                valid_indices = [i for i, m in enumerate(self.memory) if agent in m["obs_dict"]]
                if not valid_indices:
                    continue

                obs_tensor = torch.as_tensor(
                    np.stack([self.memory[i]["obs_dict"][agent] for i in valid_indices]),
                    dtype=torch.float32
                )
                actions_tensor = torch.as_tensor(
                    [self.memory[i]["actions"][agent] for i in valid_indices],
                    dtype=torch.long
                )
                old_log_probs_tensor = torch.as_tensor(
                    [self.memory[i]["log_probs"][agent] for i in valid_indices],
                    dtype=torch.float32
                )
                agent_advantages = advantages_tensor[valid_indices]

                dist = self.actors[agent](obs_tensor)
                new_log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                surr1 = ratio * agent_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * agent_advantages

                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optimizers[agent].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.max_grad_norm)
                self.actor_optimizers[agent].step()

        self.memory.clear()

    def save(self, directory="models", prefix="l2maid"):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.critic.state_dict(), f"{directory}/{prefix}_critic.pth")
        for agent in self.agents:
            torch.save(self.actors[agent].state_dict(), f"{directory}/{prefix}_actor_{agent}.pth")