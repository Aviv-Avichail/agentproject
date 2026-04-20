import os
import ast
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo import ParallelEnv
from gymnasium import spaces

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set this to False to silence the detailed step-by-step prints during training
DEBUG_MODE = True


# ==============================================================================
# 1. THE REALISTIC CYBER-PHYSICAL ENVIRONMENT WITH FREE LLM & ASYMMETRIC REWARDS
# ==============================================================================
class L2MAID_MARL_Env(ParallelEnv):
    metadata = {"render_modes": [], "name": "l2maid_marl_final"}

    # Class-level cache to persist across episodes and drastically speed up training
    llm_cache = {}

    def __init__(self):
        super().__init__()
        self.possible_agents = ["monitor_agent", "mitigator_agent"]
        self.agents = self.possible_agents.copy()

        # Observation: [Level, Pump, Valve, Threat_Prob, Normal_Prob, Isolate_Prob]
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            "monitor_agent": spaces.Discrete(2),  # 0=Idle, 1=Alert
            # 0=Idle, 1=Port Isolate, 2=Quarantine, 3=Full Shutdown
            "mitigator_agent": spaces.Discrete(4)
        }

        self.max_steps = 60
        self.setpoint = 50.0

        # Reward weights per the paper
        self.w_sec = 1.0
        self.w_proc = 2.0

        # Local Ollama configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.llm_model = "llama3.2:1b"

    def _get_llm_context(self):
        """Queries the local LLM to translate telemetry into semantic state."""
        rounded_level = round(self.level, -1)
        cache_key = f"P:{self.pump}_V:{self.valve}_L:{rounded_level}"

        if cache_key in self.__class__.llm_cache:
            return self.__class__.llm_cache[cache_key]

        prompt = f"""
        Analyze this Industrial log:
        Level: {rounded_level}
        Pump: {'ON' if self.pump == 1 else 'OFF'}
        Valve: {'OPEN' if self.valve == 1 else 'CLOSED'}

        Rule: If Level is rising AND Pump is ON AND Valve is CLOSED, this is a cyber attack.
        Output exactly a python list of three floats (Threat, Normal, Isolate): [0.9, 0.1, 0.9] if attack, [0.1, 0.9, 0.1] if normal.
        Only output the list, no text.
        """

        if DEBUG_MODE:
            print(f"  [LLM] 🧠 New state encountered ({cache_key}). Querying Ollama...")

        try:
            response = requests.post(self.ollama_url, json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }, timeout=5)

            output_text = response.json()["response"].strip()

            if "[" in output_text and "]" in output_text:
                output_text = output_text[output_text.find("["):output_text.find("]") + 1]

            semantic_vector = ast.literal_eval(output_text)

            if len(semantic_vector) == 3:
                self.__class__.llm_cache[cache_key] = semantic_vector
                if DEBUG_MODE:
                    print(f"  [LLM] ✅ Ollama returned: {semantic_vector}. Saved to cache.")
                return semantic_vector
            else:
                raise ValueError("Bad vector")

        except Exception as e:
            if DEBUG_MODE:
                print(f"  [LLM] ⚠️ Ollama failed or timed out. Falling back to default vector.")
            return [0.5, 0.5, 0.0]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.level = 50.0
        self.pump = 1
        self.valve = 1
        self.raw_anomaly = 0.0
        self.steps = 0
        self.attack_trigger_step = np.random.randint(10, 40)

        if DEBUG_MODE:
            print(f"[ENV] 🔄 Environment Reset. Attack scheduled for step {self.attack_trigger_step}.")

        return {a: self._get_obs() for a in self.agents}, {a: {} for a in self.agents}

    def _get_obs(self):
        return np.array([self.level / 100.0, self.pump, self.valve] + self._get_llm_context(), dtype=np.float32)

    def step(self, actions):
        if not self.agents: return {}, {}, {}, {}, {}

        mon_act = actions.get("monitor_agent", 0)
        mit_act = actions.get("mitigator_agent", 0)

        # 1. Baseline PLC Logic
        if self.raw_anomaly == 0.0:
            if self.level < 40.0:
                self.pump, self.valve = 1, 0
            elif self.level > 60.0:
                self.pump, self.valve = 0, 1

        # 2. Adversarial Attack
        if self.steps == self.attack_trigger_step:
            self.raw_anomaly = 1.0
            if DEBUG_MODE:
                print(f"  [ENV] 🚨 ADVERSARIAL ATTACK TRIGGERED AT STEP {self.steps}! Pump forced ON, Valve CLOSED.")

        if self.raw_anomaly == 1.0:
            self.pump, self.valve = 1, 0

            # 3. Physical Dynamics
        if self.pump == 1: self.level += 3.0
        if self.valve == 1: self.level -= 3.0
        self.level += np.random.normal(0, 0.8)

        # ==========================================
        # 4. ASYMMETRIC TIERED REWARD FUNCTION
        # ==========================================
        r_sec = 0.0
        r_proc = -abs(self.level - self.setpoint) / 20.0

        if self.raw_anomaly == 1.0:
            # TRUE POSITIVE: Actual attack underway
            if mon_act == 1 and mit_act > 0:
                self.raw_anomaly = 0.0  # Attack Neutralized!
                if DEBUG_MODE:
                    print(f"  [AGENTS] 🛡️ Attack neutralized using Mitigation Level {mit_act} at step {self.steps}!")

                # Scaled Success Rewards
                if mit_act == 1:
                    r_sec += 30.0  # Safe, surgical response
                elif mit_act == 2:
                    r_sec += 50.0  # Standard response
                elif mit_act == 3:
                    r_sec += 80.0  # Full Shutdown (High reward, but extreme risk if wrong)

            elif mon_act == 1:
                r_sec += 5.0  # Detected, but mitigator was idle
            else:
                r_sec -= 30.0  # False Negative (Missed the attack entirely)
        else:
            # FALSE POSITIVE: Normal operations, but agents panicked
            if mon_act == 1:
                r_sec -= 15.0  # Monitor raised a false alarm

            # Exponentially Scaled Failure Penalties (The Inverted Ratio)
            if mit_act == 1:
                r_sec -= 10.0  # Mild operational annoyance
            elif mit_act == 2:
                r_sec -= 40.0  # Moderate disruption
            elif mit_act == 3:
                r_sec -= 180.0  # MASSIVE penalty for unnecessary full shutdown

        # Physical consequence of a Full Shutdown
        if mit_act == 3:
            self.pump, self.valve = 0, 0
            if DEBUG_MODE:
                print(f"  [ENV] 🛑 FULL SYSTEM SHUTDOWN initiated by Mitigator!")

        total_reward = (self.w_sec * r_sec) + (self.w_proc * r_proc)

        self.level = np.clip(self.level, 0.0, 100.0)
        self.steps += 1

        terminated = self.level >= 100.0 or self.level <= 0.0 or self.steps >= self.max_steps
        if terminated:
            if self.level >= 100 or self.level <= 0:
                # Catastrophic penalty must be worse than a false shutdown
                total_reward -= 300.0
                if DEBUG_MODE:
                    print(f"  [ENV] 💥 CATASTROPHIC FAILURE! Tank level reached critical bounds ({self.level:.1f}).")
            elif DEBUG_MODE:
                print(f"  [ENV] 🏁 Episode reached max steps safely.")

        rewards = {a: float(total_reward) for a in self.agents}
        terminations = {a: terminated for a in self.agents}

        if terminated: self.agents = []
        return {a: self._get_obs() for a in self.possible_agents}, rewards, terminations, {a: False for a in
                                                                                           self.possible_agents}, {a: {}
                                                                                                                   for a
                                                                                                                   in
                                                                                                                   self.possible_agents}


# ==============================================================================
# 2. NEURAL NETWORK & AGENT (PyTorch)
# ==============================================================================
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
        print(f"[AGENT] 💾 Saved model to {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            self.q_net.load_state_dict(torch.load(filename))
            self.q_net.eval()
            print(f"[AGENT] 📂 Loaded weights from {filename}")
        else:
            print(f"[AGENT] ❌ Error: {filename} not found.")


# ==============================================================================
# 3. TRAINING & GRAPHING DASHBOARD
# ==============================================================================
def plot_comprehensive_dashboard(joint_rewards, mon_rewards, mit_rewards, psi_history):
    print("\n[SYSTEM] 📊 Generating Comprehensive Training Dashboard...")

    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    window = 25  # Slightly larger window for smoothing over 1000 episodes

    def add_trendline(ax, data, color, label):
        if len(data) >= window:
            moving_avg = np.convolve(data, np.ones(window) / window, mode='valid')
            padded_avg = np.concatenate((np.full(window - 1, np.nan), moving_avg))
            ax.plot(padded_avg, color=color, linewidth=2.5, label=label)

    # Plot 1: Joint Reward
    axs[0].plot(joint_rewards, color="lightgray", alpha=0.6, label="Raw Joint Reward")
    add_trendline(axs[0], joint_rewards, "blue", f"{window}-Ep Moving Avg")
    axs[0].set_title("1. Overall Team Performance (Joint Reward)")
    axs[0].set_ylabel("Reward")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend()

    # Plot 2: Agent vs Agent Reward
    axs[1].plot(mon_rewards, color="lightblue", alpha=0.4)
    axs[1].plot(mit_rewards, color="lightcoral", alpha=0.4)
    add_trendline(axs[1], mon_rewards, "dodgerblue", "Monitor Agent Trend")
    add_trendline(axs[1], mit_rewards, "red", "Mitigator Agent Trend")
    axs[1].set_title("2. Individual Agent Contributions")
    axs[1].set_ylabel("Reward")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend()

    # Plot 3: Process Stability (PSI)
    axs[2].plot(psi_history, color="lightgreen", alpha=0.5, label="Raw PSI")
    add_trendline(axs[2], psi_history, "darkgreen", "Process Stability Trend")
    axs[2].set_title("3. Physical System Safety (Process Stability Index)")
    axs[2].set_xlabel("Training Episode")
    axs[2].set_ylabel("PSI Score")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("marl_training_dashboard.png", dpi=300, bbox_inches="tight")
    print("[SYSTEM] ✅ Dashboard saved as 'marl_training_dashboard.png'")
    plt.show()


def train_agents(episodes=1000):  # Increased to 1000 Episodes!
    env = L2MAID_MARL_Env()

    # Slower epsilon_decay forces them to explore the new asymmetric action space longer
    agents = {
        "monitor_agent": DQNAgent(6, 2, epsilon_decay=0.997),
        "mitigator_agent": DQNAgent(6, 4, epsilon_decay=0.997)
    }

    joint_rewards, mon_rewards, mit_rewards, psi_history = [], [], [], []

    print(f"\n{'=' * 50}\n🚀 STARTING DEEP MARL TRAINING ({episodes} Episodes)\n{'=' * 50}")

    for ep in range(episodes):
        obs, _ = env.reset()
        tot_mon, tot_mit = 0, 0
        deviations = []

        while env.agents:
            actions = {a: agents[a].act(obs[a]) for a in env.agents}
            next_obs, rewards, terminations, _, _ = env.step(actions)

            for a in env.agents:
                agents[a].store(obs[a], actions[a], rewards[a], next_obs.get(a, obs[a]), terminations[a])
                agents[a].train_step()

            tot_mon += rewards.get("monitor_agent", 0)
            tot_mit += rewards.get("mitigator_agent", 0)
            if "monitor_agent" in obs:
                deviations.append((obs["monitor_agent"][0] * 100 - 50) ** 2)

            obs = next_obs

        joint_rewards.append((tot_mon + tot_mit) / 2)
        mon_rewards.append(tot_mon)
        mit_rewards.append(tot_mit)

        psi = 1.0 / (np.sqrt(np.mean(deviations)) + 1e-5) if deviations else 0
        psi_history.append(psi)

        if (ep + 1) % 20 == 0:
            avg_j = np.mean(joint_rewards[-20:])
            print(
                f"[TRAIN] Ep {ep + 1}/{episodes} | Avg Joint: {avg_j:.1f} | Eps: {agents['monitor_agent'].epsilon:.3f} | Cache: {len(env.__class__.llm_cache)}")

    print("\n[SYSTEM] 💾 Training complete. Saving trained models...")
    agents["monitor_agent"].save("monitor_model.pth")
    agents["mitigator_agent"].save("mitigator_model.pth")

    return agents, joint_rewards, mon_rewards, mit_rewards, psi_history


# ==============================================================================
# 4. EVALUATION LOOP
# ==============================================================================
def evaluate(trained_agents, episodes=100):
    env = L2MAID_MARL_Env()
    history = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "psi_scores": []}

    print(f"\n{'=' * 50}\n🔬 RUNNING EVALUATION (Exploitation Only)\n{'=' * 50}")

    for ep in range(episodes):
        obs, _ = env.reset()
        deviations = []

        while env.agents:
            actions = {a: trained_agents[a].act(obs[a], explore=False) for a in env.agents}
            is_anomaly = env.raw_anomaly == 1.0

            if is_anomaly:
                if actions["monitor_agent"] == 1:
                    history["tp"] += 1
                else:
                    history["fn"] += 1
            else:
                if actions["monitor_agent"] == 1:
                    history["fp"] += 1
                else:
                    history["tn"] += 1

            obs, rewards, term, trunc, _ = env.step(actions)
            if "monitor_agent" in obs:
                deviations.append((obs["monitor_agent"][0] * 100 - 50) ** 2)

        psi = 1.0 / (np.sqrt(np.mean(deviations)) + 1e-5) if deviations else 0
        history["psi_scores"].append(psi)

        if (ep + 1) % 10 == 0:
            print(f"[EVAL] Completed evaluation episode {ep + 1}/{episodes}")

    tp, fp, fn, tn = history['tp'], history['fp'], history['fn'], history['tn']
    dr = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Detection Rate (DR):           {dr:.2f}%")
    print(f"False Positive Rate (FPR):     {fpr:.2f}%")
    print(f"Average Process Stability:     {np.mean(history['psi_scores']):.4f}")
    print("=" * 50)


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # SET TO FALSE ONCE YOU HAVE TRAINED THE MODELS
    TRAIN_NEW_MODELS = True

    if TRAIN_NEW_MODELS:
        # Warning: This will be very noisy in the console because of DEBUG_MODE = True
        # You can change DEBUG_MODE = False at the top of the file if you want a cleaner printout
        models, j_rews, mon_rews, mit_rews, psi_hist = train_agents(episodes=1000)

        plot_comprehensive_dashboard(j_rews, mon_rews, mit_rews, psi_hist)

        # Turn off debug mode for evaluation so the results print cleanly
        DEBUG_MODE = False
        evaluate(models)
    else:
        print("\n[SYSTEM] Skipping training. Loading models...")
        DEBUG_MODE = False
        loaded_agents = {"monitor_agent": DQNAgent(6, 2), "mitigator_agent": DQNAgent(6, 4)}
        loaded_agents["monitor_agent"].load("monitor_model.pth")
        loaded_agents["mitigator_agent"].load("mitigator_model.pth")
        evaluate(loaded_agents)