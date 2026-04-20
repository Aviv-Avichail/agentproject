import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# ==============================================================================
# 1. ENVIRONMENT DEFINITION (Normalized)
# ==============================================================================
class L2MAID_Env(gym.Env):
    def __init__(self):
        super(L2MAID_Env, self).__init__()

        # FIX 1: The Observation space must be normalized between 0.0 and 1.0
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 3])

        self.max_steps = 100
        self.setpoint = 50.0

        # Internal state tracking
        self.level = 50.0
        self.pump = 1
        self.valve = 0
        self.anomaly = 0.0
        self.steps = 0
        self.attack_trigger_step = 0
        self.attack_has_occurred = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.level = 50.0
        self.pump = 1
        self.valve = 0
        self.anomaly = 0.0
        self.steps = 0

        # Attack happens randomly once per episode
        self.attack_trigger_step = np.random.randint(15, 80)
        self.attack_has_occurred = False

        return self._get_obs(), {}

    def _get_obs(self):
        # Return the normalized observation array
        return np.array([self.level / 100.0, self.pump, self.valve, self.anomaly], dtype=np.float32)

    def step(self, actions):
        monitor_action, mitigator_action = actions
        reward = 0.0

        # --- 1. ADVERSARY LOGIC ---
        if self.steps == self.attack_trigger_step and not self.attack_has_occurred:
            self.anomaly = 1.0
            self.attack_has_occurred = True

        # --- 2. THE LOUD, UNDENIABLE REWARD SYSTEM ---
        if self.anomaly == 1.0:
            if monitor_action == 1:
                if mitigator_action > 0:
                    reward += 100.0  # MASSIVE WIN: Threat spotted and neutralized
                    self.anomaly = 0.0
                else:
                    reward += 10.0  # Spotted, but mitigator is asleep
            else:
                reward -= 20.0  # Terrible: Missed the attack entirely
        else:
            if monitor_action == 1:
                reward -= 20.0  # Terrible: False alarm paranoia
            if mitigator_action > 0:
                reward -= 20.0  # Terrible: Touching valves when safe

        # --- 3. PHYSICS ---
        self.level += np.random.normal(0, 0.1)
        if self.anomaly == 1.0 and self.steps > self.attack_trigger_step:
            self.level += 10.0  # Hacker rapidly floods the system

        self.level = np.clip(self.level, 0.0, 100.0)

        # --- 4. TERMINATION & PENALTIES ---
        terminated = False
        if self.level >= 100.0 or self.level <= 0.0:
            reward -= 500.0  # Catastrophic physical failure
            terminated = True
        elif self.steps >= self.max_steps:
            terminated = True

        self.steps += 1

        return self._get_obs(), float(reward), terminated, False, {}


# ==============================================================================
# 2. TRAINING AND EVALUATION SCRIPT
# ==============================================================================
def main():
    print(">>> Initializing Normalized L2M-AID Environment...")
    env = L2MAID_Env()
    check_env(env)

    print("\n>>> Training Tactical Agents via MAPPO/PPO...")

    # Standard learning rate is fine now that the state is normalized
    model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=1024, gamma=0.99, verbose=0)

    model.learn(total_timesteps=150000)
    print(">>> Training Complete. Agents are now ready for deployment.")

    print("\n>>> Running Evaluation & Generating Report Metrics...")
    episodes = 100

    history = {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "response_times": [],
        "psi_scores": [],
    }

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        anomaly_start_step = -1
        tank_deviations = []

        while not terminated:
            action, _states = model.predict(obs, deterministic=True)
            monitor_action, mitigator_action = action

            # obs[3] is the anomaly flag (0.0 or 1.0)
            current_anomaly = obs[3]

            next_state, reward, terminated, _, _ = env.step(action)

            # next_state[0] is normalized, multiply by 100 to get real level for PSI
            actual_level = next_state[0] * 100.0
            tank_deviations.append((actual_level - env.setpoint) ** 2)

            if current_anomaly == 1.0:
                if anomaly_start_step == -1:
                    anomaly_start_step = env.steps - 1

                if monitor_action == 1:
                    history["tp"] += 1
                    if mitigator_action > 0:
                        history["response_times"].append(env.steps - anomaly_start_step)
                        anomaly_start_step = -1
                else:
                    history["fn"] += 1
            else:
                if monitor_action == 1:
                    history["fp"] += 1
                else:
                    history["tn"] += 1

            obs = next_state

        rmse = np.sqrt(np.mean(tank_deviations))
        psi = 1.0 / (rmse + 1e-5)
        history["psi_scores"].append(psi)

    # --- Math for the Final Technical Report ---
    tp, fp, fn, tn = history["tp"], history["fp"], history["fn"], history["tn"]

    dr = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
    mttr = np.mean(history["response_times"]) if history["response_times"] else 0
    avg_psi = np.mean(history["psi_scores"])

    print("\n" + "=" * 50)
    print("FINAL EXPERIMENTAL RESULTS (FOR YOUR REPORT)")
    print("=" * 50)
    print(f"Detection Rate (DR):             {dr:.2f}%")
    print(f"False Positive Rate (FPR):       {fpr:.2f}%")
    print(f"Mean Time to Respond (MTTR):     {mttr:.2f} steps")
    print(f"Process Stability Index (PSI):   {avg_psi:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()