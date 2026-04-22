import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from collections import deque


class L2MAID_MARL_Env(ParallelEnv):
    metadata = {"render_modes": [], "name": "l2maid_marl_4agent_parity"}

    def __init__(self, llm_client, debug_mode=False):
        super().__init__()
        self.llm_client = llm_client
        self.debug_mode = debug_mode

        # 4-Agent Tactical Layer
        self.possible_agents = ["network_agent", "host_agent", "threat_intel_agent", "mitigator_agent"]
        self.agents = self.possible_agents.copy()

        self.observation_spaces = {
            "network_agent": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            "host_agent": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            "threat_intel_agent": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            "mitigator_agent": spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        }

        self.action_spaces = {
            "network_agent": spaces.Discrete(2),
            "host_agent": spaces.Discrete(2),
            "threat_intel_agent": spaces.Discrete(2),
            "mitigator_agent": spaces.Discrete(4)
        }

        self.max_steps = 60
        self.setpoint = 50.0
        self.level_history = deque(maxlen=5)

        # Closer to the paper's reward structure:
        # R = wsec * Rsecurity + wproc * Rprocess + wcost * Rcost
        self.w_sec = 1.0
        self.w_proc = 2.0
        self.w_cost = 0.1

        # Simplified but paper-aligned reward component constants
        self.detector_true_positive_bonus = 0.5
        self.mitigator_true_positive_reward = {
            1: 8.0,   # isolate / mild containment
            2: 10.0,  # quarantine
            3: 12.0   # full shutdown
        }
        self.missed_attack_penalty = 6.0

        self.detector_false_positive_cost = 1.0
        self.mitigator_false_positive_cost = {
            1: 12.0,
            2: 25.0,
            3: 80.0
        }

        self.catastrophic_process_penalty = 50.0

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()

        self.level = 50.0
        self.pump = 1
        self.valve = 1
        self.cpu_load = 25.0
        self.ioc_match = 0.0

        self.level_history.clear()
        for _ in range(5):
            self.level_history.append(self.level)

        self.raw_anomaly = 0.0
        self.steps = 0
        self.attack_trigger_step = np.random.randint(10, 40)
        self.attack_type = np.random.choice(["overflow", "stealth_drain", "ransomware"])

        self.last_mit_act = 0
        self.comm_buffer = [0.0, 0.0, 0.0]

        return self._get_obs(), {a: {} for a in self.agents}

    def _get_obs(self):
        start_level = self.level_history[0]
        current_level = self.level_history[-1]
        delta = current_level - start_level

        if delta < -2.0:
            trend = "DROPPING (Unsafe)"
        elif delta > 2.0:
            trend = "RISING (Unsafe)"
        else:
            trend = "STABLE"

        llm_vector = self.llm_client.get_context(
            self.level,
            self.pump,
            self.valve,
            self.cpu_load,
            self.ioc_match,
            self.last_mit_act,
            trend
        )

        obs = {
            "network_agent": np.array([self.pump, self.valve] + self.comm_buffer + llm_vector, dtype=np.float32),
            "host_agent": np.array(
                [self.level / 100.0, self.cpu_load / 100.0] + self.comm_buffer + llm_vector,
                dtype=np.float32
            ),
            "threat_intel_agent": np.array([self.ioc_match] + self.comm_buffer + llm_vector, dtype=np.float32),
            "mitigator_agent": np.array(
                [self.level / 100.0, self.pump, self.valve, self.cpu_load / 100.0, self.ioc_match]
                + self.comm_buffer + llm_vector,
                dtype=np.float32
            )
        }
        return obs

    def _apply_normal_control_policy(self):
        self.valve = 1
        if self.level < 40.0:
            self.pump = 1
        elif self.level > 60.0:
            self.pump = 0

    def _restore_post_mitigation_state(self, mit_act):
        # Immediately clear attack indicators so the next observation is consistent
        self.raw_anomaly = 0.0
        self.ioc_match = 0.0
        self.cpu_load = 25.0 + np.random.normal(0, 1.0)

        if mit_act == 3:
            # Full shutdown
            self.pump = 0
            self.valve = 0
        else:
            # Return to normal control regime
            self._apply_normal_control_policy()

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        net_act = actions.get("network_agent", 0)
        host_act = actions.get("host_agent", 0)
        intel_act = actions.get("threat_intel_agent", 0)
        mit_act = actions.get("mitigator_agent", 0)

        self.last_mit_act = mit_act
        self.comm_buffer = [float(net_act), float(host_act), float(intel_act)]

        # Normal plant dynamics before attack injection
        if self.raw_anomaly == 0.0:
            self._apply_normal_control_policy()
            self.cpu_load = 25.0 + np.random.normal(0, 3.0)
            self.ioc_match = 0.0

        # Trigger attack at the chosen step
        if self.steps == self.attack_trigger_step:
            self.raw_anomaly = 1.0

        # Attack dynamics
        if self.raw_anomaly == 1.0:
            if self.attack_type == "overflow":
                self.pump = 1
                self.valve = 0
                self.cpu_load = 88.0 + np.random.normal(0, 5.0)
                self.ioc_match = 1.0

            elif self.attack_type == "stealth_drain":
                self.pump = 0
                self.valve = 1
                self.cpu_load = 28.0 + np.random.normal(0, 2.0)
                self.ioc_match = 1.0

            elif self.attack_type == "ransomware":
                self.pump = 0
                self.valve = 0
                self.cpu_load = 99.0
                self.ioc_match = 1.0

        # Physical process evolution
        if self.pump == 1:
            self.level += 3.0
        if self.valve == 1:
            self.level -= 3.0
        self.level += np.random.normal(0, 0.8)
        self.level = np.clip(self.level, 0.0, 100.0)

        self.level_history.append(self.level)

        # Explicit paper-style reward decomposition
        r_security = 0.0
        r_cost = 0.0

        # Process stability term: penalize deviation from safe setpoint
        # Higher magnitude when the process drifts far from safe operation
        deviation = (self.level - self.setpoint) / 25.0
        r_process = -(deviation ** 2)

        if self.raw_anomaly == 1.0:
            if net_act == 1:
                r_security += self.detector_true_positive_bonus
            if host_act == 1:
                r_security += self.detector_true_positive_bonus
            if intel_act == 1:
                r_security += self.detector_true_positive_bonus

            if mit_act > 0:
                r_security += self.mitigator_true_positive_reward.get(mit_act, 0.0)
                self._restore_post_mitigation_state(mit_act)
                # Small action-efficiency cost even for correct response
                r_cost -= {1: 1.0, 2: 2.0, 3: 6.0}.get(mit_act, 0.0)
            else:
                r_security -= self.missed_attack_penalty

        else:
            if net_act == 1:
                r_cost -= self.detector_false_positive_cost
            if host_act == 1:
                r_cost -= self.detector_false_positive_cost
            if intel_act == 1:
                r_cost -= self.detector_false_positive_cost

            if mit_act > 0:
                r_cost -= self.mitigator_false_positive_cost.get(mit_act, 0.0)

        # Catastrophic physical failure
        terminated = self.level >= 100.0 or self.level <= 0.0 or self.steps + 1 >= self.max_steps
        if self.level >= 100.0 or self.level <= 0.0:
            r_process -= self.catastrophic_process_penalty

        total_reward = (
            self.w_sec * r_security +
            self.w_proc * r_process +
            self.w_cost * r_cost
        )

        self.steps += 1

        rewards = {a: float(total_reward) for a in self.agents}
        terminations = {a: terminated for a in self.agents}

        if terminated:
            self.agents = []

        return (
            self._get_obs(),
            rewards,
            terminations,
            {a: False for a in self.possible_agents},
            {a: {} for a in self.possible_agents}
        )