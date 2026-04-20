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
        self.w_sec = 1.0
        self.w_proc = 2.0

        self.level_history = deque(maxlen=5)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()

        self.level, self.pump, self.valve = 50.0, 1, 1
        self.cpu_load = 25.0
        self.ioc_match = 0.0

        self.level_history.clear()
        for _ in range(5):
            self.level_history.append(self.level)

        self.raw_anomaly, self.steps = 0.0, 0
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
            self.level, self.pump, self.valve, self.cpu_load, self.ioc_match, self.last_mit_act, trend
        )

        obs = {
            "network_agent": np.array([self.pump, self.valve] + self.comm_buffer + llm_vector, dtype=np.float32),
            "host_agent": np.array([self.level / 100.0, self.cpu_load / 100.0] + self.comm_buffer + llm_vector,
                                   dtype=np.float32),
            "threat_intel_agent": np.array([self.ioc_match] + self.comm_buffer + llm_vector, dtype=np.float32),
            "mitigator_agent": np.array([self.level / 100.0, self.pump, self.valve, self.cpu_load / 100.0,
                                         self.ioc_match] + self.comm_buffer + llm_vector, dtype=np.float32)
        }
        return obs

    def step(self, actions):
        if not self.agents: return {}, {}, {}, {}, {}

        net_act = actions.get("network_agent", 0)
        host_act = actions.get("host_agent", 0)
        intel_act = actions.get("threat_intel_agent", 0)
        mit_act = actions.get("mitigator_agent", 0)

        self.last_mit_act = mit_act
        self.comm_buffer = [float(net_act), float(host_act), float(intel_act)]

        if self.raw_anomaly == 0.0:
            self.valve = 1
            self.cpu_load = 25.0 + np.random.normal(0, 3.0)
            self.ioc_match = 0.0
            if self.level < 40.0:
                self.pump = 1
            elif self.level > 60.0:
                self.pump = 0

        if self.steps == self.attack_trigger_step:
            self.raw_anomaly = 1.0

        if self.raw_anomaly == 1.0:
            if self.attack_type == "overflow":
                self.pump = 1;
                self.valve = 0
                self.cpu_load = 88.0 + np.random.normal(0, 5.0);
                self.ioc_match = 1.0
            elif self.attack_type == "stealth_drain":
                self.pump = 0;
                self.valve = 1
                self.cpu_load = 28.0 + np.random.normal(0, 2.0);
                self.ioc_match = 1.0
            elif self.attack_type == "ransomware":
                self.pump = 0;
                self.valve = 0
                self.cpu_load = 99.0;
                self.ioc_match = 1.0

        if self.pump == 1: self.level += 3.0
        if self.valve == 1: self.level -= 3.0
        self.level += np.random.normal(0, 0.8)

        self.level_history.append(self.level)

        r_sec = 0.0
        r_proc = -abs(self.level - self.setpoint) / 20.0

        if self.raw_anomaly == 1.0:
            if net_act == 1: r_sec += 5.0
            if host_act == 1: r_sec += 5.0
            if intel_act == 1: r_sec += 5.0

            if mit_act > 0:
                self.raw_anomaly = 0.0
                r_sec += {1: 30.0, 2: 50.0, 3: 80.0}.get(mit_act, 0)
            elif (net_act == 0 and host_act == 0 and intel_act == 0):
                r_sec -= 40.0
        else:
            if net_act == 1: r_sec -= 5.0
            if host_act == 1: r_sec -= 5.0
            if intel_act == 1: r_sec -= 5.0
            r_sec -= {1: 10.0, 2: 40.0, 3: 180.0}.get(mit_act, 0)

        if mit_act == 3:
            self.pump, self.valve = 0, 0

        total_reward = (self.w_sec * r_sec) + (self.w_proc * r_proc)
        self.level = np.clip(self.level, 0.0, 100.0)
        self.steps += 1

        terminated = self.level >= 100.0 or self.level <= 0.0 or self.steps >= self.max_steps
        if terminated and (self.level >= 100 or self.level <= 0):
            total_reward -= 300.0

        rewards = {a: float(total_reward) for a in self.agents}
        terminations = {a: terminated for a in self.agents}

        if terminated: self.agents = []
        return self._get_obs(), rewards, terminations, {a: False for a in self.possible_agents}, {a: {} for a in
                                                                                                  self.possible_agents}