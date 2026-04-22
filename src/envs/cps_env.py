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
        self.safe_low = 48.0
        self.safe_high = 52.0
        self.recovery_hold_steps = 2
        self.level_history = deque(maxlen=5)

        # Paper-style reward decomposition
        self.w_sec = 1.0
        self.w_proc = 2.0
        self.w_cost = 0.1

        self.detector_true_positive_bonus = 0.5
        self.successful_mitigation_reward = {
            1: 7.0,
            2: 9.0,
            3: 11.0
        }
        self.failed_or_delayed_mitigation_cost = {
            1: 0.5,
            2: 1.0,
            3: 2.0
        }
        self.missed_attack_penalty = 6.0

        self.detector_false_positive_cost = 1.0
        self.mitigator_false_positive_cost = {
            1: 12.0,
            2: 32.0,
            3: 85.0
        }

        self.catastrophic_process_penalty = 50.0
        self.recovery_completion_bonus = 2.0

        # Harder setting
        self.attack_probability = 0.70
        self.normal_ioc_false_positive_rate = 0.03
        self.benign_cpu_spike_prob = 0.08
        self.benign_process_disturbance_prob = 0.08

        # Imperfect IoC visibility
        self.attack_ioc_true_positive_rate = {
            "overflow": 0.55,
            "stealth_drain": 0.45,
            "ransomware": 0.80
        }

        self.mitigation_delay_steps = {
            1: 1,
            2: 2,
            3: 1
        }

        self.mitigation_success_prob = {
            "overflow": {1: 0.35, 2: 0.90, 3: 1.00},
            "stealth_drain": {1: 0.88, 2: 0.62, 3: 0.95},
            "ransomware": {1: 0.10, 2: 0.50, 3: 0.95}
        }

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

        self.recovery_mode = False
        self.recovery_stable_counter = 0

        if np.random.rand() < self.attack_probability:
            self.attack_type = np.random.choice(["overflow", "stealth_drain", "ransomware"])
            self.attack_trigger_step = np.random.randint(10, 40)
        else:
            self.attack_type = "none"
            self.attack_trigger_step = self.max_steps + 1

        self.last_mit_act = 0
        self.comm_buffer = [0.0, 0.0, 0.0]

        self.pending_mitigation_action = 0
        self.pending_mitigation_timer = 0
        self.mitigation_cooldown = 0

        return self._get_obs(), {a: {} for a in self.agents}

    def _sample_normal_ioc_noise(self):
        return 1.0 if np.random.rand() < self.normal_ioc_false_positive_rate else 0.0

    def _sample_attack_ioc_signal(self):
        rate = self.attack_ioc_true_positive_rate.get(self.attack_type, 0.0)
        return 1.0 if np.random.rand() < rate else 0.0

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
        # Explicit nominal supervisory policy around the setpoint band
        if self.level < self.safe_low:
            self.pump = 1
            self.valve = 0
        elif self.level > self.safe_high:
            self.pump = 0
            self.valve = 1
        else:
            # Neutral operating regime around setpoint
            self.pump = 1
            self.valve = 1

    def _apply_recovery_policy(self):
        # Stronger recovery controller after successful mitigation
        if self.level < self.safe_low:
            self.pump = 1
            self.valve = 0
            self.recovery_stable_counter = 0
            return False

        if self.level > self.safe_high:
            self.pump = 0
            self.valve = 1
            self.recovery_stable_counter = 0
            return False

        # Inside safe band: hold a couple of steps before declaring recovery complete
        self.pump = 1
        self.valve = 1
        self.recovery_stable_counter += 1

        if self.recovery_stable_counter >= self.recovery_hold_steps:
            self.recovery_mode = False
            self.recovery_stable_counter = 0
            return True

        return False

    def _enter_recovery_mode(self):
        self.recovery_mode = True
        self.recovery_stable_counter = 0

    def _restore_post_mitigation_state(self, mit_act):
        self.raw_anomaly = 0.0
        self.ioc_match = 0.0
        self.cpu_load = 25.0 + np.random.normal(0, 1.5)
        self._enter_recovery_mode()

        if mit_act == 3:
            # Short shutdown effect, then recovery policy will take over next step
            self.pump = 0
            self.valve = 0
        else:
            self._apply_recovery_policy()

    def _apply_benign_disturbances(self):
        if np.random.rand() < self.benign_cpu_spike_prob:
            self.cpu_load = np.clip(self.cpu_load + np.random.uniform(35.0, 55.0), 25.0, 90.0)
            if np.random.rand() < 0.15:
                self.ioc_match = 1.0

        if np.random.rand() < self.benign_process_disturbance_prob:
            self.level += np.random.normal(0.0, 2.5)

    def _start_pending_mitigation(self, mit_act):
        self.pending_mitigation_action = mit_act
        self.pending_mitigation_timer = self.mitigation_delay_steps.get(mit_act, 1)
        self.mitigation_cooldown = 2

    def _resolve_pending_mitigation(self):
        if self.pending_mitigation_action == 0 or self.raw_anomaly == 0.0:
            self.pending_mitigation_action = 0
            self.pending_mitigation_timer = 0
            return False, 0

        act = self.pending_mitigation_action
        success_prob = self.mitigation_success_prob.get(self.attack_type, {}).get(act, 0.0)
        success = np.random.rand() < success_prob

        if success:
            self._restore_post_mitigation_state(act)

        self.pending_mitigation_action = 0
        self.pending_mitigation_timer = 0
        return success, act

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        net_act = actions.get("network_agent", 0)
        host_act = actions.get("host_agent", 0)
        intel_act = actions.get("threat_intel_agent", 0)
        mit_act = actions.get("mitigator_agent", 0)

        self.last_mit_act = mit_act
        self.comm_buffer = [float(net_act), float(host_act), float(intel_act)]

        if self.mitigation_cooldown > 0:
            self.mitigation_cooldown -= 1

        recovery_completed_this_step = False

        if self.raw_anomaly == 0.0:
            if self.recovery_mode:
                recovery_completed_this_step = self._apply_recovery_policy()
                self.cpu_load = 25.0 + np.random.normal(0, 1.0)
                self.ioc_match = 0.0
            else:
                self._apply_normal_control_policy()
                self.cpu_load = 25.0 + np.random.normal(0, 3.0)
                self.ioc_match = self._sample_normal_ioc_noise()
                self._apply_benign_disturbances()

        if self.attack_type != "none" and self.steps == self.attack_trigger_step:
            self.raw_anomaly = 1.0
            self.recovery_mode = False
            self.recovery_stable_counter = 0

        if self.raw_anomaly == 1.0:
            if self.attack_type == "overflow":
                self.pump = 1
                self.valve = 0
                self.cpu_load = np.clip(82.0 + np.random.normal(0, 8.0), 60.0, 99.0)
                self.ioc_match = self._sample_attack_ioc_signal()

            elif self.attack_type == "stealth_drain":
                self.pump = 0
                self.valve = 1
                self.cpu_load = np.clip(31.0 + np.random.normal(0, 4.0), 18.0, 45.0)
                self.ioc_match = self._sample_attack_ioc_signal()
                if self.level < 40.0 and np.random.rand() < 0.25:
                    self.ioc_match = 1.0

            elif self.attack_type == "ransomware":
                self.pump = 0
                self.valve = 0
                self.cpu_load = np.clip(93.0 + np.random.normal(0, 5.0), 72.0, 99.0)
                self.ioc_match = self._sample_attack_ioc_signal()

        if self.pump == 1:
            self.level += 3.0
        if self.valve == 1:
            self.level -= 3.0
        self.level += np.random.normal(0, 0.8)
        self.level = np.clip(self.level, 0.0, 100.0)

        self.level_history.append(self.level)

        r_security = 0.0
        r_cost = 0.0

        deviation = (self.level - self.setpoint) / 25.0
        r_process = -(deviation ** 2)

        successful_mitigation = False
        successful_mitigation_action = 0

        if self.pending_mitigation_action > 0:
            self.pending_mitigation_timer -= 1
            if self.pending_mitigation_timer <= 0:
                successful_mitigation, successful_mitigation_action = self._resolve_pending_mitigation()

        if self.raw_anomaly == 1.0:
            if net_act == 1:
                r_security += self.detector_true_positive_bonus
            if host_act == 1:
                r_security += self.detector_true_positive_bonus
            if intel_act == 1:
                r_security += self.detector_true_positive_bonus

            if successful_mitigation:
                r_security += self.successful_mitigation_reward.get(successful_mitigation_action, 0.0)

            if mit_act > 0 and self.pending_mitigation_action == 0 and self.mitigation_cooldown == 0:
                self._start_pending_mitigation(mit_act)
                r_cost -= self.failed_or_delayed_mitigation_cost.get(mit_act, 0.0)
            elif mit_act > 0 and not successful_mitigation:
                r_cost -= self.failed_or_delayed_mitigation_cost.get(mit_act, 0.0)

            if self.raw_anomaly == 1.0:
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

            if recovery_completed_this_step:
                r_process += self.recovery_completion_bonus

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

        infos = {
            a: {
                "successful_mitigation": successful_mitigation,
                "successful_mitigation_action": successful_mitigation_action,
                "recovery_mode": self.recovery_mode,
                "recovery_completed_this_step": recovery_completed_this_step
            }
            for a in self.possible_agents
        }

        return (
            self._get_obs(),
            rewards,
            terminations,
            {a: False for a in self.possible_agents},
            infos
        )