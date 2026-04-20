import numpy as np
from src.utils.llm_client import LLMClient
from src.envs.cps_env import L2MAID_MARL_Env
from src.agents.mappo_agent import MAPPO_CTDE


def train_agents(episodes=1000, debug_mode=True):
    llm = LLMClient(debug_mode=debug_mode)
    env = L2MAID_MARL_Env(llm_client=llm, debug_mode=debug_mode)

    obs_dims = {
        "network_agent": 8,
        "host_agent": 8,
        "threat_intel_agent": 7,
        "mitigator_agent": 11
    }
    act_dims = {
        "network_agent": 2,
        "host_agent": 2,
        "threat_intel_agent": 2,
        "mitigator_agent": 4
    }

    mappo = MAPPO_CTDE(obs_dims, act_dims)

    joint_rewards, psi_history = [], []
    print(f"\n🚀 STARTING 4-AGENT MAPPO (CTDE) TRAINING ({episodes} Episodes)\n")

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        tot_reward = 0
        deviations = []

        while env.agents:
            global_obs = np.concatenate([obs_dict.get(a, np.zeros(obs_dims[a])) for a in obs_dims.keys()])

            actions, log_probs = mappo.get_actions(obs_dict)
            next_obs_dict, rewards, terminations, _, _ = env.step(actions)

            done = any(terminations.values())
            mappo.store(global_obs, obs_dict, actions, log_probs, rewards, done)

            tot_reward += rewards.get("mitigator_agent", 0)

            if "host_agent" in obs_dict:
                level_pct = obs_dict["host_agent"][0]
                deviations.append((level_pct * 100 - 50) ** 2)

            obs_dict = next_obs_dict

        mappo.update()

        joint_rewards.append(tot_reward)
        psi = 1.0 / (np.sqrt(np.mean(deviations)) + 1e-5) if deviations else 0
        psi_history.append(psi)

        if (ep + 1) % 20 == 0:
            avg_j = np.mean(joint_rewards[-20:])
            print(f"[TRAIN] Ep {ep + 1}/{episodes} | Avg Joint Reward: {avg_j:.1f} | Cache: {len(llm.cache)}")

    mappo.save()
    print("💾 Training complete. 4-Agent MAPPO Models saved.")


if __name__ == "__main__":
    train_agents(episodes=1000, debug_mode=True)