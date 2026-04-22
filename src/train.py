import numpy as np
from utils.llm_client import LLMClient
from envs.cps_env import L2MAID_MARL_Env
from agents.mappo_agent import MAPPO_CTDE


def train_agents(episodes=1000, debug_mode=False):
    # Initialize the LLM Client
    llm = LLMClient(debug_mode=debug_mode)

    # Load cache to bypass redundant API calls
    llm.load_cache()

    # Initialize the Environment
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

    # Initialize the Multi-Agent PPO system
    mappo = MAPPO_CTDE(obs_dims, act_dims)

    joint_rewards = []
    psi_history = []

    print(f"\n🚀 STARTING 4-AGENT MAPPO (CTDE) TRAINING ({episodes} Episodes)\n")

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        total_episode_reward = 0.0
        deviations = []

        while env.agents:
            global_obs = np.concatenate(
                [obs_dict.get(agent, np.zeros(obs_dims[agent], dtype=np.float32)) for agent in obs_dims.keys()]
            )

            actions, log_probs = mappo.get_actions(obs_dict)
            next_obs_dict, rewards, terminations, _, _ = env.step(actions)

            done = any(terminations.values())

            # Store experiences for centralized training
            mappo.store(global_obs, obs_dict, actions, log_probs, rewards, done)

            total_episode_reward += rewards.get("mitigator_agent", 0.0)

            # Track physical deviations for a PSI-like trend measure during training
            if "host_agent" in obs_dict:
                level_pct = obs_dict["host_agent"][0]
                deviations.append((level_pct * 100.0 - 50.0) ** 2)

            obs_dict = next_obs_dict

        # Update neural networks at the end of the episode
        mappo.update()

        joint_rewards.append(total_episode_reward)
        psi = 1.0 / (np.sqrt(np.mean(deviations)) + 1e-5) if deviations else 0.0
        psi_history.append(psi)

        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(joint_rewards[-20:])
            avg_psi = np.mean(psi_history[-20:])
            print(
                f"[TRAIN] Ep {ep + 1}/{episodes} | "
                f"Avg Joint Reward: {avg_reward:.2f} | "
                f"Avg PSI-like Score: {avg_psi:.4f} | "
                f"Cache Size: {len(llm.cache)}"
            )

    # Save the trained RL models
    mappo.save()

    # Save cache to disk so future runs are faster
    llm.save_cache()

    print("💾 Training complete. 4-Agent MAPPO Models saved.")


if __name__ == "__main__":
    train_agents(episodes=1000, debug_mode=False)