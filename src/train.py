import numpy as np
from src.utils.llm_client import LLMClient
from src.envs.cps_env import L2MAID_MARL_Env
from src.agents.mappo_agent import MAPPO_CTDE


def train_agents(episodes=1000, debug_mode=False):
    # Initialize the LLM Client
    llm = LLMClient(debug_mode=debug_mode)

    # ⚡ CRITICAL FIX: Load the cache to bypass redundant API calls
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

    joint_rewards, psi_history = [], []
    print(f"\n🚀 STARTING 4-AGENT MAPPO (CTDE) TRAINING ({episodes} Episodes)\n")

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        tot_reward = 0
        deviations = []

        # Episode Loop
        while env.agents:
            global_obs = np.concatenate([obs_dict.get(a, np.zeros(obs_dims[a])) for a in obs_dims.keys()])

            actions, log_probs = mappo.get_actions(obs_dict)
            next_obs_dict, rewards, terminations, _, _ = env.step(actions)

            done = any(terminations.values())

            # Store experiences for the centralized critic to learn from
            mappo.store(global_obs, obs_dict, actions, log_probs, rewards, done)

            tot_reward += rewards.get("mitigator_agent", 0)

            # Track physical deviations to calculate the Process Stability Index (PSI)
            if "host_agent" in obs_dict:
                level_pct = obs_dict["host_agent"][0]
                deviations.append((level_pct * 100 - 50) ** 2)

            obs_dict = next_obs_dict

        # Update neural networks at the end of the episode
        mappo.update()

        joint_rewards.append(tot_reward)
        psi = 1.0 / (np.sqrt(np.mean(deviations)) + 1e-5) if deviations else 0
        psi_history.append(psi)

        # Print progress every 20 episodes
        if (ep + 1) % 20 == 0:
            avg_j = np.mean(joint_rewards[-20:])
            print(f"[TRAIN] Ep {ep + 1}/{episodes} | Avg Joint Reward: {avg_j:.1f} | Cache Size: {len(llm.cache)}")

    # Save the trained RL models
    mappo.save()

    # ⚡ CRITICAL FIX: Save the cache to disk so future training runs are instant
    llm.save_cache()

    print("💾 Training complete. 4-Agent MAPPO Models saved.")


if __name__ == "__main__":
    # Ensure debug_mode is False here as well to prevent terminal spam
    train_agents(episodes=1000, debug_mode=True)