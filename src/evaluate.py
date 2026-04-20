import torch
import numpy as np
import os
from src.utils.llm_client import LLMClient
from src.envs.cps_env import L2MAID_MARL_Env
from src.agents.mappo_agent import MAPPO_CTDE


def load_mappo_models(mappo, directory="models", prefix="l2maid"):
    """Helper to load the saved PyTorch weights into the MAPPO actors."""
    if not os.path.exists(directory):
        print(f"❌ Error: Directory '{directory}' not found. Train models first.")
        return False

    try:
        # We don't need the Critic for evaluation (Decentralized Execution!)
        for agent in mappo.agents:
            model_path = f"{directory}/{prefix}_actor_{agent}.pth"
            mappo.actors[agent].load_state_dict(torch.load(model_path))
            mappo.actors[agent].eval()  # Set to evaluation mode
        print(f"✅ Successfully loaded decentralized Actor networks from '{directory}'")
        return True
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False


def evaluate(episodes=100, debug_mode=False):
    print(f"\n{'=' * 50}\n🔬 RUNNING 4-AGENT EVALUATION\n{'=' * 50}")

    # 1. Initialize Environment
    llm = LLMClient(debug_mode=debug_mode)
    env = L2MAID_MARL_Env(llm_client=llm, debug_mode=debug_mode)

    # 2. Rebuild the Architecture
    obs_dims = {"network_agent": 8, "host_agent": 8, "threat_intel_agent": 7, "mitigator_agent": 11}
    act_dims = {"network_agent": 2, "host_agent": 2, "threat_intel_agent": 2, "mitigator_agent": 4}
    mappo = MAPPO_CTDE(obs_dims, act_dims)

    # 3. Load Trained Weights
    if not load_mappo_models(mappo):
        return

    # 4. Tracking Metrics
    history = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "psi_scores": [], "response_times": []}

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        deviations = []
        attack_detected = False
        attack_start_step = env.attack_trigger_step

        while env.agents:
            # Deterministic Action Selection (No exploration)
            actions = {}
            with torch.no_grad():
                for agent in env.agents:
                    obs_tensor = torch.FloatTensor(obs_dict[agent]).unsqueeze(0)
                    logits = mappo.actors[agent].net(obs_tensor)
                    actions[agent] = torch.argmax(logits).item()  # Greedy selection

            # Environment Step
            next_obs_dict, rewards, terminations, _, _ = env.step(actions)

            is_anomaly = env.raw_anomaly == 1.0
            mit_act = actions.get("mitigator_agent", 0)

            # Metric Tracking: Did the Mitigator execute a defense?
            if is_anomaly:
                if mit_act > 0 and not attack_detected:
                    history["tp"] += 1
                    attack_detected = True
                    # Calculate Mean Time To Respond (MTTR)
                    history["response_times"].append(env.steps - attack_start_step)
            elif not is_anomaly and mit_act > 0:
                history["fp"] += 1  # False Alarm Shutdown!

            # Process Stability Tracking
            if "host_agent" in obs_dict:
                level_pct = obs_dict["host_agent"][0]
                deviations.append((level_pct * 100 - 50) ** 2)

            obs_dict = next_obs_dict

            if any(terminations.values()):
                # If episode ended and attack was missed entirely
                if env.steps >= attack_start_step and not attack_detected:
                    history["fn"] += 1
                elif not is_anomaly and mit_act == 0:
                    history["tn"] += 1  # Successfully did nothing during normal ops

        # Calculate Episode PSI
        psi = 1.0 / (np.sqrt(np.mean(deviations)) + 1e-5) if deviations else 0
        history["psi_scores"].append(psi)

        if (ep + 1) % 10 == 0:
            print(f"[EVAL] Completed episode {ep + 1}/{episodes}")

    # 5. Final Calculations
    tp, fp = history["tp"], history["fp"]
    fn, tn = history["fn"], history["tn"]

    dr = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
    mttr = np.mean(history["response_times"]) if history["response_times"] else 0

    print("\n" + "=" * 50)
    print("🏆 FINAL EVALUATION METRICS 🏆")
    print("=" * 50)
    print(f"Total Episodes:                {episodes}")
    print(f"Detection Rate (DR):           {dr:.2f}%")
    print(f"False Positive Rate (FPR):     {fpr:.2f}%")
    print(f"Mean Time To Respond (MTTR):   {mttr:.1f} steps")
    print(f"Average Process Stability:     {np.mean(history['psi_scores']):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate(episodes=100)