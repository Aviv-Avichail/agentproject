import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.utils.llm_client import LLMClient
from src.utils.logger import L2MAID_Logger
from src.envs.cps_env import L2MAID_MARL_Env
from src.agents.mappo_agent import MAPPO_CTDE


def load_mappo_models(mappo, directory="models", prefix="l2maid"):
    if not os.path.exists(directory):
        print(f"❌ Error: Directory '{directory}' not found. Train models first.")
        return False

    try:
        for agent in mappo.agents:
            model_path = f"{directory}/{prefix}_actor_{agent}.pth"
            mappo.actors[agent].load_state_dict(torch.load(model_path, map_location="cpu"))
            mappo.actors[agent].eval()
        print(f"✅ Successfully loaded decentralized Actor networks from '{directory}'")
        return True
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False


def plot_sample_episode(trajectory, filename="sample_episode_trajectory.png"):
    """Generates a graph of the physical plant and the agent's response."""
    steps = range(len(trajectory["levels"]))

    plt.figure(figsize=(12, 6))

    # Plot 1: Physical Water Level
    plt.subplot(2, 1, 1)
    plt.plot(steps, trajectory["levels"], label="Water Level", color="blue", linewidth=2)
    plt.axhline(y=50.0, color="green", linestyle="--", label="Optimal Setpoint (50)")
    plt.axhline(y=100.0, color="red", linestyle=":", label="Catastrophic Overflow")
    plt.axhline(y=0.0, color="red", linestyle=":", label="Catastrophic Drain")

    if trajectory["attack_step"] > 0:
        plt.axvline(
            x=trajectory["attack_step"],
            color="orange",
            linestyle="-",
            label=f"Attack Injected ({trajectory['attack_type']})"
        )
    if trajectory["mitigation_step"] > 0:
        plt.axvline(
            x=trajectory["mitigation_step"],
            color="purple",
            linestyle="-",
            linewidth=2,
            label="Mitigator Action"
        )

    plt.title("Cyber-Physical Trajectory (Sample Episode)")
    plt.ylabel("Water Level")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    # Plot 2: CPU Load
    plt.subplot(2, 1, 2)
    plt.plot(steps, trajectory["cpus"], label="CPU Load (%)", color="red", alpha=0.7)
    if trajectory["attack_step"] > 0:
        plt.axvline(x=trajectory["attack_step"], color="orange", linestyle="-")

    plt.ylabel("CPU %")
    plt.xlabel("Environment Timestep")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n📊 Generated trajectory graph: {filename}")
    plt.close()


def evaluate(episodes=100, debug_mode=True, use_rule_based_mitigator=True):
    print(f"\n{'=' * 50}\n🔬 RUNNING GRANULAR 4-AGENT EVALUATION\n{'=' * 50}")

    llm = LLMClient(debug_mode=debug_mode)
    logger = L2MAID_Logger()
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

    if not load_mappo_models(mappo):
        return

    if use_rule_based_mitigator:
        print("🧪 Diagnostic mode ON: overriding mitigator with IOC-based sanity rule.")
        print("   Rule: if mitigator obs[4] (ioc_match) > 0.5 -> action 1, else action 0.\n")
    else:
        print("🤖 Using learned mitigator policy only.\n")

    metrics = {
        "overflow": {"tp": 0, "fn": 0, "response_times": []},
        "stealth_drain": {"tp": 0, "fn": 0, "response_times": []},
        "ransomware": {"tp": 0, "fn": 0, "response_times": []}
    }
    global_fp = 0
    global_tn = 0
    psi_scores = []

    mitigator_action_hist = {0: 0, 1: 0, 2: 0, 3: 0}

    sample_trajectory = {
        "levels": [],
        "cpus": [],
        "attack_step": -1,
        "mitigation_step": -1,
        "attack_type": ""
    }

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        deviations = []
        attack_detected = False
        attack_start_step = env.attack_trigger_step
        current_attack_type = env.attack_type

        is_sample_ep = (ep == 0)
        if is_sample_ep:
            sample_trajectory["attack_type"] = current_attack_type
            sample_trajectory["attack_step"] = attack_start_step

        while env.agents:
            if is_sample_ep:
                sample_trajectory["levels"].append(env.level)
                sample_trajectory["cpus"].append(env.cpu_load)

            actions = {}

            with torch.no_grad():
                for agent in env.agents:
                    obs_tensor = torch.FloatTensor(obs_dict[agent]).unsqueeze(0)
                    logits = mappo.actors[agent].net(obs_tensor)
                    actions[agent] = torch.argmax(logits, dim=-1).item()

            # Temporary diagnostic override for the mitigator:
            # mitigator obs = [level, pump, valve, cpu, ioc_match] + comm_buffer + llm_vector
            if use_rule_based_mitigator and "mitigator_agent" in obs_dict:
                ioc_match = obs_dict["mitigator_agent"][4]
                actions["mitigator_agent"] = 1 if ioc_match > 0.5 else 0

            mit_act = actions.get("mitigator_agent", 0)
            mitigator_action_hist[mit_act] = mitigator_action_hist.get(mit_act, 0) + 1

            # State before stepping
            is_anomaly = env.raw_anomaly == 1.0
            current_level = env.level
            current_cpu = env.cpu_load

            llm_vec = obs_dict["mitigator_agent"][-3:]
            llm_choice = "Attack" if llm_vec[0] > 0.5 else "Normal"

            next_obs_dict, rewards, terminations, _, _ = env.step(actions)

            logger.log_step(
                ep,
                env.steps,
                current_attack_type,
                current_level,
                current_cpu,
                llm_choice,
                mit_act,
                is_anomaly
            )

            if is_anomaly:
                if mit_act > 0 and not attack_detected:
                    metrics[current_attack_type]["tp"] += 1
                    attack_detected = True
                    metrics[current_attack_type]["response_times"].append(env.steps - attack_start_step)
                    if is_sample_ep:
                        sample_trajectory["mitigation_step"] = env.steps
            elif not is_anomaly and mit_act > 0:
                global_fp += 1

            if "host_agent" in obs_dict:
                level_pct = obs_dict["host_agent"][0]
                deviations.append((level_pct * 100 - 50) ** 2)

            obs_dict = next_obs_dict

            if any(terminations.values()):
                if env.steps >= attack_start_step and not attack_detected:
                    metrics[current_attack_type]["fn"] += 1
                elif not is_anomaly and mit_act == 0:
                    global_tn += 1

        psi = 1.0 / (np.sqrt(np.mean(deviations)) + 1e-5) if deviations else 0
        psi_scores.append(psi)

        if (ep + 1) % 10 == 0:
            print(f"[EVAL] Completed episode {ep + 1}/{episodes}")

    plot_sample_episode(sample_trajectory)

    total_tp = sum(m["tp"] for m in metrics.values())
    total_fn = sum(m["fn"] for m in metrics.values())
    all_responses = sum((m["response_times"] for m in metrics.values()), [])

    overall_dr = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0
    overall_fpr = (global_fp / (global_fp + global_tn)) * 100 if (global_fp + global_tn) > 0 else 0
    overall_mttr = np.mean(all_responses) if all_responses else 0

    print("\n" + "=" * 50)
    print("🏆 FINAL EVALUATION METRICS 🏆")
    print("=" * 50)
    print(f"Total Episodes:                {episodes}")
    print(f"Overall Detection Rate (DR):   {overall_dr:.2f}%")
    print(f"Overall False Positive (FPR):  {overall_fpr:.2f}%")
    print(f"Average Process Stability:     {np.mean(psi_scores):.4f}")
    print(f"Average Response Time:         {overall_mttr:.2f} steps")
    print("\nMitigator Action Histogram:")
    for action_id in sorted(mitigator_action_hist.keys()):
        print(f"  Action {action_id}: {mitigator_action_hist[action_id]}")

    print("\n--- BREAKDOWN BY ATTACK VECTOR ---")
    for atk, data in metrics.items():
        atk_tp, atk_fn = data["tp"], data["fn"]
        atk_dr = (atk_tp / (atk_tp + atk_fn)) * 100 if (atk_tp + atk_fn) > 0 else 0
        atk_mttr = np.mean(data["response_times"]) if data["response_times"] else 0
        print(f"🔹 {atk.upper()}:")
        print(f"   Detection Rate: {atk_dr:.1f}%")
        print(f"   Mean Response:  {atk_mttr:.1f} steps")

    print("=" * 50)
    print("📂 A detailed breakdown of every step has been saved to: logs/defense_performance.csv")


if __name__ == "__main__":
    evaluate(episodes=100, debug_mode=True, use_rule_based_mitigator=True)