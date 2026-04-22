import csv
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from src.utils.llm_client import LLMClient
    from src.envs.cps_env import L2MAID_MARL_Env
    from src.agents.mappo_agent import MAPPO_CTDE
except ImportError:
    from utils.llm_client import LLMClient
    from envs.cps_env import L2MAID_MARL_Env
    from agents.mappo_agent import MAPPO_CTDE


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def apply_env_overrides(env, overrides):
    for key, value in overrides.items():
        current = getattr(env, key, None)
        if isinstance(current, dict) and isinstance(value, dict):
            merged = deepcopy(current)
            for sub_k, sub_v in value.items():
                if isinstance(merged.get(sub_k), dict) and isinstance(sub_v, dict):
                    merged[sub_k] = {**merged[sub_k], **sub_v}
                else:
                    merged[sub_k] = sub_v
            setattr(env, key, merged)
        else:
            setattr(env, key, value)


def ensure_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario",
            "Seed",
            "Episode",
            "Step",
            "Attack_Type",
            "Water_Level",
            "CPU_Load",
            "LLM_Decision",
            "Agent_Action",
            "Is_Anomaly"
        ])


def append_csv_row(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def plot_sample_episode(trajectory, filename):
    steps = range(len(trajectory["levels"]))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(steps, trajectory["levels"], label="Water Level", linewidth=2)
    plt.axhline(y=50.0, linestyle="--", label="Optimal Setpoint (50)")
    plt.axhline(y=100.0, linestyle=":", label="Catastrophic Overflow")
    plt.axhline(y=0.0, linestyle=":", label="Catastrophic Drain")

    if trajectory["attack_step"] >= 0:
        plt.axvline(
            x=trajectory["attack_step"],
            linestyle="-",
            label=f"Attack Injected ({trajectory['attack_type']})"
        )
    if trajectory["mitigation_step"] >= 0:
        plt.axvline(
            x=trajectory["mitigation_step"],
            linestyle="-",
            linewidth=2,
            label="Successful Mitigation"
        )

    plt.title("Cyber-Physical Trajectory (Sample Episode)")
    plt.ylabel("Water Level")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(steps, trajectory["cpus"], label="CPU Load (%)", alpha=0.7)
    if trajectory["attack_step"] >= 0:
        plt.axvline(x=trajectory["attack_step"], linestyle="-")
    if trajectory["mitigation_step"] >= 0:
        plt.axvline(x=trajectory["mitigation_step"], linestyle="-", linewidth=2)

    plt.ylabel("CPU %")
    plt.xlabel("Environment Timestep")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"📊 Generated trajectory graph: {filename}")
    plt.close()


def get_default_dims():
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
    return obs_dims, act_dims


def init_metric_store():
    return {
        "overflow": {"tp": 0, "fn": 0, "response_times": []},
        "stealth_drain": {"tp": 0, "fn": 0, "response_times": []},
        "ransomware": {"tp": 0, "fn": 0, "response_times": []}
    }


def select_greedy_actions(mappo, obs_dict):
    actions = {}
    with torch.no_grad():
        for agent in obs_dict.keys():
            obs_tensor = torch.as_tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
            logits = mappo.actors[agent].net(obs_tensor)
            actions[agent] = int(torch.argmax(logits, dim=-1).item())
    return actions


def summarize_single_run(metrics, global_fp, global_tn, psi_scores, mitigator_action_hist):
    total_tp = sum(m["tp"] for m in metrics.values())
    total_fn = sum(m["fn"] for m in metrics.values())
    all_responses = sum((m["response_times"] for m in metrics.values()), [])

    overall_dr = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0.0
    overall_fpr = (global_fp / (global_fp + global_tn)) * 100 if (global_fp + global_tn) > 0 else 0.0
    overall_mttr = float(np.mean(all_responses)) if all_responses else 0.0
    overall_psi = float(np.mean(psi_scores)) if psi_scores else 0.0

    per_attack = {}
    for atk, data in metrics.items():
        atk_tp = data["tp"]
        atk_fn = data["fn"]
        atk_dr = (atk_tp / (atk_tp + atk_fn)) * 100 if (atk_tp + atk_fn) > 0 else 0.0
        atk_mttr = float(np.mean(data["response_times"])) if data["response_times"] else 0.0
        per_attack[atk] = {
            "dr": atk_dr,
            "mttr": atk_mttr,
            "tp": atk_tp,
            "fn": atk_fn
        }

    return {
        "overall_dr": overall_dr,
        "overall_fpr": overall_fpr,
        "overall_mttr": overall_mttr,
        "overall_psi": overall_psi,
        "per_attack": per_attack,
        "action_hist": mitigator_action_hist
    }


def run_single_seed(
    scenario_name,
    env_overrides,
    seed,
    episodes=100,
    debug_mode=False,
    save_plot=False,
    csv_log_path=None
):
    set_global_seed(seed)

    llm = LLMClient(debug_mode=debug_mode)
    llm.load_cache()

    env = L2MAID_MARL_Env(llm_client=llm, debug_mode=debug_mode)
    apply_env_overrides(env, env_overrides)

    obs_dims, act_dims = get_default_dims()
    mappo = MAPPO_CTDE(obs_dims, act_dims)

    if not load_mappo_models(mappo):
        raise RuntimeError("Could not load trained models.")

    metrics = init_metric_store()
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

    if csv_log_path:
        ensure_csv(csv_log_path)

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        deviations = []
        attack_detected = False
        current_attack_type = env.attack_type
        attack_start_step = env.attack_trigger_step
        is_attack_episode = current_attack_type != "none"

        is_sample_ep = (ep == 0)
        if is_sample_ep:
            sample_trajectory["attack_type"] = current_attack_type
            sample_trajectory["attack_step"] = attack_start_step if is_attack_episode else -1

        while env.agents:
            if is_sample_ep:
                sample_trajectory["levels"].append(env.level)
                sample_trajectory["cpus"].append(env.cpu_load)

            actions = select_greedy_actions(mappo, obs_dict)
            mit_act = actions.get("mitigator_agent", 0)
            mitigator_action_hist[mit_act] += 1

            was_anomaly = env.raw_anomaly == 1.0
            current_level = env.level
            current_cpu = env.cpu_load
            llm_vec = obs_dict["mitigator_agent"][-3:]
            llm_choice = f"[{llm_vec[0]:.2f}, {llm_vec[1]:.2f}, {llm_vec[2]:.2f}]"

            next_obs_dict, rewards, terminations, _, infos = env.step(actions)
            mitigator_info = infos.get("mitigator_agent", {})
            successful_mitigation = bool(mitigator_info.get("successful_mitigation", False))

            if csv_log_path:
                append_csv_row(csv_log_path, [
                    scenario_name,
                    seed,
                    ep,
                    env.steps,
                    current_attack_type,
                    round(float(current_level), 2),
                    round(float(current_cpu), 2),
                    llm_choice,
                    mit_act,
                    was_anomaly
                ])

            if is_attack_episode and was_anomaly:
                if successful_mitigation and not attack_detected:
                    metrics[current_attack_type]["tp"] += 1
                    attack_detected = True
                    metrics[current_attack_type]["response_times"].append(env.steps - attack_start_step)
                    if is_sample_ep and sample_trajectory["mitigation_step"] < 0:
                        sample_trajectory["mitigation_step"] = env.steps
            else:
                if mit_act > 0:
                    global_fp += 1
                else:
                    global_tn += 1

            if "host_agent" in obs_dict:
                level_pct = obs_dict["host_agent"][0]
                deviations.append((level_pct * 100.0 - 50.0) ** 2)

            obs_dict = next_obs_dict

            if any(terminations.values()):
                if is_attack_episode and env.steps > attack_start_step and not attack_detected:
                    metrics[current_attack_type]["fn"] += 1

        psi = 1.0 / (np.sqrt(np.mean(deviations)) + 1e-5) if deviations else 0.0
        psi_scores.append(psi)

    if save_plot:
        plot_name = f"sample_episode_{scenario_name}_seed_{seed}.png"
        plot_sample_episode(sample_trajectory, plot_name)

    return summarize_single_run(metrics, global_fp, global_tn, psi_scores, mitigator_action_hist)


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_runs(run_results):
    aggregate = {
        "overall_dr": [],
        "overall_fpr": [],
        "overall_mttr": [],
        "overall_psi": [],
        "action_hist": {0: [], 1: [], 2: [], 3: []},
        "per_attack": {
            "overflow": {"dr": [], "mttr": []},
            "stealth_drain": {"dr": [], "mttr": []},
            "ransomware": {"dr": [], "mttr": []}
        }
    }

    for result in run_results:
        aggregate["overall_dr"].append(result["overall_dr"])
        aggregate["overall_fpr"].append(result["overall_fpr"])
        aggregate["overall_mttr"].append(result["overall_mttr"])
        aggregate["overall_psi"].append(result["overall_psi"])

        for action_id in [0, 1, 2, 3]:
            aggregate["action_hist"][action_id].append(result["action_hist"][action_id])

        for atk in aggregate["per_attack"].keys():
            aggregate["per_attack"][atk]["dr"].append(result["per_attack"][atk]["dr"])
            aggregate["per_attack"][atk]["mttr"].append(result["per_attack"][atk]["mttr"])

    return aggregate


def print_aggregate_report(title, aggregate):
    dr_mean, dr_std = mean_std(aggregate["overall_dr"])
    fpr_mean, fpr_std = mean_std(aggregate["overall_fpr"])
    mttr_mean, mttr_std = mean_std(aggregate["overall_mttr"])
    psi_mean, psi_std = mean_std(aggregate["overall_psi"])

    print("\n" + "=" * 60)
    print(f"🏆 {title}")
    print("=" * 60)
    print(f"Overall DR:    {dr_mean:.2f}% ± {dr_std:.2f}")
    print(f"Overall FPR:   {fpr_mean:.2f}% ± {fpr_std:.2f}")
    print(f"Overall MTTR:  {mttr_mean:.2f} ± {mttr_std:.2f} steps")
    print(f"Overall PSI:   {psi_mean:.4f} ± {psi_std:.4f}")

    print("\nAverage Mitigator Action Histogram:")
    for action_id in [0, 1, 2, 3]:
        hist_mean, hist_std = mean_std(aggregate["action_hist"][action_id])
        print(f"  Action {action_id}: {hist_mean:.1f} ± {hist_std:.1f}")

    print("\nPer-Attack Breakdown:")
    for atk in ["overflow", "stealth_drain", "ransomware"]:
        atk_dr_mean, atk_dr_std = mean_std(aggregate["per_attack"][atk]["dr"])
        atk_mttr_mean, atk_mttr_std = mean_std(aggregate["per_attack"][atk]["mttr"])
        print(f"🔹 {atk.upper()}:")
        print(f"   DR:   {atk_dr_mean:.2f}% ± {atk_dr_std:.2f}")
        print(f"   MTTR: {atk_mttr_mean:.2f} ± {atk_mttr_std:.2f} steps")


def evaluate_suite(episodes_per_seed=100, debug_mode=False):
    print(f"\n{'=' * 60}\n🔬 RUNNING BASE + STRESS EVALUATION SUITE\n{'=' * 60}")

    seeds = [11, 22, 33, 44, 55]

    base_overrides = {}

    stress_overrides = {
    "attack_probability": 0.45,
    "normal_ioc_false_positive_rate": 0.10,
    "benign_cpu_spike_prob": 0.22,
    "benign_process_disturbance_prob": 0.20,
    "recovery_hold_steps": 3,
    "mitigation_delay_steps": {
        1: 2,
        2: 3,
        3: 2
    },
    "attack_ioc_true_positive_rate": {
        "overflow": 0.30,
        "stealth_drain": 0.15,
        "ransomware": 0.45
    },
    "mitigation_success_prob": {
        "overflow": {1: 0.25, 2: 0.65, 3: 0.88},
        "stealth_drain": {1: 0.60, 2: 0.28, 3: 0.72},
        "ransomware": {1: 0.02, 2: 0.18, 3: 0.58}
    }
}

    # Convert inner dict keys back to ints in case of accidental string handling
    stress_overrides["mitigation_success_prob"] = {
        atk: {int(k): float(v) for k, v in actions.items()}
        for atk, actions in stress_overrides["mitigation_success_prob"].items()
    }

    base_runs = []
    stress_runs = []

    for idx, seed in enumerate(seeds):
        print(f"\n[BASE] Running seed {seed} ({idx + 1}/{len(seeds)})...")
        base_csv = f"logs/eval_base_seed_{seed}.csv"
        base_runs.append(
            run_single_seed(
                scenario_name="base",
                env_overrides=base_overrides,
                seed=seed,
                episodes=episodes_per_seed,
                debug_mode=debug_mode,
                save_plot=(idx == 0),
                csv_log_path=base_csv
            )
        )

    for idx, seed in enumerate(seeds):
        print(f"\n[STRESS] Running seed {seed} ({idx + 1}/{len(seeds)})...")
        stress_csv = f"logs/eval_stress_seed_{seed}.csv"
        stress_runs.append(
            run_single_seed(
                scenario_name="stress",
                env_overrides=stress_overrides,
                seed=seed,
                episodes=episodes_per_seed,
                debug_mode=debug_mode,
                save_plot=(idx == 0),
                csv_log_path=stress_csv
            )
        )

    base_agg = aggregate_runs(base_runs)
    stress_agg = aggregate_runs(stress_runs)

    print_aggregate_report("BASE EVALUATION (mean ± std over 5 seeds)", base_agg)
    print_aggregate_report("STRESS EVALUATION (mean ± std over 5 seeds)", stress_agg)

    print("\nSaved per-seed CSV logs under logs/eval_base_seed_*.csv and logs/eval_stress_seed_*.csv")


if __name__ == "__main__":
    evaluate_suite(episodes_per_seed=100, debug_mode=False)