# L2M-AID-Style Autonomous Cyber-Physical Defense with Neuro-Symbolic MARL

This repository contains a **simplified implementation and experimental extension** inspired by **L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning**.

The project models a small industrial control / cyber-physical system (CPS) in which a team of cooperative agents must detect, mitigate, and recover from attacks on a simulated water-level process. The implementation combines:

- **Multi-Agent Reinforcement Learning (MARL)** with a MAPPO-style setup
- **Centralized Training with Decentralized Execution (CTDE)**
- A lightweight **LLM semantic channel** injected into agent observations
- A **plant-side recovery controller** so that successful defense means not only *containment* but also *return to safe operation*

---

## Project Goal

The original L2M-AID paper targets autonomous cyber-physical defense in IIoT / ICS settings. This repository does **not** attempt a full reproduction of the paper’s SWaT benchmark pipeline. Instead, it implements a **course-scale simulator** that preserves the core idea:

> use a semantic LLM-style signal to enrich local observations for a multi-agent defender, then learn cooperative defense policies that balance security and physical process stability.

The main lesson from this project is that in cyber-physical defense, **clearing the anomaly is not enough**. A system that stops the attack but leaves the physical process drifting away from the setpoint is still incomplete. The final version of this project explicitly addresses that problem.

---

## Final Highlights

### Final base evaluation (5 seeds)
- **Detection Rate (DR):** 99.70 ± 0.60%
- **False Positive Rate (FPR):** 0.04 ± 0.03%
- **Mean Time To Respond (MTTR):** 5.06 ± 0.15 steps
- **Process Stability Index (PSI):** 0.5047 ± 0.0230

### Final held-out stress evaluation (5 seeds)
- **Detection Rate (DR):** 95.46 ± 3.77%
- **False Positive Rate (FPR):** 0.25 ± 0.05%
- **Mean Time To Respond (MTTR):** 7.24 ± 0.27 steps
- **Process Stability Index (PSI):** 0.4829 ± 0.0115

### Stress per-attack breakdown
- **Overflow:** 97.16 ± 3.57% DR, 7.01 ± 0.56 steps MTTR
- **Stealth Drain:** 88.02 ± 8.48% DR, 9.81 ± 1.13 steps MTTR
- **Ransomware:** 100.00 ± 0.00% DR, 5.76 ± 0.49 steps MTTR

These results are the main final outcome of the project. The **stress evaluation** is the most important one because it is intentionally harder than the training-like base setting.

---

## Core Ideas in This Implementation

### 1) Four-agent cooperative defense team
The environment uses four agents:

- **Network Agent** – watches simple command / network-side state
- **Host Agent** – monitors host-like telemetry such as CPU behavior
- **Threat-Intelligence Agent** – tracks IoC-style signals
- **Mitigation Agent** – executes the final defense action

The agents are trained jointly with a shared cyber-physical reward.

### 2) LLM semantic context as a state shaper
A local LLM client produces a compact **3-dimensional semantic context vector** that is appended to every agent’s observation.

The semantic channel is meant to act as a lightweight version of the paper’s idea:
- not a full autonomous LLM controller
- not a pure text chatbot
- but a **context signal** that helps tactical agents interpret local patterns

### 3) Containment + recovery
Earlier versions of the environment could clear attacks while still leaving the water level stuck far from the setpoint.  
The final environment adds an explicit **recovery mode**, so successful mitigation is followed by recovery toward the nominal operating band.

This is one of the most important design fixes in the project.

### 4) Base vs held-out stress evaluation
The final evaluation is split into two settings:

- **Base evaluation:** similar to the main simulator setting
- **Stress evaluation:** harder held-out configuration with:
  - more benign confusion
  - weaker IoC visibility
  - lower mitigation success
  - more process disturbances

This prevents the project from relying on a single easy-world score.

---

## Environment Summary

The simulated plant is a water-level control system with:
- a **water level**
- a **pump**
- a **valve**
- **CPU load**
- an **IoC signal**
- stochastic benign disturbances

The default episode horizon is **60 steps**.

### Attack types
The environment can inject one attack per episode (or no attack, depending on configuration):

1. **Overflow**
   - pushes the plant toward overfill
   - strong physical effect
   - moderate cyber signal

2. **Stealth Drain**
   - gradually drives the level downward
   - weaker cyber signature
   - hardest final attack family under stress

3. **Ransomware**
   - shutdown-like actuation pattern
   - strong CPU anomaly
   - easiest attack family in final stress evaluation

---

## Reward Design

The reward follows the same high-level logic as the paper:

R = w_sec * R_security + w_proc * R_process + w_cost * R_cost

with the final local weights:

- `w_sec = 1.0`
- `w_proc = 2.0`
- `w_cost = 0.1`

This means the agents are encouraged to:
- detect and stop attacks
- avoid unnecessary mitigation
- preserve or restore physical stability

---

## Repository Structure

```text
project-root/
│
├── src/
│   ├── agents/
│   │   └── mappo_agent.py        # MAPPO-style actor/critic implementation
│   │
│   ├── envs/
│   │   └── cps_env.py            # Cyber-physical environment
│   │
│   ├── utils/
│   │   ├── llm_client.py         # Local semantic LLM interface + cache
│   │   └── logger.py             # CSV step logger
│   │
│   ├── train.py                  # Training entry point
│   └── evaluate.py               # Base + stress multi-seed evaluation
│
├── models/                       # Saved actor / critic weights + cache
├── logs/                         # Evaluation CSV logs
├── sample_episode_base_seed_11.png
├── sample_episode_stress_seed_11.png
├── README.md
└── requirements.txt