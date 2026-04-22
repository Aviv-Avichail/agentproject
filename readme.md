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

`R = w_sec * R_security + w_proc * R_process + w_cost * R_cost`

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
```

---

## Requirements

### Python version
- Python **3.9+** is recommended.

### Python libraries
Install the required packages with:

```bash
pip install torch numpy matplotlib requests pettingzoo gymnasium
```

If you want a `requirements.txt`, it should include at least:

```text
torch
numpy
matplotlib
requests
pettingzoo
gymnasium
```

### Local LLM requirement
This project expects a **local Ollama server** and a **Llama 3 model** for the semantic context channel.

Start Ollama:

```bash
ollama serve
```

Pull the model used by the project:

```bash
ollama pull llama3
```

If your `src/utils/llm_client.py` is configured for a different local model name, update it there and pull that model instead.

### Quick setup
```bash
pip install torch numpy matplotlib requests pettingzoo gymnasium
ollama serve
ollama pull llama3
```

---

## Installation

### 1) Python
Use Python 3.9+.

### 2) Dependencies
Install the required libraries:

```bash
pip install torch numpy matplotlib requests pettingzoo gymnasium
```

### 3) Local LLM backend
Run **Ollama** locally:

```bash
ollama serve
```

Then pull the model expected by the project:

```bash
ollama pull llama3
```

> If your local config points to a different model name, update `src/utils/llm_client.py` accordingly.

---

## How to Run

### Train from scratch
```bash
python -m src.train
```

This trains the 4-agent policy and saves weights into `models/`.

### Run final evaluation
```bash
python -m src.evaluate
```

The final evaluation script runs:
- **Base evaluation**
- **Stress evaluation**
- **5 random seeds**
- mean ± std summary across runs

It also writes per-seed CSV logs to `logs/` and saves representative trajectory plots.

---

## Typical Outputs

Running evaluation should generate files such as:

- `logs/eval_base_seed_11.csv`
- `logs/eval_base_seed_22.csv`
- `logs/eval_stress_seed_11.csv`
- `logs/eval_stress_seed_22.csv`
- `sample_episode_base_seed_11.png`
- `sample_episode_stress_seed_11.png`

These artifacts were used in the final report and presentation.

---

## What Changed During the Project

This project went through several major revisions.

### Early issue: perfect-looking scores
At one stage, the system achieved near-100% results in a way that looked too easy.  
The fix was **not** endless retraining. Instead, the final version added a **held-out stress evaluation**.

### Early issue: containment without recovery
A more serious bug appeared when the anomaly would clear, but the plant would still drift away from the setpoint afterward.

The final version fixed this by adding:
- an explicit **recovery mode**
- a tighter **safe operating band**
- a **hold condition** before exiting recovery

This made the sample trajectories much more realistic.

---

## Main Findings

1. **The LLM semantic channel helps shape the tactical state.**
2. **Recovery matters as much as containment in CPS defense.**
3. **Single-setting near-perfect performance is not a good headline result.**
4. **Held-out stress evaluation is far more meaningful than one easy score.**
5. **Stealth drain is the hardest attack family in the final setup.**
6. **The learned mitigation policy is still somewhat narrow** (notably, action 2 was never used in the final runs).

---

## Limitations

This is still a **simplified simulator**, not a full reproduction of the original paper.

Limitations include:
- no direct SWaT replay pipeline
- simplified plant physics
- simplified cyber telemetry
- compact LLM semantic channel instead of a full orchestration pipeline
- limited attack diversity
- a mitigation policy that still underuses part of the action space

So the results should be interpreted as:
- a **working proof-of-concept implementation**
- a **course project extension**
- not a claim of real-world deployment readiness

---

## Links

- **Paper reviewed:** *L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning*
- **GitHub repository:** `<PASTE_GITHUB_LINK_HERE>`
- **Technical report (LaTeX/PDF):** `<PASTE_REPORT_LINK_HERE>`
- **Presentation slides:** `<PASTE_SLIDES_LINK_HERE>`

---

## Suggested Citation

If you want to cite this repository in your report or slides, use something like:

```bibtex
@misc{l2maid_course_project_2026,
  title        = {L2M-AID-Style Autonomous Cyber-Physical Defense with Neuro-Symbolic MARL},
  author       = {Your Name and Your Partner Name},
  year         = {2026},
  note         = {Course project implementation and experimental extension}
}
```

---

## Acknowledgment

This project was developed as part of a course assignment on **AI Agents in Cybersecurity**. It builds on the ideas introduced in the L2M-AID paper while implementing a simplified simulation and a held-out stress evaluation designed to test robustness in a more realistic way.
