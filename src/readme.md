# L2M-AID: Autonomous Cyber-Physical Defense via Neuro-Symbolic MARL

This repository contains an implementation of the **L2M-AID** (Large Language Model to Multi-Agent Intrusion Detection) framework, adapted for a simulated Industrial Control System (ICS) environment. 

This project demonstrates the fusion of **Large Language Models (LLMs)** for high-level semantic reasoning with **Multi-Agent Reinforcement Learning (MARL)** for high-frequency, adaptive tactical control. It is designed to autonomously detect and mitigate sophisticated cyber-physical threats in an Industrial IoT (IIoT) water treatment facility.

## 🧠 Architectural Overview

The system operates on a **Centralized Training with Decentralized Execution (CTDE)** paradigm using the MAPPO algorithm. It is structured hierarchically:

1. **Strategic Orchestrator (LLM):** Powered by a local instance of `llama3.2:1b`, this agent acts as the cognitive core. It asynchronously correlates multi-source alerts and physical telemetry to generate a continuous contextual embedding vector (`[Threat, Normal, Isolate]`).
2. **Tactical Layer (4-Agent MARL):**
   * **Network Monitoring Agent:** Inspects simulated network commands (Pump/Valve states).
   * **Host Analysis Agent:** Monitors endpoint telemetry (CPU load spikes).
   * **Threat Intelligence Agent:** Cross-references external Indicators of Compromise (IoCs).
   * **Mitigation Agent:** The execution arm, authorized to perform dynamic port isolation, quarantines, or full system shutdowns.

## 🚀 Novel Extensions (Project Innovation)

This implementation extends the original academic framework by introducing two novel mechanisms to test the agents against Advanced Persistent Threats (APTs) and prevent policy overfitting:

* **Stateful Temporal Memory (Low & Slow APT Detection):** Traditional rule-engines fail against "low and slow" attacks. We implemented a temporal memory queue (`deque`) that calculates the Rate of Change ($\Delta$) of the physical water level over a sliding window. By passing this trend to the LLM, the system can detect physics-command mismatches (e.g., Pump is ON, but water is DROPPING), successfully thwarting stealthy draining attacks.
* **Domain Randomization Attack Matrix:** To prevent the MARL agents from memorizing a single attack pattern, the environment dynamically injects one of three distinct attack vectors per episode:
  1. *Overflow:* High noise, pump forced ON, valve CLOSED.
  2. *Stealth Drain:* Low noise, pump forced OFF, valve OPEN, spoofed normal CPU.
  3. *Ransomware / DoS:* Pump OFF, valve CLOSED, 99% CPU spike.

## 📂 Repository Structure

```text
l2maid-simulation/
│
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── llm_client.py       # Handles Ollama API, Semantic prompting, and caching
│   │
│   ├── envs/
│   │   ├── __init__.py
│   │   └── cps_env.py          # PettingZoo Dec-POMDP environment with continuous physics
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── mappo_agent.py      # CTDE MAPPO implementation (Actor & Critic networks)
│   │
│   ├── train.py                # Main training loop and memory buffer collection
│   └── evaluate.py             # Deterministic evaluation script 
│
├── models/                     # Saved PyTorch .pth weights
├── README.md
└── requirements.txt
```

## ⚙️ Installation & Prerequisites

**1. Python Dependencies**
Ensure you have Python 3.9+ installed. Install the required packages:
```bash
pip install torch numpy pettingzoo gymnasium requests matplotlib
```

**2. Local LLM (Ollama)**
This project relies on a local LLM to eliminate API latency and costs during training. 
* Download and install [Ollama](https://ollama.com/).
* Pull the required Llama 3.2 model:
```bash
ollama run llama3.2:1b
```
*Ensure the Ollama service is running locally on port `11434` before starting the training script.*

## 💻 Usage

**Training the Agents**
To train the 4-agent team from scratch using the CTDE MAPPO algorithm:
```bash
python -m src.train
```
*Note: Set `debug_mode=True` in `train.py` to see the real-time LLM semantic vector generations and physical environment responses.*

**Evaluating the Model**
To test the trained actors (Decentralized Execution) without the global critic or exploration noise:
```bash
python -m src.evaluate
```

## 📊 Evaluation Metrics

The system's performance is jointly evaluated on both cybersecurity efficacy and physical process safety:
* **Detection Rate (DR) & False Positive Rate (FPR)**
* **Mean Time To Respond (MTTR):** Measured in timesteps from attack injection to mitigation.
* **Process Stability Index (PSI):** An engineered metric that heavily penalizes the AI for triggering false-positive shutdowns that disrupt the continuous physical flow of the simulated water plant.