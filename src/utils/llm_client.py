import requests
import json
import os
import re


class LLMClient:
    def __init__(self, url="http://localhost:11434/api/generate", model="llama3", debug_mode=False):
        self.url = url
        self.model = model
        self.debug_mode = debug_mode
        self.cache = {}

    def get_context(self, level, pump, valve, cpu_load, ioc_match, mitigator_ack, level_trend):
        rounded_level = round(level, -1)
        rounded_cpu = round(cpu_load, -1)

        ack_map = {0: "Idle", 1: "Port Isolated", 2: "Quarantined", 3: "Full Shutdown"}
        ack_str = ack_map.get(mitigator_ack, "Idle")

        cache_key = f"P:{pump}_V:{valve}_L:{rounded_level}_C:{rounded_cpu}_IOC:{ioc_match}_ACK:{mitigator_ack}_T:{level_trend}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # The Strict Pattern Prompt: Shows the model exactly how to behave without yelling at it.
        prompt = f"""Classify the log using the rules. Output ONLY the bracketed letter.

Rules:
[A]: OVERFLOW (Pump ON + Valve CLOSED)
[B]: DRAIN (Pump OFF + Valve OPEN + Low Level)
[C]: APT (Pump ON + Level Trend DROPPING)
[D]: RANSOMWARE (CPU > 90% + IoC YES)
[E]: NORMAL (None of the above)

Log: Pump: ON, Valve: CLOSED, CPU: 20%, Trend: RISING, IoC: NO
Output: [A]

Log: Pump: ON, Valve: OPEN, CPU: 20%, Trend: STABLE, IoC: NO
Output: [E]

Log: Pump: OFF, Valve: OPEN, CPU: 20%, Trend: DROPPING, IoC: NO
Output: [B]

Log: Pump: {'ON' if pump == 1 else 'OFF'}, Valve: {'OPEN' if valve == 1 else 'CLOSED'}, CPU: {rounded_cpu}%, Trend: {level_trend}, IoC: {'YES' if ioc_match == 1 else 'NO'}
Output:"""

        try:
            response = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2  # Forces the model to be 100% robotic and deterministic
                }
            }, timeout=30)

            output_text = response.json()["response"].strip().upper()

            # The Forgiving Regex Sniper
            choices = re.findall(r'\[\s*([A-E])\s*\]', output_text)

            if choices:
                final_choice = choices[-1]

                vector_map = {
                    "A": [0.9, 0.1, 0.1],
                    "B": [0.1, 0.9, 0.1],
                    "C": [0.5, 0.5, 0.1],
                    "D": [0.1, 0.1, 0.9],
                    "E": [0.0, 0.0, 0.0]
                }

                semantic_vector = vector_map.get(final_choice, [0.0, 0.0, 0.0])

                if self.debug_mode:
                    print(f"\n👀 [RAW OLLAMA]: {output_text}")
                    print(f"  [LLM] ✅ Conclusion '{final_choice}' mapped to {semantic_vector}")
            else:
                semantic_vector = [0.0, 0.0, 0.0]
                if self.debug_mode:
                    print(f"\n👀 [RAW OLLAMA]: {output_text}")
                    print(f"  [LLM] ⚠️ No bracketed [A-E] found. Defaulting to Normal [0.0, 0.0, 0.0].")

            self.cache[cache_key] = semantic_vector
            return semantic_vector

        except Exception as e:
            if self.debug_mode:
                print(f"  [LLM] ⚠️ CRASH/TIMEOUT! Defaulting to [0.0, 0.0, 0.0]")
            return [0.0, 0.0, 0.0]

    def save_cache(self, filepath="models/llm_cache.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.cache, f)

    def load_cache(self, filepath="models/llm_cache.json"):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.cache = json.load(f)