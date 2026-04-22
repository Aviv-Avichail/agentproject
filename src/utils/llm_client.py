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

    def _level_band(self, level):
        if level < 42:
            return "LOW"
        if level > 58:
            return "HIGH"
        return "NORMAL"

    def get_context(self, level, pump, valve, cpu_load, ioc_match, mitigator_ack, level_trend):
        rounded_level = round(level, -1)
        rounded_cpu = round(cpu_load, -1)
        level_band = self._level_band(level)

        cache_key = (
            f"P:{pump}_V:{valve}_L:{rounded_level}_LB:{level_band}"
            f"_C:{rounded_cpu}_IOC:{ioc_match}_ACK:{mitigator_ack}_T:{level_trend}"
        )

        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""Classify the control-system situation using the rules. Output ONLY the bracketed letter.

Rules:
[A]: OVERFLOW (Pump ON + Valve CLOSED + Trend RISING + Level HIGH)
[B]: DRAIN (Pump OFF + Valve OPEN + Trend DROPPING + Level LOW)
[C]: SUSPICIOUS (unsafe physical trend or partial mismatch, but not enough for A/B/D)
[D]: RANSOMWARE (CPU > 90% + IoC YES)
[E]: NORMAL (none of the above)

Example 1
Log: Pump: ON, Valve: CLOSED, CPU: 20%, Trend: RISING (Unsafe), Level: HIGH, IoC: NO
Output: [A]

Example 2
Log: Pump: OFF, Valve: OPEN, CPU: 25%, Trend: DROPPING (Unsafe), Level: LOW, IoC: NO
Output: [B]

Example 3
Log: Pump: OFF, Valve: OPEN, CPU: 28%, Trend: DROPPING (Unsafe), Level: NORMAL, IoC: NO
Output: [C]

Example 4
Log: Pump: OFF, Valve: CLOSED, CPU: 99%, Trend: STABLE, Level: NORMAL, IoC: YES
Output: [D]

Example 5
Log: Pump: ON, Valve: OPEN, CPU: 23%, Trend: STABLE, Level: NORMAL, IoC: NO
Output: [E]

Current Log:
Pump: {'ON' if pump == 1 else 'OFF'}
Valve: {'OPEN' if valve == 1 else 'CLOSED'}
CPU: {rounded_cpu}%
Trend: {level_trend}
Level: {level_band}
IoC: {'YES' if ioc_match == 1 else 'NO'}
Output:"""

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2
                    }
                },
                timeout=30
            )

            output_text = response.json()["response"].strip().upper()
            choices = re.findall(r"\[\s*([A-E])\s*\]", output_text)

            if choices:
                final_choice = choices[-1]

                # Three semantic channels broadcast to agents:
                # [overflow_like, drain_like, ransomware_like]
                vector_map = {
                    "A": [0.95, 0.05, 0.05],
                    "B": [0.05, 0.95, 0.05],
                    "C": [0.30, 0.40, 0.10],
                    "D": [0.05, 0.05, 0.95],
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
                    print("  [LLM] ⚠️ No bracketed [A-E] found. Defaulting to [0.0, 0.0, 0.0].")

            self.cache[cache_key] = semantic_vector
            return semantic_vector

        except Exception:
            if self.debug_mode:
                print("  [LLM] ⚠️ CRASH/TIMEOUT! Defaulting to [0.0, 0.0, 0.0]")
            return [0.0, 0.0, 0.0]

    def save_cache(self, filepath="models/llm_cache.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.cache, f)

    def load_cache(self, filepath="models/llm_cache.json"):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.cache = json.load(f)