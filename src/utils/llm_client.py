import requests
import ast

class LLMClient:
    def __init__(self, url="http://localhost:11434/api/generate", model="llama3.2:1b", debug_mode=False):
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

        prompt = f"""
        Analyze this Industrial Control System log:
        Water Level: {rounded_level}
        Level Trend (Last 5 mins): {level_trend}
        Pump Command: {'ON' if pump == 1 else 'OFF'}
        Valve Command: {'OPEN' if valve == 1 else 'CLOSED'}
        Host CPU Load: {rounded_cpu}%
        Known Malicious IoC Detected: {'YES' if ioc_match == 1 else 'NO'}
        Mitigator ACK (Last Action): {ack_str}

        Rules for Threat Detection:
        1. OVERFLOW ATTACK: Pump is ON and Valve is CLOSED.
        2. FAST DRAIN ATTACK: Pump is OFF and Valve is OPEN while Level is dangerously low.
        3. LOW & SLOW (APT): Pump is ON, but Level Trend is 'DROPPING'. (Physics mismatch).
        4. DoS/RANSOMWARE: CPU is > 90% and IoC is YES, regardless of physics.

        If ANY of these conditions are met, this is a cyber attack. 
        Note: If Mitigator ACK is 'Full Shutdown' or 'Port Isolated', the threat may be temporarily contained.
        Output exactly a python list of three floats (Threat, Normal, Isolate). Example: [0.9, 0.1, 0.9] if attack, [0.1, 0.9, 0.1] if normal.
        Only output the list, no text.
        """

        try:
            response = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }, timeout=30)

            output_text = response.json()["response"].strip()

            if "[" in output_text and "]" in output_text:
                output_text = output_text[output_text.find("["):output_text.find("]") + 1]

            semantic_vector = ast.literal_eval(output_text)

            if len(semantic_vector) == 3:
                self.cache[cache_key] = semantic_vector
                if self.debug_mode:
                    print(f"  [LLM] ✅ Ollama: {semantic_vector}. Cached.")
                return semantic_vector

        except Exception as e:
            if self.debug_mode:
                print(f"  [LLM] ⚠️ Ollama failed. Error: {e}")
            return [0.5, 0.5, 0.0]

        return [0.5, 0.5, 0.0]