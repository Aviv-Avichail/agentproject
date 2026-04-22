import csv
import os
from datetime import datetime


class L2MAID_Logger:
    def __init__(self, filename="defense_performance.csv"):
        self.log_dir = "logs"
        self.log_path = os.path.join(self.log_dir, filename)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize headers if file doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Episode", "Step", "Attack_Type",
                    "Water_Level", "CPU_Load", "LLM_Decision",
                    "Agent_Action", "Is_Anomaly"
                ])

    def log_step(self, ep, step, attack_type, level, cpu, llm_decision, action, is_anomaly):
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ep, step, attack_type, round(level, 2),
                round(cpu, 2), llm_decision, action, is_anomaly
            ])