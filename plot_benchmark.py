# plot_benchmark.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from stable_baselines3 import PPO

from src.environment.network_traffic_env import NetworkTrafficEnv
from src.agents.agent import QLearningAgent

# Load configuration
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DECISION_INTERVAL = CONFIG["traffic_env"]["decision_interval"]
FIXED_HOLD_SECONDS = CONFIG["benchmark"]["fixed_time_hold_seconds"]
HOLD_STEPS = max(1, FIXED_HOLD_SECONDS // DECISION_INTERVAL)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class FixedTimeBaselineAgent:
    def __init__(self, cycle_phases=[0, 1, 2, 3], hold_steps=HOLD_STEPS):
        self.cycle_phases = cycle_phases
        self.hold_steps = hold_steps
        self.step_count = 0
        self.current_idx = 0

    def select_action(self, *args, **kwargs):
        action = self.cycle_phases[self.current_idx]
        self.step_count += 1
        if self.step_count >= self.hold_steps:
            self.step_count = 0
            self.current_idx = (self.current_idx + 1) % len(self.cycle_phases)
        return action


def collect_policy_trajectory(policy_name: str):
    """
    Run one full-day simulation and record time-series queue and phase data.
    """
    env = NetworkTrafficEnv(decision_interval=DECISION_INTERVAL)
    ppo_model = None

    if policy_name == "Baseline (Fixed-Time)":
        agent = FixedTimeBaselineAgent()
    elif policy_name == "Deep RL (PPO)":
        model_file = os.path.join(CONFIG["training"]["model_dir"], CONFIG["training"]["best_model_name"])
        ppo_model = PPO.load(model_file)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    state, _ = env.reset(seed=42)
    done = False

    time_records = []
    queue_records = []
    phase_records = []
    delay_records = []

    while not done:
        if policy_name == "Baseline (Fixed-Time)":
            action = agent.select_action()
        else:
            action_arr, _ = ppo_model.predict(state, deterministic=True)
            action = int(action_arr[0])

        current_second = env.time_step
        next_state, reward, terminated, truncated, _ = env.step([action])
        done = terminated or truncated

        # Extract current queue count across all incoming lanes
        intersection = env.traffic_map.intersections["Mayor_Magrath"]
        total_queued = 0
        for lane in intersection.incoming_lanes:
            total_queued += sum(1 for car in lane.vehicles if car.speed <= 0.1)
            total_queued += getattr(lane, "virtual_queue_count", 0)

        time_records.append(current_second / 3600.0)  # Convert seconds to hours
        queue_records.append(total_queued)
        phase_records.append(action)
        delay_records.append(abs(reward))

        state = next_state

    stats = env.traffic_map.intersections["Mayor_Magrath"].stats
    total_thru = sum(stats.values())

    return {
        "times": np.array(time_records),
        "queues": np.array(queue_records),
        "phases": np.array(phase_records),
        "delays": np.array(delay_records),
        "throughput": total_thru,
        "total_delay": np.sum(delay_records)
    }


def generate_visualizations():
    print("Collecting full-day simulation trajectories...")
    baseline_traj = collect_policy_trajectory("Baseline (Fixed-Time)")
    ppo_traj = collect_policy_trajectory("Deep RL (PPO)")

    # -------------------------------------------------------------
    # Figure 1: 24-Hour Queue Length Time-Series Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(13, 5))
    plt.plot(
        baseline_traj["times"],
        baseline_traj["queues"],
        label="Fixed-Time Baseline",
        color="#e74c3c",
        linewidth=1.8,
        alpha=0.85
    )
    plt.plot(
        ppo_traj["times"],
        ppo_traj["queues"],
        label="Deep RL (PPO)",
        color="#2ecc71",
        linewidth=2.0
    )

    plt.title("24-Hour Queue Length Comparison (Lethbridge Intersection)", fontsize=14, fontweight="bold")
    plt.xlabel("Simulation Time (Hour of Day)", fontsize=12)
    plt.ylabel("Total Queued Vehicles", fontsize=12)
    plt.xlim(0, 24)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig("Queue_Comparison_24H.png", dpi=300)
    plt.close()
    print("Saved: Queue_Comparison_24H.png")

    # -------------------------------------------------------------
    # Figure 2: Macro Metrics Bar Chart
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    policies = ["Baseline", "Deep RL (PPO)"]
    colors = ["#e74c3c", "#2ecc71"]

    # Avg Delay per Vehicle
    avg_delays = [
        baseline_traj["total_delay"] / max(baseline_traj["throughput"], 1),
        ppo_traj["total_delay"] / max(ppo_traj["throughput"], 1)
    ]
    axes[0].bar(policies, avg_delays, color=colors, width=0.45)
    axes[0].set_title("Average Delay per Vehicle (s/veh)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Seconds")
    for i, val in enumerate(avg_delays):
        axes[0].text(i, val * 1.02, f"{val:.1f}s", ha="center", fontweight="bold")

    # Throughput
    throughputs = [baseline_traj["throughput"], ppo_traj["throughput"]]
    axes[1].bar(policies, throughputs, color=colors, width=0.45)
    axes[1].set_title("Total 24H Throughput (Vehicles)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Vehicle Count")
    for i, val in enumerate(throughputs):
        axes[1].text(i, val * 1.02, f"{val:,}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("Macro_Performance_Benchmark.png", dpi=300)
    plt.close()
    print("Saved: Macro_Performance_Benchmark.png")

    # -------------------------------------------------------------
    # Figure 3: Phase Allocation Proportions (Hourly Distribution)
    # -------------------------------------------------------------
    df_ppo = pd.DataFrame({
        "hour": np.floor(ppo_traj["times"]).astype(int),
        "phase": ppo_traj["phases"]
    })
    phase_hourly = pd.crosstab(df_ppo["hour"], df_ppo["phase"], normalize='index') * 100

    plt.figure(figsize=(13, 5))
    phase_labels = ["Phase 0 (NS-Straight)", "Phase 1 (NS-Left)", "Phase 2 (EW-Straight)", "Phase 3 (EW-Left)"]
    bottom = np.zeros(len(phase_hourly))

    for phase_idx in range(4):
        if phase_idx in phase_hourly.columns:
            values = phase_hourly[phase_idx].values
            plt.bar(
                phase_hourly.index,
                values,
                bottom=bottom,
                label=phase_labels[phase_idx],
                width=0.8
            )
            bottom += values

    plt.title("PPO Dynamic Green-Time Phase Allocation by Hour", fontsize=14, fontweight="bold")
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Phase Proportion (%)", fontsize=12)
    plt.xlim(-0.5, 23.5)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("PPO_Phase_Allocation_Hourly.png", dpi=300)
    plt.close()
    print("Saved: PPO_Phase_Allocation_Hourly.png")


if __name__ == "__main__":
    generate_visualizations()