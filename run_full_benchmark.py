import os
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environment.network_traffic_env import NetworkTrafficEnv
from src.agents.agent import QLearningAgent

# Load global configuration
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DECISION_INTERVAL = CONFIG["traffic_env"]["decision_interval"]
EVAL_EPISODES = CONFIG["benchmark"]["eval_episodes"]
Q_TABLE_PATH = CONFIG["benchmark"]["q_table_path"]
FIXED_HOLD_SECONDS = CONFIG["benchmark"]["fixed_time_hold_seconds"]

# Which policies to evaluate (configurable via config.json)
POLICIES_TO_EVAL = CONFIG["benchmark"].get(
    "policies",
    ["Baseline (Fixed-Time)", "Deep RL (PPO)"]
)

# Calculate how many decision steps to hold each phase
HOLD_STEPS = max(1, FIXED_HOLD_SECONDS // DECISION_INTERVAL)


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


def array_to_dict(state_array):
    return {
        'current_phase': state_array[0],
        'phase_timer': state_array[1],
        'queue_ns_straight': state_array[2],
        'queue_ns_left': state_array[3],
        'queue_ew_straight': state_array[4],
        'queue_ew_left': state_array[5],
        'max_wait_ns_straight': state_array[6],
        'max_wait_ns_left': state_array[7],
        'max_wait_ew_straight': state_array[8],
        'max_wait_ew_left': state_array[9]
    }


def evaluate_policy(policy_type, episodes=EVAL_EPISODES):
    # Base environment for Baseline and Q-Learning (no VecNormalize needed)
    env = NetworkTrafficEnv(decision_interval=DECISION_INTERVAL)

    if policy_type == "Baseline (Fixed-Time)":
        agent = FixedTimeBaselineAgent()
        norm_env = None  # No VecNormalize for Baseline

    elif policy_type == "Tabular Q-Learning":
        agent = QLearningAgent()
        if os.path.exists(Q_TABLE_PATH):
            agent.load_q_table(Q_TABLE_PATH)
        agent.epsilon = 0.0
        norm_env = None

    elif policy_type == "Deep RL (PPO)":
        def make_env():
            return NetworkTrafficEnv(decision_interval=DECISION_INTERVAL)

        vec_env = DummyVecEnv([make_env])
        norm_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

        # Load normalization statistics saved during training
        vec_norm_path = os.path.join(CONFIG["training"]["model_dir"], "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            norm_env.load(vec_norm_path)
            print(f"  └── Loaded VecNormalize stats from {vec_norm_path}")

        model_file = os.path.join(CONFIG["training"]["model_dir"], CONFIG["training"]["best_model_name"])
        ppo_model = PPO.load(model_file, env=norm_env)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    delays, throughputs, phase_switches = [], [], []

    for ep in range(episodes):
        # Reset the correct environment
        if policy_type == "Deep RL (PPO)":
            state = norm_env.reset()
        else:
            state, _ = env.reset(seed=42 + ep)

        done = False
        ep_delay = 0.0
        switches = 0
        last_phase = state[0]

        while not done:
            if policy_type == "Baseline (Fixed-Time)":
                action = agent.select_action()
            elif policy_type == "Tabular Q-Learning":
                dict_state = array_to_dict(state)
                action = agent.select_action(dict_state)
            elif policy_type == "Deep RL (PPO)":
                action_arr, _ = ppo_model.predict(state, deterministic=True)
                action = int(action_arr[0])

            if action != last_phase:
                switches += 1
                last_phase = action

            # Step the correct environment
            if policy_type == "Deep RL (PPO)":
                next_state, reward, terminated, truncated, _ = norm_env.step([action])
            else:
                next_state, reward, terminated, truncated, _ = env.step([action])

            done = terminated or truncated
            ep_delay += abs(reward)
            state = next_state

        stats = env.traffic_map.intersections["Mayor_Magrath"].stats
        total_thru = stats["straight"] + stats["left"] + stats["right"] + stats["u_turn"]

        delays.append(ep_delay)
        throughputs.append(total_thru)
        phase_switches.append(switches)
        print(f"  └── [{policy_type}] Finished Ep {ep + 1}/{episodes} | Delay: {ep_delay:,.0f} | Throughput: {total_thru}")

    mean_delay = np.mean(delays)
    mean_throughput = np.mean(throughputs)
    avg_delay_per_veh = mean_delay / max(mean_throughput, 1.0)
    mean_switches = np.mean(phase_switches)

    return {
        "Policy": policy_type,
        "Total Delay (Veh-s)": mean_delay,
        "Throughput (Veh)": mean_throughput,
        "Avg Delay per Vehicle (s/veh)": avg_delay_per_veh,
        "Phase Switches (/day)": mean_switches
    }


def run_full_benchmark():
    print(f"Running Benchmark with Global Config (Decision Interval = {DECISION_INTERVAL}s)...")
    print(f"Policies to evaluate: {POLICIES_TO_EVAL}")

    results = []

    for p in POLICIES_TO_EVAL:
        print(f"\nEvaluating: {p}")
        res = evaluate_policy(p, episodes=EVAL_EPISODES)
        results.append(res)

    df = pd.DataFrame(results)

    # Calculate improvement relative to Baseline if Baseline is present
    if "Baseline (Fixed-Time)" in df["Policy"].values:
        base_delay = df.loc[df["Policy"] == "Baseline (Fixed-Time)", "Total Delay (Veh-s)"].values[0]
        df["Delay Improvement vs Baseline (%)"] = ((base_delay - df["Total Delay (Veh-s)"]) / base_delay) * 100.0

    print("\n" + "=" * 80)
    print("                     FINAL MULTI-METRIC COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Auto-save results to CSV
    df.to_csv("benchmark_results.csv", index=False, float_format="%.2f")
    print("Results successfully exported to benchmark_results.csv")


if __name__ == "__main__":
    run_full_benchmark()