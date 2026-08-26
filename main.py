import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.network_traffic_env import NetworkTrafficEnv

# 1. Load global configuration parameters
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DECISION_INTERVAL = CONFIG["traffic_env"]["decision_interval"]
DAY_SECONDS = CONFIG["traffic_env"]["simulation_day_seconds"]
TRAIN_DAYS = CONFIG["training"]["train_days"]

# Calculate total timesteps dynamically based on config
STEPS_PER_DAY = DAY_SECONDS // DECISION_INTERVAL
TOTAL_TIMESTEPS = STEPS_PER_DAY * TRAIN_DAYS

# Number of parallel environments (adjust based on CPU cores)
NUM_ENVS = 1


class TrafficLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrafficLoggingCallback, self).__init__(verbose)
        self.best_reward = float("-inf")
        self.episode_count = 0
        self.current_ep_normalized_reward = 0.0

    def _on_step(self) -> bool:
        step_rewards = self.locals.get("rewards")
        if step_rewards is not None:
            self.current_ep_normalized_reward += float(np.mean(step_rewards))

        dones = self.locals.get("dones")
        if dones is not None and any(dones):
            # Print normalized cumulative reward instead of raw Monitor reward
            print(f"   Episode finished")
            print(f"   └── Normalized Cumulative Reward: {self.current_ep_normalized_reward:.2f}")

            if self.current_ep_normalized_reward > self.best_reward:
                self.best_reward = self.current_ep_normalized_reward
                print(f"   New Best Normalized Reward: {self.best_reward:.2f}! Saving best model...")
                os.makedirs(CONFIG["training"]["model_dir"], exist_ok=True)
                best_model_path = os.path.join(
                    CONFIG["training"]["model_dir"],
                    CONFIG["training"]["best_model_name"],
                )
                self.model.save(best_model_path)
                self.training_env.save(
                    os.path.join(CONFIG["training"]["model_dir"], "vec_normalize.pkl")
                )

            print("-" * 80)
            self.current_ep_normalized_reward = 0.0

        return True
def train_agent(timesteps=TOTAL_TIMESTEPS):
    def make_env():
        base_env = NetworkTrafficEnv(decision_interval=DECISION_INTERVAL)
        return Monitor(base_env)

    print(f"Environment setup: Interval = {DECISION_INTERVAL}s | Steps/Day = {STEPS_PER_DAY} | Total Timesteps = {timesteps}")
    print("Starting PPO training with real traffic data.")

    os.makedirs(CONFIG["training"]["model_dir"], exist_ok=True)

    # Single environment (fastest for lightweight simulations)
    vec_env = DummyVecEnv([make_env])
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=CONFIG["training"]["n_steps"],
        batch_size=CONFIG["training"]["batch_size"],
        verbose=CONFIG["training"]["verbose"],
        tensorboard_log=CONFIG["training"]["tensorboard_log"],
    )
    logging_callback = TrafficLoggingCallback()

    model.learn(total_timesteps=timesteps, callback=logging_callback)

    final_model_path = os.path.join(CONFIG["training"]["model_dir"], CONFIG["training"]["final_model_name"])
    model.save(final_model_path)
    env.save(os.path.join(CONFIG["training"]["model_dir"], "vec_normalize_final.pkl"))
    print(f"Training complete. Final model saved to {final_model_path}.")

if __name__ == "__main__":
    import numpy as np  # for callback mean calculation

    train_agent(timesteps=TOTAL_TIMESTEPS)
