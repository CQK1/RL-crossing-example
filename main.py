import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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

class TrafficLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrafficLoggingCallback, self).__init__(verbose)
        self.best_reward = float('-inf')
        self.episode_count = 0
        self.current_ep_normalized_reward = 0.0

    def _on_step(self) -> bool:
        step_rewards = self.locals.get("rewards")
        if step_rewards is not None:
            self.current_ep_normalized_reward += float(step_rewards[0])

        if self.locals.get("dones")[0]:
            self.episode_count += 1
            info = self.locals.get("infos")[0]
            
            episode_reward = info.get("episode", {}).get("r", 0.0)
            episode_length = info.get("episode", {}).get("l", 0)
            
            print(f"   Episode {self.episode_count} | Steps: {episode_length}")
            print(f"   └── Raw Reward:  {episode_reward:.1f}")
            print(f"   └── Norm Reward: {self.current_ep_normalized_reward:.2f}")

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                print(f" New Best Raw Reward: {self.best_reward:.1f}! Saving best model...")
                os.makedirs(CONFIG["training"]["model_dir"], exist_ok=True)
                
                # Save both model weights and normalization statistics
                best_model_path = os.path.join(CONFIG["training"]["model_dir"], CONFIG["training"]["best_model_name"])
                self.model.save(best_model_path)
                self.training_env.save(os.path.join(CONFIG["training"]["model_dir"], "vec_normalize.pkl"))
            
            print("-" * 80)
            self.current_ep_normalized_reward = 0.0

        return True

def train_agent(timesteps=TOTAL_TIMESTEPS):
    # Pass the dynamic decision interval from config
    def make_env():
        base_env = NetworkTrafficEnv(decision_interval=DECISION_INTERVAL)
        return Monitor(base_env)

    print(f"Environment setup: Interval = {DECISION_INTERVAL}s | Steps/Day = {STEPS_PER_DAY} | Total Timesteps = {timesteps}")
    print("Starting PPO training with real traffic data.")

    os.makedirs(CONFIG["training"]["model_dir"], exist_ok=True)

    vec_env = DummyVecEnv([make_env])
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=CONFIG["training"]["n_steps"],
        batch_size=CONFIG["training"]["batch_size"],
        verbose=CONFIG["training"]["verbose"], 
        tensorboard_log=CONFIG["training"]["tensorboard_log"]
    )
    logging_callback = TrafficLoggingCallback()

    model.learn(total_timesteps=timesteps, callback=logging_callback)
    
    # Save final models and final statistics
    final_model_path = os.path.join(CONFIG["training"]["model_dir"], CONFIG["training"]["final_model_name"])
    model.save(final_model_path)
    env.save(os.path.join(CONFIG["training"]["model_dir"], "vec_normalize_final.pkl"))
    print(f"Training complete. Final model saved to {final_model_path}.")

if __name__ == "__main__":
    train_agent(timesteps=TOTAL_TIMESTEPS)