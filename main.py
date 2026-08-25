import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.network_traffic_env import NetworkTrafficEnv

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
            
            #### PRINT
            print(f"   Episode {self.episode_count} | Steps: {episode_length}")
            print(f"   └── Raw Reward:       {episode_reward:.1f}")
            print(f"   └── Norm Reward: {self.current_ep_normalized_reward:.2f}")

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                print(f"New Best Raw Reward: {self.best_reward:.1f}! Saving best model...")
                os.makedirs("models", exist_ok=True)
                self.model.save("models/best_mayor_magrath_ppo")
            
            print("-" * 80)

        return True

def train_agent(timesteps=500):
    def make_env():
        base_env = NetworkTrafficEnv(decision_interval=240)
        return Monitor(base_env) # Monitor must wrap the base environment directly

    # 1. Define factory function wrapping environment with Monitor first
    
    print("Environment setup successful. Target: Mayor Magrath Intersection.")
    print("Starting PPO training with real traffic data.")

    os.makedirs("models", exist_ok=True)

    # Wrap the environment with automatic reward normalization
    vec_env = DummyVecEnv([make_env])
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=360,
        batch_size=60,
        verbose=1, 
        tensorboard_log="./traffic_tensorboard/"
    )
    logging_callback = TrafficLoggingCallback()

    model.learn(total_timesteps=timesteps, callback=logging_callback)
    model.save("models/final_mayor_magrath_ppo")
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    train_agent(timesteps=3600)