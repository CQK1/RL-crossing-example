import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.network_traffic_env import NetworkTrafficEnv

class TrafficLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrafficLoggingCallback, self).__init__(verbose)
        self.best_reward = float('-inf')
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            self.episode_count += 1
            info = self.locals.get("infos")[0]
            
            episode_reward = info.get("episode", {}).get("r", 0.0)
            episode_length = info.get("episode", {}).get("l", 0)
            
            # 这里我们只负责打印 SB3 的训练轮数和奖励
            print(f"⏱️ Episode {self.episode_count} | Reward: {episode_reward:.1f} | Length: {episode_length}")

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                print(f"🔥 New Best Reward Broken: {self.best_reward:.1f}! Saving best model...")
                os.makedirs("models", exist_ok=True)
                self.model.save("models/best_mayor_magrath_ppo")
            
            print("-" * 80)

        return True

def train_agent(timesteps=500):
    env = NetworkTrafficEnv()
    check_env(env, warn=True)
    
    print("Environment setup successful. Target: Mayor Magrath Intersection.")
    print("Starting PPO training with real traffic data.")

    os.makedirs("models", exist_ok=True)
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=8192,
        batch_size=1024,
        verbose=1, 
        tensorboard_log="./traffic_tensorboard/"
    )
    logging_callback = TrafficLoggingCallback()

    model.learn(total_timesteps=timesteps, callback=logging_callback)
    model.save("models/final_mayor_magrath_ppo")
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    train_agent(timesteps=90000)