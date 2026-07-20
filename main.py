import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.network_traffic_env import NetworkTrafficEnv

class TrafficLoggingCallback(BaseCallback):
    """
    用于在每个 Episode 结束时追踪和记录交通吞吐量指标的回调函数。
    """
    def __init__(self, verbose=0):
        super(TrafficLoggingCallback, self).__init__(verbose)
        # 初始化历史最佳奖励和 episode 计数器
        self.best_reward = float('-inf')
        self.episode_count = 0

    def _on_step(self) -> bool:
        # 1. 检查当前 Episode 是否结束
        if self.locals.get("dones")[0]:
            self.episode_count += 1
            info = self.locals.get("infos")[0]
            
            # 从 Stable-Baselines3 的 Monitor 结构中提取真实奖惩和步数
            episode_reward = info.get("episode", {}).get("r", 0.0)
            episode_length = info.get("episode", {}).get("l", 0)

            # 2. 提取单路口的吞吐量统计（Mayor_Magrath）
            unwrapped_env = self.training_env.envs[0].unwrapped
            intersection = unwrapped_env.traffic_map.intersections["Mayor_Magrath"]
            
            stats_msg = f"⏱️ Episode {self.episode_count} | Reward: {episode_reward:.1f} | Length: {episode_length} \n"
            stats_msg += f"📊 Throughput -> {intersection.stats}"
            
            print("-" * 80)
            print(stats_msg)

            # 3. 核心加回：追踪历史最高 reward 并保存最佳模型
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                print(f"🔥 New Best Reward Broken: {self.best_reward:.1f}! Saving best model...")
                os.makedirs("models", exist_ok=True)
                self.model.save("models/best_mayor_magrath_ppo")
            
            print("-" * 80)
            
            # 4. 为下一个 Episode 重置路口吞吐量统计
            intersection.reset_stats()

        return True

def train_agent(timesteps=500):
    # 环境初始化 (注意：此处已移除 controlled_nodes 参数以适配你的新环境)
    env = NetworkTrafficEnv()
    check_env(env, warn=True)
    
    print("Environment setup successful. Target: Mayor Magrath Intersection.")
    print("Starting PPO training with real traffic data.")

    # 模型与保存路径初始化
    os.makedirs("models", exist_ok=True)
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=8192,          # 每次累积 8192 步才更新一次网络，减少计算卡顿
        batch_size=1024,       # 增大批处理大小，加速矩阵运算
        verbose=1, 
        tensorboard_log="./traffic_tensorboard/"
    )
    logging_callback = TrafficLoggingCallback()

    # 开始训练
    model.learn(total_timesteps=timesteps, callback=logging_callback)

    # 保存最终权重
    model.save("models/final_mayor_magrath_ppo")
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    # 执行训练
    train_agent(timesteps=90000)