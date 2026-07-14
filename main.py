import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback  # Import SB3's callback base class
from src.environment.network_traffic_env import NetworkTrafficEnv

class TrafficLoggingCallback(BaseCallback):
    """
    Custom callback for tracking and logging traffic metrics at the end of each episode.
    """
    def __init__(self, controlled_nodes, q_table_dir="data/q_table", verbose=0):
        super(TrafficLoggingCallback, self).__init__(verbose)
        self.controlled_nodes = controlled_nodes
        self.q_table_dir = q_table_dir
        self.best_reward = float('-inf')
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if the episode has ended (dones is True in VecEnv)
        # DummyVecEnv wraps the "done" flags of multiple environments into an array; we retrieve the first one here
        if self.locals.get("dones")[0]:
            self.episode_count += 1
            
            # 1. Extract total reward for the current episode
            # SB3's Monitor wrapper automatically stores the latest episode's true reward and length into "infos"
            info = self.locals.get("infos")[0]
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
            else:
                # Fallback option: if the Monitor was not successfully triggered, retrieve directly from the environment
                episode_reward = self.training_env.get_attr("time_step")[0] # This is for structural demonstration only
                episode_reward = 0.0 # In practice, the Monitor's data takes precedence

            # 2. Extract throughput statistics for each intersection from the unwrapped real environment
            # VecEnv requires using env_method or get_attr to access properties of the underlying environment
            unwrapped_env = self.training_env.envs[0].unwrapped # Retrieve the first pure, unwrapped environment inside DummyVecEnv
            traffic_map = unwrapped_env.traffic_map

            # 3. Construct an informative log string
            status_msg = f"⏱️ Episode {self.episode_count} | Total Reward: {episode_reward:.1f} | Length: {episode_length} \n"
            stats_msg = "📊 Throughput -> "
            
            for node_id in self.controlled_nodes:
                intersection = traffic_map.intersections[node_id]
                stats_msg += f"{node_id}: {intersection.stats} "
            
            print("-" * 80)
            print(status_msg + stats_msg)

            # 4. Track and save the historical best model (Best Reward)
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                print(f"🔥 New Best Reward Broken: {self.best_reward:.1f}! Saving best model...")
                best_model_path = os.path.join(self.q_table_dir, "best_ppo_model")
                self.model.save(best_model_path)
            
            print("-" * 80)

            # 5. Reset throughput counters for each intersection to prepare for the next episode
            for node_id in self.controlled_nodes:
                traffic_map.intersections[node_id].reset_stats()

        return True


def train_agent(timesteps=500):
    controlled_nodes = ["Node_A", "Node_B", "Node_C"]
    
    env = NetworkTrafficEnv(controlled_nodes=controlled_nodes)
        
    node_positions = {
        "Start_Node" : 0.0,
        "Node_A" : 200.0,
        "Node_B" : 500.0,
        "Node_C" : 700.0
    }

    road_segments = [
        {"from": "Start_Node", "to": "Node_A", "speed_limit": 13.89},
        {"from": "Node_A",     "to": "Node_B", "speed_limit": 11.11},
        {"from": "Node_B",     "to": "Node_C", "speed_limit": 8.33}
    ]

    for node_id, x_coord in node_positions.items():
        env.traffic_map.add_intersection(node_id, x_coord, y = 0.0)

    for segment in road_segments:
        env.traffic_map.add_line(
            from_node_id = segment["from"],
            to_node_id = segment["to"],
            speed_limit = segment["speed_limit"]
        )

    check_env(env, warn=True)
    
    q_table_dir = "data/q_table"
    os.makedirs(q_table_dir, exist_ok=True)

    print(f"Controlled nodes in the road network: {controlled_nodes}")
    print("Starting model training using Stable-Baselines3...")

    # Initialize the model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./traffic_tensorboard/")

    # Instantiate our custom logging and tracking callback
    logging_callback = TrafficLoggingCallback(controlled_nodes=controlled_nodes, q_table_dir=q_table_dir)

    # Pass the callback function into the learn method
    model.learn(total_timesteps=timesteps, callback=logging_callback)

    # Save the final policy weights
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_traffic_model")
    print("Training complete! The final model has been saved.")

if __name__ == "__main__":
    train_agent(timesteps=500)