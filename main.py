# main.py
import os
import json
from src.environment.network_traffic_env import NetworkTrafficEnv
from src.agents.agent import QLearningAgent

def train_agent(episodes=500):
    target_node = "Node_A"
    env = NetworkTrafficEnv(target_id=target_node)
    
    # ====== 核心：在这里（外部）动态配置路网拓扑 ======
    env.traffic_map.add_intersection(target_node, x=150.0, y=0.0)
    env.traffic_map.add_intersection("Start_Node", x=0.0, y=0.0)
    env.traffic_map.add_line("Start_Node", target_node, speed_limit=13.89)
    # ==================================================

    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)

    table_path = "data/q_table/q_table.json"
    best_table_path = "data/q_table/best_q_table.json"
    agent.load_q_table(table_path) 
    
    print("Training started with continuous Network Environment...")

    best_reward = float('-inf')

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            
            # 包装成多路口支持的 action_dict 传给环境
            action_dict = {target_node: action}
            
            next_state, reward, done = env.step(action_dict)
            total_reward += reward
            
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        
        if total_reward > best_reward:
            best_reward = total_reward
            os.makedirs(os.path.dirname(best_table_path), exist_ok=True)
            agent.save_q_table(best_table_path)
            print(f"New best reward: {best_reward:.1f} points at episode {episode + 1}")

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward:.1f} | Q-Table Size: {len(agent.q_table)}")

    agent.save_q_table(table_path)
    print(f"Training complete. Best reward achieved: {best_reward:.1f} points")

if __name__ == "__main__":
    train_agent(episodes=500)