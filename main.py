# main.py
import os
import json
from src.environment.network_traffic_env import NetworkTrafficEnv
from src.agents.agent import QLearningAgent

def train_agent(episodes=500):
    controlled_nodes = ["Node_A", "Node_B", "Node_C"]
    env = NetworkTrafficEnv(controlled_nodes=controlled_nodes)
        
    node_positions = {
        "Start_Node" : 0.0,
        "Node_A" : 200.0,
        "Node_B" : 500.0,
        "Node_C" : 700.0
    }

    road_segments = [
        {"from": "Start_Node", "to": "Node_A", "speed_limit": 13.89},  # 13.89 m/s (~50 km/h)
        {"from": "Node_A",     "to": "Node_B", "speed_limit": 11.11},  # 11.11 m/s (~40 km/h)
        {"from": "Node_B",     "to": "Node_C", "speed_limit": 8.33}   # 8.33 m/s  (~30 km/h)
    ]

    for node_id, x_coord in node_positions.items():
        env.traffic_map.add_intersection(node_id, x_coord, y = 0.0)

    for segment in road_segments:
        env.traffic_map.add_line(
            from_node_id = segment["from"],
            to_node_id = segment["to"],
            speed_limit = segment["speed_limit"]
        )

    agents = {}

    for node_id in controlled_nodes:
        agents[node_id] = QLearningAgent()

    q_table_dir = "data/q_table"
    os.makedirs(q_table_dir, exist_ok=True)

    for node_id, agent in agents.items():
        table_path = os.path.join(q_table_dir, f"q_table_{node_id}.json")
        agent.load_q_table(table_path)

 
    print(f"All nodes that are under control{controlled_nodes}")
    print("Start...")
    
    best_reward = float('-inf')

    for episode in range(episodes):
        # reset 现在返回两个值
        full_state, info = env.reset()
        
        done = False
        total_reward = 0
        
        while not done:
            action_dict = {}
            for node_id in controlled_nodes:
                action_dict[node_id] = agents[node_id].select_action(full_state[node_id])
            
            # step 现在返回五个值
            next_full_state, reward, terminated, truncated, info = env.step(action_dict)
            
            # 只要触发其中一个，本轮结束
            done = terminated or truncated 
            total_reward += reward
            
            # 智能体学习
            for node_id in controlled_nodes:
                agents[node_id].learn(
                    full_state[node_id], 
                    action_dict[node_id], 
                    reward, 
                    next_full_state[node_id], 
                    done
                )
            
            full_state = next_full_state
        
        # Save the best reward
        if total_reward > best_reward:
            best_reward = total_reward
            for node_id, agent in agents.items():
                best_path = os.path.join(q_table_dir, f"best_q_table_{node_id}.json")
                agent.save_q_table(best_path)
            print(f"Best reward: {best_reward:.1f}, happened at {episode + 1}")

        # 每 100 轮打印一次各个 Q-table 的探索状态大小
        if (episode + 1) % 100 == 0:
            status_msg = f"Episode {episode + 1}/{episodes} | Total reward: {total_reward:.1f} | Q-Table Size: "
            sizes = [f"{node_id}: {len(agent.q_table)}" for node_id, agent in agents.items()]
            print(status_msg + ", ".join(sizes))

    # 训练结束，持久化所有智能体的最终决策表
    for node_id, agent in agents.items():
        final_path = os.path.join(q_table_dir, f"q_table_{node_id}.json")
        agent.save_q_table(final_path)
        
    print(f"Finished. Best rewards: {best_reward:.1f}")

if __name__ == "__main__":
    train_agent(episodes=500)