# main.py
import os
import json
from src.environment.network_traffic_env import NetworkTrafficEnv
from src.agents.agent import QLearningAgent

def train_agent(episodes=500):
    target_node = "Node_A"
    env = NetworkTrafficEnv(target_id=target_node)
        
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

    controlled_nodes = ["Node_A", "Node_B", "Node_C"]
    agents = {}

    q_table_dir = "data/q_table"
    os.makedirs(q_table_dir, exist_ok=True)

    for node_id, agent in agents.items():
        table_path = os.path.join(q_table_dir, f"q_table_{node_id}.json")
        agent.load_q_table(table_path)

 
    print(f"All nodes that are under control{controlled_nodes}")
    print("Start...")
    
    best_reward = float('-inf')

    for episode in range(episodes):
        env.reset()
        done = False
        total_reward = 0
        
        # Get initial states of all intersections
        full_state = env.get_state()
        
        # 为每个 Agent 提取并构建适合其输入格式的局部状态
        local_states = {}
        for node_id in controlled_nodes:
            local_states[node_id] = {
                "light_state": full_state[node_id]["light_state"],
                "waiting_vehicles": full_state[node_id]["waiting_vehicles"],
                "waiting_pedestrians": 0  
            }
        
        while not done:
            # 1. 每个 Agent 根据各自路口的局部状态，独立决定自己的红绿灯动作
            action_dict = {}
            for node_id in controlled_nodes:
                action_dict[node_id] = agents[node_id].select_action(local_states[node_id])
            
            # 2. 将动作字典传入环境，执行联合物理仿真步进
            _, reward, done = env.step(action_dict)
            total_reward += reward
            
            # 3. 获取更新后的全局状态，并解构为各个路口的下一步局部状态
            next_full_state = env.get_state()
            next_local_states = {}
            for node_id in controlled_nodes:
                next_local_states[node_id] = {
                    "light_state": next_full_state[node_id]["light_state"],
                    "waiting_vehicles": next_full_state[node_id]["waiting_vehicles"],
                    "waiting_pedestrians": 0
                }
                
                # 4. every agent updates its own Q-table
                # 它们共享同一个全局系统延误奖励（cooperative reward），从而自发学习绿波带协调
                agents[node_id].learn(
                    local_states[node_id], 
                    action_dict[node_id], 
                    reward, 
                    next_local_states[node_id], 
                    done
                )
            
            # updates states
            local_states = next_local_states
        
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