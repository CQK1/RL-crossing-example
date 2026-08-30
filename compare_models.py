import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environment.network_traffic_env import NetworkTrafficEnv
from src.agents.agent import QLearningAgent

# Map state vector to dictionary format for your existing QLearningAgent
def array_to_dict(state_array):
    # Align dictionary keys strictly with observation layout
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

def evaluate_q_learning(episodes=3, q_table_path="data/q_table/best_q_table_Node_A.json"):
    # 1. Instantiate the environment
    env = NetworkTrafficEnv(decision_interval=240)
    
    # 2. Instantiate your original Q-Learning agent
    agent = QLearningAgent()
    
    # 3. Load pre-trained Q-table from your legacy runs
    if os.path.exists(q_table_path):
        agent.load_q_table(q_table_path)
    else:
        # Fallback to general q-table file if node-specific one is not found
        fallback_path = "data/q_table/q_table.json"
        if os.path.exists(fallback_path):
            agent.load_q_table(fallback_path)
        else:
            print(f"Warning: No pre-trained Q-table found at {q_table_path}. Running with empty Q-table.")
            
    # Set exploration rate to 0 to evaluate pure learned policy (greedy actions only)
    agent.epsilon = 0.0
    
    total_costs = []
    print("\n--- Evaluating Legacy Q-Learning (Pre-trained) ---")
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            dict_state = array_to_dict(state)
            
            # Select greedy action from the loaded Q-table
            action = agent.select_action(dict_state)
            next_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            
            state = next_state
            ep_reward += reward
            
        cost = abs(ep_reward)
        total_costs.append(cost)
        print(f"Episode {ep + 1} Cost (Total Delay): {cost:,.1f}")
        
    return float(np.mean(total_costs))

def evaluate_ppo(episodes=3, model_path="models/best_mayor_magrath_ppo"):
    def make_env():
        return NetworkTrafficEnv(decision_interval=240)
    
    vec_env = DummyVecEnv([make_env])
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)
    
    # Load your newly trained PPO model
    model = PPO.load(model_path, env=env)
    
    total_costs = []
    print("\n--- Evaluating SB3 PPO (Pre-trained) ---")
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            ep_reward += reward[0]
            
        cost = abs(ep_reward)
        total_costs.append(cost)
        print(f"Episode {ep + 1} Cost (Total Delay): {cost:,.1f}")
        
    return float(np.mean(total_costs))

def run_benchmark():
    # Evaluate both trained policies over identical test runs
    ql_mean_cost = evaluate_q_learning(episodes=3)
    ppo_mean_cost = evaluate_ppo(episodes=3)
    
    print("\n" + "=" * 55)
    print("           TRAFFIC COST BENCHMARK RESULTS")
    print("=" * 55)
    print(f"Legacy Q-Learning Average Cost: {ql_mean_cost:,.1f}")
    print(f"SB3 PPO Average Cost:           {ppo_mean_cost:,.1f}")
    print("-" * 55)
    
    diff = abs(ql_mean_cost - ppo_mean_cost)
    if ppo_mean_cost < ql_mean_cost:
        pct = (diff / ql_mean_cost) * 100.0
        print(f"Result: PPO performed better (Cost reduced by {pct:.2f}%).")
    elif ql_mean_cost < ppo_mean_cost:
        pct = (diff / ppo_mean_cost) * 100.0
        print(f"Result: Q-Learning performed better (Cost reduced by {pct:.2f}%).")
    else:
        print("Result: Both algorithms yielded identical performance.")
    print("=" * 55)

if __name__ == "__main__":
    run_benchmark()