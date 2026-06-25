import streamlit as st
import pandas as pd
import os
import time

# Import the refactored multi-intersection environment and agents
from src.environment.network_traffic_env import NetworkTrafficEnv
from src.agents.agent import QLearningAgent

# Set page configuration and layout
st.set_page_config(page_title="RL Traffic Control Dashboard", layout="wide")
st.title("Multi-Agent Traffic Control Dashboard (Lethbridge Corridor)")

# ==========================================
# Helper Function for Visual Map
# ==========================================
def generate_map_html(stats_a, stats_b, stats_c):
    """Generates a visual schematic of the traffic corridor with live throughput data."""
    # Strict HTML without indentation or blank lines to prevent Streamlit 
    # from parsing it as a Markdown code block.
    return f"""<div style="display: flex; align-items: center; justify-content: space-between; font-family: sans-serif; margin: 20px 0; background: #1e1e1e; padding: 20px; border-radius: 12px; color: white;">
<div style="padding: 15px; background: #333; border-radius: 8px; text-align: center; min-width: 120px;">
<strong style="color: #bbb; font-size: 16px;">Start Node</strong><br>
<div style="margin-top: 10px; font-size: 14px;">🚗 Vehicle Spawn<br><span style="color: #888;">(Source)</span></div>
</div>
<div style="font-size: 24px; color: #666;"> ➡ </div>
<div style="padding: 15px; border: 2px solid #4CAF50; border-radius: 10px; background: #233323; min-width: 160px; text-align: center;">
<strong style="color: #4CAF50; font-size: 18px;">Node A</strong><br>
<div style="margin-top: 10px; font-size: 15px; text-align: left; padding-left: 20px;">
<span style="color: #4CAF50;">⬆️ Straight:</span> <b>{stats_a['straight']}</b><br>
<span style="color: #ff5252;">⬅️ Left Turn:</span> <b>{stats_a['left']}</b>
</div>
</div>
<div style="font-size: 24px; color: #666;"> ➡ </div>
<div style="padding: 15px; border: 2px solid #2196F3; border-radius: 10px; background: #202b38; min-width: 160px; text-align: center;">
<strong style="color: #2196F3; font-size: 18px;">Node B</strong><br>
<div style="margin-top: 10px; font-size: 15px; text-align: left; padding-left: 20px;">
<span style="color: #2196F3;">⬆️ Straight:</span> <b>{stats_b['straight']}</b><br>
<span style="color: #ff5252;">⬅️ Left Turn:</span> <b>{stats_b['left']}</b>
</div>
</div>
<div style="font-size: 24px; color: #666;"> ➡ </div>
<div style="padding: 15px; border: 2px solid #FF9800; border-radius: 10px; background: #382c1a; min-width: 160px; text-align: center;">
<strong style="color: #FF9800; font-size: 18px;">Node C</strong><br>
<div style="margin-top: 10px; font-size: 15px; text-align: left; padding-left: 20px;">
<span style="color: #FF9800;">⬆️ Straight:</span> <b>{stats_c['straight']}</b><br>
<span style="color: #ff5252;">⬅️ Left Turn:</span> <b>{stats_c['left']}</b>
</div>
</div>
</div>"""

# ==========================================
# 1. Sidebar Configuration
# ==========================================
st.sidebar.header("Hyperparameters")

alpha = st.sidebar.slider("Learning Rate (Alpha)", 0.01, 1.0, 0.1, 0.05, 
                          help="Determines the rate at which new information overrides existing knowledge.")
gamma = st.sidebar.slider("Discount Factor (Gamma)", 0.0, 1.0, 0.9, 0.05,
                          help="Determines the importance of long-term future rewards relative to immediate rewards.")
epsilon = st.sidebar.slider("Exploration Rate (Epsilon)", 0.0, 0.5, 0.1, 0.05,
                          help="Determines the probability that the agent takes a random action to explore new states.")
total_episodes = st.sidebar.number_input("Total Episodes", 10, 2000, 500, 50)

load_existing = st.sidebar.checkbox("Load existing Q-Table (Historical Data)", value=True)
start_training = st.sidebar.button("Start Cooperative Training", type="primary")

# ==========================================
# 2. Main Panel Layout
# ==========================================
st.subheader("Real-time Status")
status_col, reward_col = st.columns(2)
with status_col:
    status_card = st.empty()
    status_card.metric(label="Current Status", value="Not Started")
with reward_col:
    best_reward_card = st.empty()
    best_reward_card.metric(label="Best Reward (Current Run)", value="-")

st.markdown("---")
st.subheader("Live Map Visualization & Traffic Routing")
st.caption("Visualizing the Origin-Destination (O-D) routing engine in action. Left turns act as sinks.")
map_placeholder = st.empty()
# Show initial empty map
map_placeholder.markdown(generate_map_html({"straight": 0, "left": 0}, {"straight": 0, "left": 0}, {"straight": 0, "left": 0}), unsafe_allow_html=True)

st.markdown("---")
st.subheader("Agent Learning Progress")
col_a, col_b, col_c = st.columns(3)
with col_a:
    q_size_a = st.empty()
    q_size_a.metric(label="Node_A Q-Table Size", value="0")
with col_b:
    q_size_b = st.empty()
    q_size_b.metric(label="Node_B Q-Table Size", value="0")
with col_c:
    q_size_c = st.empty()
    q_size_c.metric(label="Node_C Q-Table Size", value="0")

st.markdown("---")
st.subheader("Global Reward Trend")
chart_placeholder = st.empty()

# ==========================================
# 3. Core Training & UI Update Loop
# ==========================================
if start_training:
    # Define controlled nodes
    controlled_nodes = ["Node_A", "Node_B", "Node_C"]
    
    # Initialize the latest environment
    env = NetworkTrafficEnv(controlled_nodes=controlled_nodes)
    
    # Construct the Lethbridge-style map
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
    
    # Instantiate independent agents for each intersection
    agents = {node_id: QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon) for node_id in controlled_nodes}
    
    q_table_dir = "data/q_table"
    os.makedirs(q_table_dir, exist_ok=True)
    
    # Load existing experience tables
    if load_existing:
        for node_id, agent in agents.items():
            table_path = os.path.join(q_table_dir, f"q_table_{node_id}.json")
            agent.load_q_table(table_path)
            
    status_card.metric(label="Current Status", value="Training in progress...")
    
    # Prepare chart data
    rewards_df = pd.DataFrame(columns=["Episode", "Total Reward"])
    chart_placeholder.line_chart(rewards_df.set_index("Episode"))
    
    best_reward = -float('inf')
    
    # Start training loop
    for episode in range(int(total_episodes)):
        full_state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action_dict = {}
            for node_id in controlled_nodes:
                action_dict[node_id] = agents[node_id].select_action(full_state[node_id])
            
            # Environment step (Gymnasium 5-value unpacking)
            next_full_state, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated 
            episode_reward += reward
            
            # Independent learning for each agent
            for node_id in controlled_nodes:
                agents[node_id].learn(
                    full_state[node_id], 
                    action_dict[node_id], 
                    reward, 
                    next_full_state[node_id], 
                    done
                )
            full_state = next_full_state
            
        # Record best score and save optimal states
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_reward_card.metric(label="Best Reward (Current Run)", value=f"{best_reward:.1f}")
            for node_id, agent in agents.items():
                best_path = os.path.join(q_table_dir, f"best_q_table_{node_id}.json")
                agent.save_q_table(best_path)
                
        # Update chart data
        new_row = pd.DataFrame({"Episode": [episode + 1], "Total Reward": [episode_reward]})
        rewards_df = pd.concat([rewards_df, new_row], ignore_index=True)
        
        # Update UI every 5 episodes to reduce rendering overhead
        if (episode + 1) % 5 == 0 or (episode + 1) == total_episodes:
            chart_placeholder.line_chart(rewards_df.set_index("Episode"))
            
            # Update Q-Table sizes for each node
            q_size_a.metric(label="Node_A Q-Table Size", value=str(len(agents["Node_A"].q_table)))
            q_size_b.metric(label="Node_B Q-Table Size", value=str(len(agents["Node_B"].q_table)))
            q_size_c.metric(label="Node_C Q-Table Size", value=str(len(agents["Node_C"].q_table)))
            
            # Update Map Visualization with live throughput
            stats_a = env.traffic_map.intersections['Node_A'].stats
            stats_b = env.traffic_map.intersections['Node_B'].stats
            stats_c = env.traffic_map.intersections['Node_C'].stats
            
            map_html = generate_map_html(stats_a, stats_b, stats_c)
            map_placeholder.markdown(map_html, unsafe_allow_html=True)
            
        # Reset counters at the end of each episode to track true throughput per round
        for node_id in controlled_nodes:
            env.traffic_map.intersections[node_id].reset_stats()
            
    # Save finally upon completion
    for node_id, agent in agents.items():
        final_path = os.path.join(q_table_dir, f"q_table_{node_id}.json")
        agent.save_q_table(final_path)
        
    status_card.metric(label="Current Status", value="Training Completed!")
    st.success(f"Successfully completed {total_episodes} episodes. Multi-agent experience data has been saved.")