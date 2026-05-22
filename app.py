import streamlit as st
import pandas as pd
import numpy as np
import time
from environment import Crossing
from agent import QLearningAgent

# Set page configuration and layout
st.set_page_config(page_title="RL Traffic Signal Training Dashboard", layout="wide")
st.title("Reinforcement Learning - Single Intersection Traffic Signal Control Training Dashboard")

# ==========================================
# 1. Sidebar Configuration
# ==========================================
st.sidebar.header("Reinforcement Learning Hyperparameters")

# Allows user to adjust hyperparameters interactively
alpha = st.sidebar.slider(
    "Learning Rate (Alpha)", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.1, 
    step=0.05, 
    help="Determines the rate at which new information overrides existing knowledge."
)
gamma = st.sidebar.slider(
    "Discount Factor (Gamma)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.9, 
    step=0.05,
    help="Determines the importance of long-term future rewards relative to immediate rewards."
)
epsilon = st.sidebar.slider(
    "Exploration Rate (Epsilon)", 
    min_value=0.0, 
    max_value=0.5, 
    value=0.1, 
    step=0.05,
    help="Determines the probability that the agent takes a random action to explore the state space."
)
total_episodes = st.sidebar.number_input(
    "Total Episodes", 
    min_value=10, 
    max_value=2000, 
    value=300, 
    step=50
)

# Option to load or override existing models
load_existing = st.sidebar.checkbox("Load existing q_table.json", value=True)

start_training = st.sidebar.button("Start Agent Training", type="primary")

# ==========================================
# 2. Main Panel Layout
# ==========================================
col1, col2, col3 = st.columns(3)
with col1:
    status_card = st.empty()
    status_card.metric(label="Current Status", value="Not Started")
with col2:
    q_size_card = st.empty()
    q_size_card.metric(label="Q-Table State Count", value="0")
with col3:
    best_reward_card = st.empty()
    best_reward_card.metric(label="Best Reward This Run", value="-")

st.subheader("Real-time Training Reward Trend (Total Reward)")
# Placeholder chart component to dynamically append streaming values
chart_placeholder = st.empty()

# ==========================================
# 3. Core Training and Real-time UI Update Loop
# ==========================================
if start_training:
    # Initialize environment and Agent
    env = Crossing()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    
    # Load historical experiences if checked
    if load_existing:
        agent.load_q_table("q_table.json")
    
    status_card.metric(label="Current Status", value="Training...")
    q_size_card.metric(label="Q-Table State Count", value=str(len(agent.q_table)))
    
    # Create empty dataframe for live training graph plotting
    rewards_df = pd.DataFrame(columns=["Episode", "Total Reward"])
    chart_placeholder.line_chart(rewards_df.set_index("Episode"))
    
    best_reward = -float('inf')
    
    # Run reinforcement learning simulation loop
    for episode in range(int(total_episodes)):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
        
        # Track and update peak reward performance
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_q_table("best_q_table.json")
            best_reward_card.metric(label="Best Reward This Run", value=f"{best_reward:.1f}")
            
        # Append current episode details to the dataset
        new_row = pd.DataFrame({"Episode": [episode + 1], "Total Reward": [episode_reward]})
        rewards_df = pd.concat([rewards_df, new_row], ignore_index=True)
        
        # Update metrics and refresh charts every 5 episodes to reduce overhead
        if (episode + 1) % 5 == 0 or (episode + 1) == total_episodes:
            chart_placeholder.line_chart(rewards_df.set_index("Episode"))
            q_size_card.metric(label="Q-Table State Count", value=str(len(agent.q_table)))
            
    # Persist updated values to disk upon final completion
    agent.save_q_table("q_table.json")
    status_card.metric(label="Current Status", value="Training Completed")
    st.success(f"Successfully completed {total_episodes} training episodes. The latest Q-table has been successfully saved to local storage.")