from environment.crossing import Crossing
from agent import QLearningAgent
import json

def train_agent(episodes=500):
    env = Crossing()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)

    agent.load_q_table("q_table.json")  # Load existing Q-table if available
    
    print("training...")

    print(f"Initial Q-table size: {len(agent.q_table)} entries")

    best_reward = float('-inf')

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 1. Agent chooses an action based on the current state
            action = agent.select_action(state)
            
            # 2. Environment receives the action and returns the next state, reward, and done flag
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. Agent learns: updates its Q-table
            agent.learn(state, action, reward, next_state, done)
            
            # 4. Update the current state to the next state for the next iteration
            state = next_state
        
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_q_table("q_table_best.json")  # Save the Q-table whenever we get a new best reward
            print(f"New best reward: {best_reward:.1f} points at episode {episode + 1}")

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward:.1f} | Q-Table Size: {len(agent.q_table)}")

    agent.save_q_table("q_table.json")
    print("Best reward achieved during training: {:.1f} points".format(best_reward))

if __name__ == "__main__":
    train_agent()

