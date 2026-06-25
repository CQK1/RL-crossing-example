import random
import json
import os
import ast

class QLearningAgent:
    def __init__(self, action_space=[0, 1], alpha=0.1, gamma=0.9, epsilon=0.1):
        self.action_space = action_space
        self.alpha = alpha       # Learning rate: how much the agent updates its knowledge based on new information (0 means no learning, 1 means only the latest info matters)
        self.gamma = gamma       # Discount factor: how much the agent values future rewards (0 means only immediate rewards matter, 1 means all future rewards are equally important)
        self.epsilon = epsilon   # Exploration rate: the probability of taking a random action (helps the agent explore new strategies)
        self.q_table = {}        # Q-table: the agent's memory, storing the expected rewards for each state-action pair

    def _get_state_key(self, state):
        # 将 NumPy 数组转为整数标量，并提供默认值防止找不到键
        vehicles = int(state['waiting_vehicles'][0]) if hasattr(state['waiting_vehicles'], '__len__') else int(state['waiting_vehicles'])
        peds = int(state.get('waiting_pedestrians', 0))
        return (state['light_state'], vehicles, peds)
    
    def _ensure_state_in_q_table(self, state):
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            # Initialize Q-values for all actions to 0 for this new state
            self.q_table[state_key] = {action: 0.0 for action in self.action_space}

    def select_action(self, state):
        self._ensure_state_in_q_table(state)
        state_key = self._get_state_key(state)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.choice(self.action_space)
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            q_values = self.q_table[state_key]
            max_q = max(q_values)
            best_actions = [action for action, q_value in self.q_table[state_key].items() if q_value == max_q]

            if not best_actions:
                return random.choice(self.action_space)  # If no best action, choose randomly
            
            return random.choice(best_actions)  # If multiple actions have the same max Q-value, choose randomly among them
        
    def learn(self, state, action, reward, next_state, done):
        self._ensure_state_in_q_table(state)
        self._ensure_state_in_q_table(next_state)

        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # Q-learning update rule
        if done:
            target = reward
        else:
            best_next_q = max(self.q_table[next_state_key].values())
            target = reward + self.gamma * best_next_q
        
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])

    def save_q_table(self, filename="q_table.json"):
        exportable_q_table = {str(state): values for state, values in self.q_table.items()}
        with open(filename, "w") as f:
            json.dump(exportable_q_table, f)
        print(f"Q-table has been saved to {filename}")

    def load_q_table(self, filename="q_table.json"):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                loaded_q_table = json.load(f)
                self.q_table = {
                    ast.literal_eval(state_str): {int(act_str): q_val for act_str, q_val in q_values.items()}
                    for state_str, q_values in loaded_q_table.items()
            }
            print(f"Q-table has been loaded from {filename}")
        else:
            print(f"No Q-table file found at {filename}. Starting with an empty Q-table.")