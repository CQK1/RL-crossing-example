import time
from environment import Crossing

def test_fixed_policy():
    """
    Strategy for testing a fixed policy: 30 seconds of green for vehicles, then 30 seconds of green for pedestrians, and repeat.
    """

    env = Crossing()
    state = env.reset()
    print(f"Initial state -> Light: {state['light_state']}, Cars in line: {state['waiting_vehicles']}, Pedestrians in line: {state['waiting_pedestrians']}")

    done = False
    total_reward = 0
    while not done:
        if env.time_step == 30:
            action = 1 # switch light state after 30 time steps
        else:
            action = 0 # keep the current light state

        next_state, reward, done = env.step(action)
        total_reward += reward

        # --- print micro-snapshot ---
        light_desc = "🚗green/🚶red" if state['light_state'] == 1 else "🚗red/🚶green"
        act_desc = "switch" if action == 1 else "keep"
        
        print(f"Time: {env.time_step:02d}s | Current Light: {light_desc} | Action: {act_desc} "
              f"| Cars in line: {state['waiting_vehicles']} | Pedestrians in line: {state['waiting_pedestrians']} | Step Reward: {reward:.1f}")

        # update
        state = next_state
        
        # simulate real-time delay
        time.sleep(0.1)
    
    print(f"Total runtime: {env.time_step} seconds")
    print(f"Total reward (cumulative penalty): {total_reward:.1f} points")

if __name__ == "__main__":
    test_fixed_policy()