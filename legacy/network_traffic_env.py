import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.environment.map import TrafficMap
from src.generators.traffic_generator import TrafficGenerator
from src.data_reader import TrafficDataReader

try:
    from src.generators.poisson_process import InhomogeneousPoissonProcess
except ImportError:
    from src.generators.traffic_generator import InhomogeneousPoissonProcess

class NetworkTrafficEnv(gym.Env):
    def __init__(self, data_file_path=None):
        super(NetworkTrafficEnv, self).__init__()
        self.dt = 1.0          
        self.time_step = 0     

        if data_file_path is None:
            data_file_path = "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx"
            if not os.path.exists(data_file_path):
                data_file_path = os.path.join("data", "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx")
        
        self.data_reader = TrafficDataReader(data_file_path)
        self.traffic_data = self.data_reader.load_data()
        
        self.poisson_engine = InhomogeneousPoissonProcess(self.traffic_data, cyclic=True)
        self.traffic_generator = TrafficGenerator(rate_model=self.poisson_engine)

        self.traffic_map = TrafficMap()
        self.controlled_nodes = ["Mayor_Magrath"]
        
        self.action_space = spaces.MultiDiscrete([4])
        self.observation_space = spaces.Box(low=0, high=500, shape=(5,), dtype=np.float32)

        self.traffic_map.add_intersection("Mayor_Magrath", 0.0, 0.0)
        
        spawns = {"North": (0, 200), "South": (0, -200), "East": (200, 0), "West": (-200, 0)}
        for dir_name, (x, y) in spawns.items():
            spawn_node = f"{dir_name}_Spawn"
            exit_node = f"{dir_name}_Exit"
            self.traffic_map.add_intersection(spawn_node, x, y)
            self.traffic_map.add_intersection(exit_node, x * 2, y * 2) 
            self.traffic_map.add_line(spawn_node, "Mayor_Magrath", speed_limit=15.0)
            self.traffic_map.add_line("Mayor_Magrath", exit_node, speed_limit=15.0)

    def get_state(self):
        obs = []
        intersection = self.traffic_map.intersections["Mayor_Magrath"]
        ns_straight_count, ns_left_count = 0, 0
        ew_straight_count, ew_left_count = 0, 0

        for lane in intersection.incoming_lanes:
            is_ns_lane = lane.approach_direction in ["North", "South"]
            for car in lane.vehicles:
                if car.speed <= 0.1:
                    is_left_turn = str(car.destination).endswith("_left")
                    if is_ns_lane:
                        if is_left_turn: ns_left_count += 1
                        else: ns_straight_count += 1
                    else:
                        if is_left_turn: ew_left_count += 1
                        else: ew_straight_count += 1

        obs.extend([
            intersection.current_phase_index,
            ns_straight_count, ns_left_count,
            ew_straight_count, ew_left_count
        ])
        return np.array(obs, dtype=np.float32)

    def calculate_reward(self):
        queue_length = 0
        intersection = self.traffic_map.intersections["Mayor_Magrath"]
        for lane in intersection.incoming_lanes:
            for car in lane.vehicles:
                if car.speed <= 0.1:
                    queue_length += 1
        
        # 强制锁死：哪怕堵满1000辆车，单步最多只扣 200 分！彻底消灭天文数字
        penalty = min(queue_length, 200)
        return float(-penalty)

    def step(self, action_array):
        action = action_array[0]
        self.traffic_map.intersections["Mayor_Magrath"].apply_action(action, dt=self.dt)

        new_entities_dict = self.traffic_generator.generate_entities(float(self.time_step))
        
        for lane in self.traffic_map.lanes:
            if lane.to_node_id == "Mayor_Magrath":
                direction = lane.approach_direction
                entities_to_add = new_entities_dict.get(direction, [])
                if entities_to_add:
                    if len(lane.vehicles) < 40:
                        lane.vehicles.extend(entities_to_add)
        
        for direction in new_entities_dict.keys():
            new_entities_dict[direction] = []

        self.traffic_map.step(dt=self.dt)

        observation = self.get_state()
        reward = self.calculate_reward()

        self.time_step += 1
        terminated = False 
        truncated = self.time_step >= 86400 
        
        # 【终极防丢输出】直接在环境里打印，不走 SB3 回调，SB3 的 reset 就拿我们没办法了！
        if terminated or truncated:
            stats = self.traffic_map.intersections["Mayor_Magrath"].stats
            print(f"\n{'='*60}")
            print(f"🏁 [ENV INTERNAL] Episode Done! 86400 seconds passed.")
            print(f"📊 Real Throughput: {stats}")
            print(f"{'='*60}\n")

        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        for lane in self.traffic_map.lanes:
            lane.vehicles.clear() 
        for inter in self.traffic_map.intersections.values():
            inter.current_phase_index = 0
            inter.current_phase = inter.phases[0]
            inter.phase_timer = 0.0
            inter.reset_stats()
            
        return self.get_state(), {}