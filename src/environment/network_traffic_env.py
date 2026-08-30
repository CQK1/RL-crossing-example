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
    def __init__(self, decision_interval: int = 10, data_file_path=None):
        super(NetworkTrafficEnv, self).__init__()

        self.dt = 1.0
        self.decision_interval = decision_interval
        self.time_step = 0
        self.yellow_duration = 3

        # 1. Data & Mathematical Engine
        if data_file_path is None:
            data_file_path = "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx"
            if not os.path.exists(data_file_path):
                data_file_path = os.path.join(
                    "data",
                    "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx",
                )

        self.data_reader = TrafficDataReader(data_file_path)
        self.traffic_data = self.data_reader.load_data()

        self.poisson_engine = InhomogeneousPoissonProcess(self.traffic_data, cyclic=True)
        self.traffic_generator = TrafficGenerator(rate_model=self.poisson_engine)

        # 2. Physics & Map Topology
        self.traffic_map = TrafficMap()
        self.controlled_nodes = ["Mayor_Magrath"]
        self.num_nodes = 1

        # Action space: 1 intersection, 4 discrete actions
        self.action_space = spaces.MultiDiscrete([4])

        # State space: 10 features
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(10,), dtype=np.float32)

        # Create the central intersection
        self.traffic_map.add_intersection("Mayor_Magrath", 0.0, 0.0)

        # Define 4 spawn nodes and 4 exit nodes
        spawns = {
            "North": (0, 200),
            "South": (0, -200),
            "East": (200, 0),
            "West": (-200, 0),
        }

        for dir_name, (x, y) in spawns.items():
            spawn_node = f"{dir_name}_Spawn"
            exit_node = f"{dir_name}_Exit"

            self.traffic_map.add_intersection(spawn_node, x, y)
            self.traffic_map.add_intersection(exit_node, x * 2, y * 2)

            # Incoming lanes: 2 per direction
            self.traffic_map.add_line(
                spawn_node, "Mayor_Magrath",
                speed_limit=15.0, lane_type="straight_right"
            )
            self.traffic_map.add_line(
                spawn_node, "Mayor_Magrath",
                speed_limit=15.0, lane_type="left_uturn"
            )

            # Outgoing lane: 1 per direction
            self.traffic_map.add_line(
                "Mayor_Magrath", exit_node,
                speed_limit=15.0, lane_type="all"
            )

    def get_state(self):
        """
        Build the observation vector from the current intersection state.

        Returns:
            np.ndarray of shape (10,)
        """
        intersection = self.traffic_map.intersections["Mayor_Magrath"]

        ns_straight_count, ns_left_count = 0, 0
        ew_straight_count, ew_left_count = 0, 0

        ns_straight_max_wait = 0.0
        ns_left_max_wait = 0.0
        ew_straight_max_wait = 0.0
        ew_left_max_wait = 0.0

        for lane in intersection.incoming_lanes:
            is_ns_lane = lane.approach_direction in ["North", "South"]

            for car in lane.vehicles:
                if car.speed <= 0.1:
                    is_left_turn = str(car.destination).lower().endswith("_left")
                    wait_time = getattr(car, "waiting_time", 0.0)

                    if is_ns_lane:
                        if is_left_turn:
                            ns_left_count += 1
                            ns_left_max_wait = max(ns_left_max_wait, wait_time)
                        else:
                            ns_straight_count += 1
                            ns_straight_max_wait = max(ns_straight_max_wait, wait_time)
                    else:
                        if is_left_turn:
                            ew_left_count += 1
                            ew_left_max_wait = max(ew_left_max_wait, wait_time)
                        else:
                            ew_straight_count += 1
                            ew_straight_max_wait = max(ew_straight_max_wait, wait_time)

            # Include virtual queue overflow
            if lane.virtual_queue_count > 0:
                if is_ns_lane:
                    if lane.lane_type == "left_uturn":
                        ns_left_count += lane.virtual_queue_count
                        ns_left_max_wait = max(ns_left_max_wait, 10.0)
                    else:
                        ns_straight_count += lane.virtual_queue_count
                        ns_straight_max_wait = max(ns_straight_max_wait, 10.0)
                else:
                    if lane.lane_type == "left_uturn":
                        ew_left_count += lane.virtual_queue_count
                        ew_left_max_wait = max(ew_left_max_wait, 10.0)
                    else:
                        ew_straight_count += lane.virtual_queue_count
                        ew_straight_max_wait = max(ew_straight_max_wait, 10.0)

        phase_timer_normalized = min(intersection.phase_timer / 60.0, 1.0)

        max_wait_cap = 180.0
        ns_straight_max_wait_norm = min(ns_straight_max_wait / max_wait_cap, 1.0)
        ns_left_max_wait_norm = min(ns_left_max_wait / max_wait_cap, 1.0)
        ew_straight_max_wait_norm = min(ew_straight_max_wait / max_wait_cap, 1.0)
        ew_left_max_wait_norm = min(ew_left_max_wait / max_wait_cap, 1.0)

        max_capacity = 40.0
        ns_s_norm = min(ns_straight_count / max_capacity, 2.0)
        ns_l_norm = min(ns_left_count / max_capacity, 2.0)
        ew_s_norm = min(ew_straight_count / max_capacity, 2.0)
        ew_l_norm = min(ew_left_count / max_capacity, 2.0)

        obs = np.array([
            intersection.current_phase_index / 3.0,
            phase_timer_normalized,
            ns_s_norm,
            ns_l_norm,
            ew_s_norm,
            ew_l_norm,
            ns_straight_max_wait_norm,
            ns_left_max_wait_norm,
            ew_straight_max_wait_norm,
            ew_left_max_wait_norm,
        ], dtype=np.float32)

        return obs

    def calculate_reward(self):
        """
        Penalty based on cumulative waiting time, including virtual queue overflow.
        """
        total_penalty = 0.0
        intersection = self.traffic_map.intersections["Mayor_Magrath"]

        for lane in intersection.incoming_lanes:
            for car in lane.vehicles:
                if car.speed <= 0.1:
                    total_penalty += car.waiting_time

            if lane.virtual_queue_count > 0:
                max_physical_wait = 0.0
                for car in lane.vehicles:
                    if car.speed <= 0.1:
                        max_physical_wait = max(max_physical_wait, car.waiting_time)

                estimated_wait = max(max_physical_wait + self.dt, self.dt)
                total_penalty += lane.virtual_queue_count * estimated_wait

        return float(-total_penalty)

    def step(self, action_array):
        """
        Advance the simulation by one decision interval.
        """
        action = int(action_array[0])
        total_reward = 0.0

        intersection = self.traffic_map.intersections["Mayor_Magrath"]
        current_phase_index = intersection.current_phase_index
        is_phase_change = action != current_phase_index

        yellow_duration = self.yellow_duration if is_phase_change else 0

        for step_idx in range(self.decision_interval):
            # 1. Apply signal control
            if step_idx < yellow_duration:
                intersection.apply_action(-1, dt=self.dt)
            else:
                intersection.apply_action(action, dt=self.dt)

            # 2. Generate vehicles directly for each incoming lane
            for lane in self.traffic_map.lanes:
                if lane.to_node_id != "Mayor_Magrath":
                    continue

                new_cars = self.traffic_generator.generate_for_lane(
                    int(self.time_step),
                    lane.approach_direction,
                    lane.lane_type
                )

                for car in new_cars:
                    if len(lane.vehicles) < 40:
                        lane.vehicles.append(car)
                    else:
                        lane.virtual_queue_count += 1

            # 3. Advance physics simulation
            self.traffic_map.step(dt=self.dt)

            # 4. Accumulate reward
            total_reward += self.calculate_reward()

            # 5. Advance global clock
            self.time_step += 1

            if self.time_step >= 86400:
                break

        observation = self.get_state()
        terminated = False
        truncated = self.time_step >= 86400
        info = {}

        return observation, total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the start of a new day.
        """
        super().reset(seed=seed)

        self.time_step = 0

        for lane in self.traffic_map.lanes:
            lane.vehicles.clear()
            lane.virtual_queue_count = 0

        for inter in self.traffic_map.intersections.values():
            inter.current_phase_index = 0
            inter.current_phase = inter.phases[0]
            inter.phase_timer = 0.0
            inter.reset_stats()

        return self.get_state(), {}