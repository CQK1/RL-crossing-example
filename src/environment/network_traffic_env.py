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

        self.dt = 1.0                  # Physics step size: 1 second
        self.decision_interval = decision_interval
        self.time_step = 0             # Global simulation clock in seconds
        self.yellow_duration = 3       # Combined yellow + all-red clearance time when switching phases

        # ---------------------------------------------------------
        # 1. Data & Mathematical Engine
        # ---------------------------------------------------------
        if data_file_path is None:
            data_file_path = "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx"
            if not os.path.exists(data_file_path):
                data_file_path = os.path.join("data", "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx")

        self.data_reader = TrafficDataReader(data_file_path)
        self.traffic_data = self.data_reader.load_data()

        self.poisson_engine = InhomogeneousPoissonProcess(self.traffic_data, cyclic=True)
        self.traffic_generator = TrafficGenerator(rate_model=self.poisson_engine)

        # ---------------------------------------------------------
        # 2. Physics & Map Topology
        # ---------------------------------------------------------
        self.traffic_map = TrafficMap()
        self.controlled_nodes = ["Mayor_Magrath"]
        self.num_nodes = 1

        # Action space: 1 intersection, 4 discrete actions
        # 0 = NS_Straight, 1 = NS_Left, 2 = EW_Straight, 3 = EW_Left
        self.action_space = spaces.MultiDiscrete([4])

        # State space: 6 features
        # [current_phase_id, phase_timer_normalized, ns_straight_queue, ns_left_queue, ew_straight_queue, ew_left_queue]
        self.observation_space = spaces.Box(low=0, high=500, shape=(6,), dtype=np.float32)

        # Create the central intersection
        self.traffic_map.add_intersection("Mayor_Magrath", 0.0, 0.0)

        # Define 4 spawn nodes and 4 exit nodes around the central intersection
        spawns = {
            "North": (0, 200),
            "South": (0, -200),
            "East": (200, 0),
            "West": (-200, 0)
        }

        for dir_name, (x, y) in spawns.items():
            spawn_node = f"{dir_name}_Spawn"
            exit_node = f"{dir_name}_Exit"

            self.traffic_map.add_intersection(spawn_node, x, y)
            self.traffic_map.add_intersection(exit_node, x * 2, y * 2)

            # Incoming lanes: 2 per direction (straight+right, left+u-turn)
            self.traffic_map.add_line(
                spawn_node, "Mayor_Magrath",
                speed_limit=15.0,
                lane_type="straight_right"
            )
            self.traffic_map.add_line(
                spawn_node, "Mayor_Magrath",
                speed_limit=15.0,
                lane_type="left_uturn"
            )

            # Outgoing lane: 1 per direction
            self.traffic_map.add_line(
                "Mayor_Magrath", exit_node,
                speed_limit=15.0,
                lane_type="all"
            )

    def get_state(self):
        """
        Build the observation vector from the current intersection state.

        Returns:
            np.ndarray of shape (6,) containing:
            [phase_id, phase_timer_normalized, ns_straight_queue, ns_left_queue, ew_straight_queue, ew_left_queue]
        """
        intersection = self.traffic_map.intersections["Mayor_Magrath"]
        ns_straight_count, ns_left_count = 0, 0
        ew_straight_count, ew_left_count = 0, 0

        for lane in intersection.incoming_lanes:
            is_ns_lane = lane.approach_direction in ["North", "South"]

            # Count queued (stopped) vehicles in this lane
            for car in lane.vehicles:
                if car.speed <= 0.1:
                    # Determine if this vehicle is a left-turn movement
                    is_left_turn = str(car.destination).lower().endswith("_left")

                    if is_ns_lane:
                        if is_left_turn:
                            ns_left_count += 1
                        else:
                            ns_straight_count += 1
                    else:
                        if is_left_turn:
                            ew_left_count += 1
                        else:
                            ew_straight_count += 1

        # Normalize phase timer to [0, 1] range (max_green = 60s)
        phase_timer_normalized = min(intersection.phase_timer / 60.0, 1.0)

        obs = np.array([
            intersection.current_phase_index,
            phase_timer_normalized,
            ns_straight_count,
            ns_left_count,
            ew_straight_count,
            ew_left_count
        ], dtype=np.float32)

        return obs

    def calculate_reward(self):
        """
        Reward is a linear penalty: -1 point per stopped vehicle per second.
        """
        stopped_vehicles = 0
        intersection = self.traffic_map.intersections["Mayor_Magrath"]

        for lane in intersection.incoming_lanes:
            stopped_vehicles += sum(1 for car in lane.vehicles if car.speed <= 0.1)

        return float(-stopped_vehicles)

    def step(self, action_array):
        """
        Advance the simulation by one decision interval.

        :param action_array: Array-like containing a single integer action (0-3).
        :return: (observation, reward, terminated, truncated, info)
        """
        action = int(action_array[0])
        total_reward = 0.0

        intersection = self.traffic_map.intersections["Mayor_Magrath"]
        current_phase_index = intersection.current_phase_index
        is_phase_change = (action != current_phase_index)

        # If switching phases, insert 3 seconds of all-red clearance first
        yellow_duration = self.yellow_duration if is_phase_change else 0

        # Run micro-steps for the full decision interval
        for step_idx in range(self.decision_interval):
            # 1. Apply signal control
            if step_idx < yellow_duration:
                # All-red clearance period
                intersection.apply_action(-1, dt=self.dt)
            else:
                # Normal green phase for the target action
                intersection.apply_action(action, dt=self.dt)

            # 2. Generate stochastic arrivals for this second
            new_entities_dict = self.traffic_generator.generate_entities(float(self.time_step))

            # 3. Inject new arrivals into the correct incoming lanes based on movement type
            for lane in self.traffic_map.lanes:
                if lane.to_node_id != "Mayor_Magrath":
                    continue

                direction = lane.approach_direction
                entities_to_add = new_entities_dict.get(direction, [])

                if not entities_to_add:
                    continue

                # Filter entities: pedestrians are not placed on vehicle lanes
                vehicles_to_add = [e for e in entities_to_add if hasattr(e, "movement_type")]

                # Assign each vehicle to the correct lane based on its movement type
                for vehicle in vehicles_to_add:
                    if lane.lane_type == "straight_right":
                        if vehicle.movement_type in ["straight", "right"]:
                            if len(lane.vehicles) < 40:
                                lane.vehicles.append(vehicle)
                    elif lane.lane_type == "left_uturn":
                        if vehicle.movement_type in ["left", "u_turn"]:
                            if len(lane.vehicles) < 40:
                                lane.vehicles.append(vehicle)

            # Clear the temporary entities dictionary for the next second
            for direction in new_entities_dict.keys():
                new_entities_dict[direction] = []

            # 4. Advance physics simulation by 1 second
            self.traffic_map.step(dt=self.dt)

            # 5. Accumulate reward
            total_reward += self.calculate_reward()

            # Advance global clock
            self.time_step += 1

            # Break if end of day is reached
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

        # Clear all vehicles from all lanes
        for lane in self.traffic_map.lanes:
            lane.vehicles.clear()

        # Reset all intersections to default phase
        for inter in self.traffic_map.intersections.values():
            inter.current_phase_index = 0
            inter.current_phase = inter.phases[0]
            inter.phase_timer = 0.0
            inter.reset_stats()

        return self.get_state(), {}