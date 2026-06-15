# src/environment/network_traffic_env.py
from src.environment.map import TrafficMap
from src.generators.traffic_generator import TrafficGenerator

class NetworkTrafficEnv:
    def __init__(self):
        self.traffic_map = TrafficMap()
        self.traffic_generator = TrafficGenerator()
        self.time_step = 0

    def get_state(self):
        """
        collect the state of every intersection in the map
        Then return it, example:
        {
            "Node_B": {"light_state": 0, "waiting_vehicles": 3},
            "Node_C": {"light_state": 1, "waiting_vehicles": 0}
        }
        """
        state_dict = {}

        for inter_id, intersection in self.traffic_map.intersection.items():
            total_waiting_vehicles = []
            for lane in intersection.incoming_lanes:
                total_waiting_vehicles += sum(1 for car in lane.vehicles if car.speed == 0.0)

        state_dict[inter_id] = {
            "light_state": intersection.light_state,
            "waiting_vehicles": total_waiting_vehicles
            # TODO: After adding pedestrains, calculate the amount that is waiting
            # "waiting_pedestrians": ...
        }
        
        return state_dict
    
    def calculate_reward(self):
        total_penalty = 0.0
        for inter_id, intersection in self.traffic_map.intersections.items():
            for lane in intersection.incoming_lanes:
                total_penalty += sum(1.0 for car in lane.vehicles if car.speed == 0.0)
        return total_penalty

    def step(self, action_dict):
        """
        action_dict: The dictionary that contains multiple actions for each intersections, like {"Node_B": 1, "Node_C": 0}
        """
        self.time_step += 1

        for intersection_id, action in action_dict.items():
            if action == 1 and intersection_id in self.traffic_map.intersections:
                self.traffic_map.intersections[intersection_id].toggle_light()

        for lane in self.traffic_map.lanes():
            new_car = self.traffic_generator.generate_vehicle()
            if new_car:
                lane.vehicles.append(new_car)
            
        self.traffic_map.step(dt = 1.0)

        state = self.get_state()
        reward = self.calculate_reward()
        done = self.time_step >= 100

        return state, reward, done
    