import random
from entities import Vehicle, Pedestrian

class Intersection:
    def __init__(self):
        self.ns_weight = 1.0  # north-south direction vehicles weight
        self.ew_weight = 1.0  # east-west direction vehicles weight
        self.ped_weight = 1.5

        self.road_length = 10
        self.crossing_point = 5
        self.car_stop_line = 4

        self.ped_path_length = 5
        self.ped_stop_line = 2
        self.reset()
    
    def reset(self):
        self.light_state = 1 # assume north-south vehicles have green light initially, east-west vehicles and pedestrians have red light
        self.vehicles = {
            "N": [], 
            "S": [],
            "E": [],
            "W": []
        }
        self.pedestrians = []
        self.time_step = 0

        return self.get_state()
    
    def get_state(self):
        ns_waiting = 0
        if self.light_state == 0:
            ns_waiting += sum(1 for v in self.vehicles["N"] if v.position <= self.car_stop_line)
            ns_waiting += sum(1 for v in self.vehicles["S"] if v.position <= self.car_stop_line)

        ew_waiting = 0
        if self.light_state == 1:
            ew_waiting += sum(1 for v in self.vehicles["E"] if v.position <= self.car_stop_line)
            ew_waiting += sum(1 for v in self.vehicles["W"] if v.position <= self.car_stop_line)

        return {
            "light_state": self.light_state,
            "waiting_ns_vehicles": ns_waiting,
            "waiting_ew_vehicles": ew_waiting
        }
    
    def step(self, action):
        """
        action: 0 for keeping the current light state, 1 for switching the light state
        """

        if action == 1:
            self.light_state = 1 - self.light_state

        if random.random() < 0.2: self.vehicles["N"].append(Vehicle(start_pos=0))
        if random.random() < 0.2: self.vehicles["S"].append(Vehicle(start_pos=0))
        if random.random() < 0.2: self.vehicles["E"].append(Vehicle(start_pos=0))
        if random.random() < 0.2: self.vehicles["W"].append(Vehicle(start_pos=0))

        current_ns_wait = 0
        current_ew_wait = 0

        is_ns_green = (self.light_state == 1)
        is_ew_green = (self.light_state == 0)

        for direction in ["N", "S"]:
            for v in self.vehicles[direction]:
                old_pos = v.position
                v.move(is_ns_green)
                if v.position == old_pos:
                    current_ns_wait += 1

        for direction in ["E", "W"]:
            for v in self.vehicles[direction]:
                old_pos = v.position
                v.move(is_ew_green)
                if v.position == old_pos:
                    current_ew_wait += 1

        for direction in ["N", "S", "E", "W"]:
            self.vehicles[direction] = [v for v in self.vehicles[direction] if v.position < self.road_length]

        reward = - (self.ns_weight * current_ns_wait + self.ew_weight * current_ew_wait)

        done = self.time_step >= 100

        return self.get_state(), reward, done