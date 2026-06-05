import random
from src.entities.entities import Vehicle, Pedestrian
from src.generators.traffic_generator import TrafficGenerator

class Crossing:
    def __init__(self):
        # Define weights for reward calculation
        self.car_weight = 1
        self.ped_weight = 1.5

        self.road_length = 10
        self.crossing_point = 5
        self.car_stop_line = 4

        self.ped_path_length = 5
        self.ped_stop_line = 2

        # initialize the traffic generator
        self.traffic_generator = TrafficGenerator(vehicle_rate= 0.3, pedestrain_rate= 0.2)

        self.reset()

    def reset(self):
        self.light_state = 1 # assume vehicles have green light initially, pedestrians have red light
        self.vehicles = []
        self.pedestrians = []
        self.time_step = 0

        return self.get_state()
    
    def get_state(self):
        waiting_vehicles = sum(1 for v in self.vehicles if v.position <= self.car_stop_line and self.light_state == 0)
        waiting_pedestrians = sum(1 for p in self.pedestrians if p.position <= self.ped_stop_line and self.light_state == 1)

        return {
            "light_state": self.light_state,
            "waiting_vehicles": waiting_vehicles,
            "waiting_pedestrians": waiting_pedestrians
        }
    
    def step(self, action):
        """
        action: 0 for keeping the current light state, 1 for switching the light state
        """
        self.time_step += 1

        if action == 1:
            self.light_state = 1 - self.light_state # switch light state

        if random.random() < 0.3: # 30% chance of a new vehicle arriving
            self.vehicles.append(Vehicle(start_pos=0))
        if random.random() < 0.2: # 20% chance of a new pedestrian arriving
            self.pedestrians.append(Pedestrian(start_pos=0))

        current_step_car_wait = 0
        current_step_ped_wait = 0

        for v in self.vehicles:
            is_car_green = (self.light_state == 1)
            old_pos = v.position
            v.move(is_car_green)
            if v.position == old_pos:
                current_step_car_wait += 1 # count how many vehicles are waiting at the stop line

        for p in self.pedestrians:
            is_ped_green = (self.light_state == 0)
            old_pos = p.position
            p.move(is_ped_green)
            if p.position == old_pos:
                current_step_ped_wait += 1 # count how many pedestrians are waiting at the stop line

        self.vehicles = [v for v in self.vehicles if v.position < self.road_length] # remove vehicles that have crossed
        self.pedestrians = [p for p in self.pedestrians if p.position < self.ped_path_length] # remove pedestrians that have crossed

        reward = - (self.car_weight * current_step_car_wait + self.ped_weight * current_step_ped_wait) # negative reward for waiting vehicles and pedestrians

        done = self.time_step >= 100 # episode ends after 100 time steps

        return self.get_state(), reward, done

