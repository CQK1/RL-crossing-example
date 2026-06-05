import random
from src.entities.entities import Vehicle, Pedestrian

class TrafficGenerator:
    def __init__(self, vehicle_rate=0.3, pedestrain_rate=0.2):
        self.vehicle_rate = vehicle_rate
        self.pedestrain_rate = pedestrain_rate

    def generate_vehicle(self):
        if random.random() < self.vehicle_rate:
            return Vehicle(start_pos=0)
        return None
    
    def generate_pedastrains(self):
        if random.random() < self.vehicle_rate:
            return Vehicle(start_pos=0)
        return None