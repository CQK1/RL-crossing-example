import random
from src.entities.pedestrian import Pedestrian
from src.entities.vehicle import Vehicle

class TrafficGenerator:
    def __init__(self, vehicle_rate=0.3, pedestrain_rate=0.2):
        self.vehicle_rate = vehicle_rate
        self.pedestrain_rate = pedestrain_rate

    def generate_vehicle(self, start_pos=0):
        if random.random() < self.vehicle_rate:
            return Vehicle(start_pos)
        return None
    
    def generate_pedastrains(self, start_pos=0):
        if random.random() < self.vehicle_rate:
            return Pedestrian(start_pos)
        return None