import random
from src.entities.pedestrian import Pedestrian
from src.entities.vehicle import Vehicle
from src.generators.traffic_generator import TrafficGenerator

class Intersection:
    def __init__(self, name = "Intersection_1", x = 0.0, y = 0.0):
        self.name = name
        self.x = x
        self.y = y
        self.light_state = 0
        self.stats = {"straight": 0, "left": 0}
        self.incoming_lanes = []
        self.outgoing_lanes = []

    def toggle_light(self):
        self.light_state = 1 - self.light_state

    def reset_stats(self):
        self.stats = {"straight": 0, "left": 0}