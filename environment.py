import random
from entities import Vehicle, Pedestrian

class Crossing:
    def __init__(self):
        self.road_length = 10
        self.crossing_point = 5
        self.reset()

    def reset(self):
        self.light_state = 'green' # assume vehicles have green light initially, pedestrians have red light
        self.vehicles = []
        self.pedestrians = []
        self.time_step = 0
