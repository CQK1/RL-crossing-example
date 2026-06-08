# src/environment/map.py
from src.environment.lane import Lane
from src.environment.intersection import Intersection

class TrafficMap:
    def __init__(self, map_length=300.0):
        self.map_length = map_length # default length 300
        self.lane = Lane(length=map_length) # let the length of the lane equal to map first
        self.intersections = []  # list of intersection in the map
        
    def add_intersection(self, name, position):
        """if the intersection is within the length of the map then add"""
        if position < self.map_length:
            intersection = Intersection(name, position)
            self.intersections = sorted(self.intersections + [intersection], key=lambda x: x.position)
            
    def step(self, dt=1.0):
        """the time of the whole map changes"""
        # goes through every intersection and notify the car the information of the intersections
        # assume a car only needs to consider the red light in front of it
        for intersection in self.intersections:
            self.lane.update_vehicles_physics(
                dt=dt, 
                stop_line=intersection.position - 2.0,  # let the stop line be 2 meters before the stop line
                is_red=(intersection.light_state == 0)
            )