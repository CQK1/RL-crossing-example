# src/environment/map.py
import math
from src.environment.lane import Lane
from src.environment.intersection import Intersection

class TrafficMap:
    def __init__(self):
        self.intersections = {}
        self.lanes = []
        
    def add_intersection(self, node_id, x, y):
        if node_id not in self.intersections:
            self.intersections[node_id] = Intersection(node_id, x, y)

    def add_line(self, from_node_id, to_node_id, speed_limit):
        if from_node_id not in self.intersections or to_node_id not in self.intersections:
            raise ValueError("Intersetions you typed do not exist, add them first.")
        from_node = self.intersections[from_node_id]
        to_node = self.intersections[to_node_id]

        length = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)

        new_lane = Lane(length = length, speed_limit = speed_limit)

        from_node.outgoing_lanes.append(new_lane)
        to_node.incoming_lanes.append(new_lane)

        self.lanes.append(new_lane)
        return new_lane
            
    def step(self, dt=1.0):
        """the time of the whole map changes"""
        # goes through every intersection and notify the car the information of the intersections
        # assume a car only needs to consider the red light in front of it
        for intersection in self.intersections.values():
            is_red_light = (intersection.light_state == 0)
            
            # 【核心】：只让“流向当前路口”的进口车道去对齐这个路口的红绿灯和坐标
            for lane in intersection.incoming_lanes:
                lane.update_vehicles_physics(     # 1. 修正了单数拼写
                    dt=dt, 
                    stop_line=intersection.x - 2.0,  # 2. 修正了坐标属性为 .x
                    is_red=is_red_light
                )