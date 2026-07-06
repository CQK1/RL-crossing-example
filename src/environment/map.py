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

        new_lane.from_node_id = from_node_id
        new_lane.to_node_id = to_node_id

        from_node.outgoing_lanes.append(new_lane)
        to_node.incoming_lanes.append(new_lane)

        self.lanes.append(new_lane)
        return new_lane
            

    def step(self, dt=0.1):
        """the time of the whole map changes"""
        # goes through every intersection and notify the car the information of the intersections
        for intersection in self.intersections.values():
            is_red_light = (intersection.light_state == 0)
            
            # 只让“流向当前路口”的进口车道去对齐这个路口的红绿灯
            for lane in intersection.incoming_lanes:
                # 接住驶出当前车道的车辆
                leaving_cars = lane.update_vehicles_physics(
                    dt=dt, 
                    stop_line=lane.length - 2.0,
                    is_red=is_red_light
                )
                
                # 车辆跨路段交接逻辑 (Handoff)
                if leaving_cars:
                    for car in leaving_cars:
                        if car.destination == f"{intersection.name}_left":
                            intersection.stats["left"] += 1
                            pass
                        elif car.destination == intersection.name:
                            intersection.stats["straight"] += 1
                            pass
                        elif len(intersection.outgoing_lanes) > 0:
                            intersection.stats["straight"] += 1
                            next_lane = intersection.outgoing_lanes[0]
                            # 重置车辆在新车道的相对起点坐标
                            car.position = 0.0 
                            # 保持原本的速度驶入下一路段
                            next_lane.vehicles.append(car)