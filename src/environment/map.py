import math
from src.environment.lane import Lane
from src.environment.intersection import Intersection

class TrafficMap:
    def __init__(self):
        """
        Initialize the TrafficMap representing nodes (intersections) and edges (lanes).
        """
        self.intersections = {}
        self.lanes = []
        
    def add_intersection(self, node_id, x, y):
        """
        Add a new intersection node to the map if it doesn't already exist.
        """
        if node_id not in self.intersections:
            self.intersections[node_id] = Intersection(node_id, x, y)

    def add_line(self, from_node_id, to_node_id, speed_limit):
        """
        Construct a road segment (lane) connecting two intersections.
        """
        if from_node_id not in self.intersections or to_node_id not in self.intersections:
            raise ValueError("Intersections you typed do not exist, add them first.")
        
        from_node = self.intersections[from_node_id]
        to_node = self.intersections[to_node_id]

        # Calculate Euclidean distance between the intersections
        length = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)

        new_lane = Lane(length=length, speed_limit=speed_limit)
        new_lane.from_node_id = from_node_id
        new_lane.to_node_id = to_node_id

        # Smart Inference: Calculate the vehicle's approach direction based on coordinates.
        # For example, driving from West to East (dx > 0) means the vehicle enters the intersection from the West.
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        if abs(dx) >= abs(dy):
            new_lane.approach_direction = "West" if dx > 0 else "East"
        else:
            new_lane.approach_direction = "South" if dy > 0 else "North"

        # Register the lane link within intersections
        from_node.outgoing_lanes.append(new_lane)
        to_node.incoming_lanes.append(new_lane)

        self.lanes.append(new_lane)
        return new_lane
            
    def step(self, dt=0.1):
        """
        Progress the simulation time steps across the entire road network.
        """
        # Go through every intersection to notify vehicles of signal configurations
        for intersection in self.intersections.values():
            
            # Only update incoming lanes flowing into the current intersection to align with its signals
            for lane in intersection.incoming_lanes:
                
                # Define a closure function to determine if a specific vehicle is facing a red light
                def check_red_light_for_car(car):
                    # Parse the movement intention from the vehicle's destination
                    destination = str(car.destination).lower()
                    if destination.endswith("_left"):
                        movement = "left"
                    elif destination.endswith("_right"):
                        movement = "right"
                    elif destination.endswith("_uturn"):
                        movement = "u_turn"
                    else:
                        movement = "straight"
                        
                    # Query whether the current phase of the intersection permits the vehicle's movement
                    is_allowed = intersection.is_movement_allowed(
                        approach_direction=lane.approach_direction,
                        movement_type=movement
                    )
                    # If allowed is False, it translates to a red light (True) for the vehicle physical simulation
                    return not is_allowed 

                # Pass the evaluation function to the vehicle physics engine for autonomous checking
                leaving_cars = lane.update_vehicles_physics(
                    dt=dt, 
                    stop_line=lane.length - 2.0,
                    is_red_func=check_red_light_for_car
                )
                
                # Vehicle handoff logic across intersections
                if leaving_cars:
                    for car in leaving_cars:
                        if car.destination == f"{intersection.name}_left":
                            intersection.stats["left"] += 1
                        elif car.destination == f"{intersection.name}_right":
                            intersection.stats["right"] += 1
                        elif car.destination == intersection.name:
                            intersection.stats["straight"] += 1
                        elif len(intersection.outgoing_lanes) > 0:
                            intersection.stats["straight"] += 1
                            next_lane = intersection.outgoing_lanes[0]
                            # Reset the vehicle's position to the beginning of the new lane
                            car.position = 0.0 
                            # Enter the next road segment while retaining its original speed
                            next_lane.vehicles.append(car)