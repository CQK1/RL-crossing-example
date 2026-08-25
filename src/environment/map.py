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

    def add_line(self, from_node_id, to_node_id, speed_limit, lane_type="all"):
        """
        Construct a road segment (lane) connecting two intersections.

        :param from_node_id: ID of the source intersection.
        :param to_node_id: ID of the target intersection.
        :param speed_limit: Maximum speed allowed on this lane (m/s).
        :param lane_type: Type of movements allowed on this lane.
                          Options:
                            - "all": All movements allowed (legacy behavior)
                            - "straight_right": Straight and right-turn vehicles only
                            - "left_uturn": Left-turn and U-turn vehicles only
        """
        if from_node_id not in self.intersections or to_node_id not in self.intersections:
            raise ValueError("Intersections you typed do not exist, add them first.")

        from_node = self.intersections[from_node_id]
        to_node = self.intersections[to_node_id]

        # Calculate Euclidean distance between the intersections
        length = math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)

        new_lane = Lane(length=length, speed_limit=speed_limit, lane_type=lane_type)
        new_lane.from_node_id = from_node_id
        new_lane.to_node_id = to_node_id

        # Smart Inference: Calculate the vehicle's approach direction based on coordinates.
        # Example: driving from West to East (dx > 0) means the vehicle enters from the West.
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
        for intersection in self.intersections.values():
            for lane in intersection.incoming_lanes:

                # Exit nodes have no traffic light; vehicles always proceed
                if "exit" in str(intersection.name).lower():
                    def check_red_light_for_car(car):
                        return False
                else:
                    # Normal intersection: check signal state for each vehicle
                    def check_red_light_for_car(car):
                        movement = self._get_movement_type(car)
                        is_allowed = intersection.is_movement_allowed(
                            approach_direction=lane.approach_direction,
                            movement_type=movement
                        )
                        return not is_allowed

                leaving_cars = lane.update_vehicles_physics(
                    dt=dt,
                    stop_line=lane.length - 2.0,
                    is_red_func=check_red_light_for_car
                )

                # Hand off vehicles that have left this lane
                if leaving_cars:
                    self._handoff_vehicles(intersection, leaving_cars)

    def _get_movement_type(self, car):
        """
        Determine the movement type of a vehicle based on its destination string.

        :param car: Vehicle object.
        :return: Movement type string: "straight", "left", "right", or "u_turn".
        """
        destination = str(car.destination).lower()

        if destination.endswith("_left"):
            return "left"
        elif destination.endswith("_right"):
            return "right"
        elif destination.endswith("_u_turn"):  # Fixed: was "_uturn"
            return "u_turn"
        else:
            return "straight"

    def _handoff_vehicles(self, intersection, leaving_cars):
        """
        Route vehicles that have exited an incoming lane to the correct outgoing lane.

        :param intersection: The intersection the vehicles just passed through.
        :param leaving_cars: List of vehicles that exited the incoming lane.
        """
        for car in leaving_cars:
            # Update throughput statistics
            dest = str(car.destination).lower()
            if dest.endswith("_left"):
                intersection.stats["left"] += 1
            elif dest.endswith("_right"):
                intersection.stats["right"] += 1
            elif dest.endswith("_u_turn"):
                intersection.stats["u_turn"] += 1
            else:
                intersection.stats["straight"] += 1

            # Route vehicle to the correct outgoing lane based on its destination
            next_lane = self._select_outgoing_lane(intersection, car)
            if next_lane is not None:
                car.position = 0.0
                next_lane.vehicles.append(car)

    def _select_outgoing_lane(self, intersection, car):
        """
        Select the appropriate outgoing lane for a vehicle based on its destination.

        :param intersection: The intersection the vehicle just passed through.
        :param car: Vehicle object.
        :return: The outgoing Lane object, or None if no matching lane exists.
        """
        dest = str(car.destination).lower()
        outgoing_lanes = intersection.outgoing_lanes

        if not outgoing_lanes:
            return None

        # If destination is an Exit node name (e.g., "north_exit"),
        # match to the outgoing lane whose to_node_id matches that exit node.
        for lane in outgoing_lanes:
            to_node_id = str(lane.to_node_id).lower()
            if dest.endswith(to_node_id) or to_node_id.endswith(dest):
                return lane

        # Fallback: if no exact match, route to the first available outgoing lane
        return outgoing_lanes[0]