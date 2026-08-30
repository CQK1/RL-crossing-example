import random
from typing import Any
from src.entities.pedestrian import Pedestrian
from src.entities.vehicle import Vehicle


class TrafficGenerator:
    def __init__(self, rate_model: Any):
        """
        Traffic Generator Engine.

        This class acts as the "factory" that creates physical vehicle instances.
        It is decoupled from the mathematical logic of *when* to spawn them.

        :param rate_model: A mathematical model instance (e.g., InhomogeneousPoissonProcess)
                           that provides methods to query arrival probability.
        """
        self.rate_model = rate_model

    def generate_for_lane(self, second_idx: int, approach_direction: str, lane_type: str):
        """
        Generate vehicles directly matched to a specific incoming lane.

        :param second_idx: Current simulation time in seconds (integer).
        :param approach_direction: "North", "South", "East", or "West".
        :param lane_type: "straight_right" or "left_uturn".
        :return: List of Vehicle objects to inject into the lane.
        """
        new_cars = []
        sec = second_idx % 86400
        prefix = approach_direction.lower()

        if lane_type == "straight_right":
            thru_prob = self.rate_model.get_rate_fast(sec, f"{prefix}_thru")
            right_prob = self.rate_model.get_rate_fast(sec, f"{prefix}_right")

            if random.random() < thru_prob:
                car = Vehicle(start_pos=0.0, destination="Mayor_Magrath")
                car.movement_type = "straight"
                new_cars.append(car)

            if random.random() < right_prob:
                car = Vehicle(start_pos=0.0, destination="Mayor_Magrath_right")
                car.movement_type = "right"
                new_cars.append(car)

        elif lane_type == "left_uturn":
            left_prob = self.rate_model.get_rate_fast(sec, f"{prefix}_left")
            uturn_prob = self.rate_model.get_rate_fast(sec, f"{prefix}_u_turn")

            if random.random() < left_prob:
                car = Vehicle(start_pos=0.0, destination="Mayor_Magrath_left")
                car.movement_type = "left"
                new_cars.append(car)

            if random.random() < uturn_prob:
                car = Vehicle(start_pos=0.0, destination="Mayor_Magrath_u_turn")
                car.movement_type = "u_turn"
                new_cars.append(car)

        return new_cars