import random
from typing import Any
from src.entities.pedestrian import Pedestrian
from src.entities.vehicle import Vehicle


class TrafficGenerator:
    def __init__(self, rate_model: Any):
        """
        Traffic Generator Engine.

        This class acts as the "factory" that creates physical vehicle/pedestrian instances.
        It is decoupled from the mathematical logic of *when* to spawn them.

        :param rate_model: A mathematical model instance (e.g., InhomogeneousPoissonProcess)
                           that provides a method to query the arrival probability lambda(t)
                           for any given time step and direction.
        """
        # Dependency injection: the mathematical model provides arrival rates
        self.rate_model = rate_model

        # Mapping: physical approach directions (North, South, East, West)
        # to the column names in the Excel dataset (mathematical model).
        self.movements_mapping = {
            "North": {
                "straight": "north_thru",
                "left": "north_left",
                "right": "north_right",
                "u_turn": "north_u_turn",
                "pedestrian": "north_peds_cw"
            },
            "South": {
                "straight": "south_thru",
                "left": "south_left",
                "right": "south_right",
                "u_turn": "south_u_turn",
                "pedestrian": "south_peds_cw"
            },
            "East": {
                "straight": "east_thru",
                "left": "east_left",
                "right": "east_right",
                "u_turn": "east_u_turn",
                "pedestrian": "east_peds_cw"
            },
            "West": {
                "straight": "west_thru",
                "left": "west_left",
                "right": "west_right",
                "u_turn": "west_u_turn",
                "pedestrian": "west_peds_cw"
            }
        }

    def generate_entities(self, time_in_seconds: float):
        """
        Generate physical vehicles and pedestrians based on arrival probabilities
        provided by the external rate_model at the current simulation time.

        :param time_in_seconds: Current global simulation time in seconds.
        :return: A dictionary containing lists of generated entities for each approach direction.
                 Format: {"North": [Vehicle, Pedestrian, ...], "South": [...], ...}
                 Each Vehicle has a `destination` string and a `movement_type` attribute.
        """
        new_entities = {"North": [], "South": [], "East": [], "West": []}

        for direction, movements in self.movements_mapping.items():
            # 1. Generate motor vehicles
            for intent, column_name in movements.items():
                if intent == "pedestrian":
                    continue  # Pedestrians are handled separately below

                # Query the mathematical model for the arrival probability at this time
                probability = self.rate_model.get_rate_interpolated(time_in_seconds, column_name)

                # Roll the dice: Poisson arrival simulation
                if random.random() < probability:
                    # Build the destination label that matches intersection.py's is_movement_allowed format
                    # Example: intent "left" -> destination "Mayor_Magrath_left"
                    if intent == "straight":
                        dest = "Mayor_Magrath"
                    else:
                        dest = f"Mayor_Magrath_{intent}"

                    car = Vehicle(start_pos=0.0, destination=dest)
                    car.movement_type = intent  # Store movement type for lane assignment
                    new_entities[direction].append(car)

            # 2. Generate pedestrians
            ped_column = movements.get("pedestrian")
            if ped_column:
                ped_prob = self.rate_model.get_rate_interpolated(time_in_seconds, ped_column)
                if random.random() < ped_prob:
                    pedestrian = Pedestrian(start_pos=0.0)
                    new_entities[direction].append(pedestrian)

        return new_entities