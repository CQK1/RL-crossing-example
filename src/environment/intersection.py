import random

class Intersection:
    def __init__(self, name="Intersection_1", x=0.0, y=0.0):
        self.name = name
        self.x = x
        self.y = y
        
        # Extended statistics dictionary to track all movement types from real-world data
        self.stats = {
            "straight": 0, 
            "left": 0, 
            "right": 0, 
            "u_turn": 0, 
            "pedestrian": 0
        }
        
        self.incoming_lanes = []
        self.outgoing_lanes = []

        # Typical 4-phase design (integrating right turns, U-turns, and pedestrians logically)
        self.phases = [
            "NS_Straight",  # 0: North/South straight + right + parallel pedestrians
            "NS_Left",      # 1: North/South left + U-turn
            "EW_Straight",  # 2: East/West straight + right + parallel pedestrians
            "EW_Left"       # 3: East/West left + U-turn
        ]

        self.current_phase_index = 0  # Default is North/South straight
        self.current_phase = self.phases[self.current_phase_index]
        self.phase_timer = 0.0        # Duration of the current phase
        
        # Physical and safety constraints
        self.min_green = 10.0
        self.max_green = 60.0

    def is_movement_allowed(self, approach_direction, movement_type):
        """
        Determine if the current phase allows a specific traffic movement.
        This prepares the environment for integrating with the real-world data generator.
        
        :param approach_direction: Entering direction, e.g., "North", "South", "East", "West"
        :param movement_type: Movement type, e.g., "straight", "left", "right", "u_turn", "pedestrian"
        :return: bool
        """
        movement_type = movement_type.lower()
        
        if self.current_phase == "NS_Straight":
            # North/South straight green: North/South straight and right turns can proceed
            if approach_direction in ["North", "South"] and movement_type in ["straight", "right"]:
                return True
            # Pedestrian logic: When North/South vehicles go straight, East/West crosswalks are green (parallel)
            if approach_direction in ["East", "West"] and movement_type == "pedestrian":
                return True

        elif self.current_phase == "NS_Left":
            # North/South left turn green: Left turns and U-turns are allowed
            if approach_direction in ["North", "South"] and movement_type in ["left", "u_turn"]:
                return True

        elif self.current_phase == "EW_Straight":
            # East/West straight green: East/West straight and right turns can proceed
            if approach_direction in ["East", "West"] and movement_type in ["straight", "right"]:
                return True
            # Pedestrian logic: When East/West vehicles go straight, North/South crosswalks are green
            if approach_direction in ["North", "South"] and movement_type == "pedestrian":
                return True

        elif self.current_phase == "EW_Left":
            # East/West left turn green: Left turns and U-turns are allowed
            if approach_direction in ["East", "West"] and movement_type in ["left", "u_turn"]:
                return True

        # Red light for all other directions and actions
        return False

    def is_lane_allowed(self, lane_id):
        """
        Backward compatibility layer for legacy lane_id string checking.
        It is recommended to migrate to is_movement_allowed during Map and Lane refactoring.
        """
        if self.current_phase == "NS_Straight":
            if ("North" in lane_id or "South" in lane_id) and ("Straight" in lane_id or "Right" in lane_id):
                return True
        elif self.current_phase == "NS_Left":
            if ("North" in lane_id or "South" in lane_id) and ("Left" in lane_id or "U_Turn" in lane_id):
                return True
        elif self.current_phase == "EW_Straight":
            if ("East" in lane_id or "West" in lane_id) and ("Straight" in lane_id or "Right" in lane_id):
                return True
        elif self.current_phase == "EW_Left":
            if ("East" in lane_id or "West" in lane_id) and ("Left" in lane_id or "U_Turn" in lane_id):
                return True
        return False

    def apply_action(self, rl_action, dt=1.0):
        """
        Process the phase action chosen by the Reinforcement Learning Agent.
        """
        requested_change = (rl_action != self.current_phase_index)

        # 1. Hard Rule: Max Green constraint (Force a phase shift to prevent starvation)
        if self.phase_timer >= self.max_green:
            # Shift to the next logical phase sequentially if maximum green time is exceeded
            next_phase = (self.current_phase_index + 1) % len(self.phases)
            self.current_phase_index = next_phase
            self.current_phase = self.phases[self.current_phase_index]
            self.phase_timer = 0.0
            return self.current_phase

        # 2. Process RL agent transition requests
        if requested_change:
            # Check if the current phase has met the minimum duration requirements
            if self.phase_timer >= self.min_green:
                # Allowed to shift directly to the target phase
                self.current_phase_index = rl_action
                self.current_phase = self.phases[self.current_phase_index]
                self.phase_timer = 0.0
            else:
                # Retain current phase because min_green constraint is not satisfied
                self.phase_timer += dt
        else:
            # Agent decided to hold current phase, increment phase timer
            self.phase_timer += dt

        return self.current_phase

    def reset_stats(self):
        """Reset the throughput statistics for this intersection."""
        self.stats = {
            "straight": 0, 
            "left": 0, 
            "right": 0, 
            "u_turn": 0, 
            "pedestrian": 0
        }