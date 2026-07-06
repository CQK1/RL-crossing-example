import random
from src.entities.pedestrian import Pedestrian
from src.entities.vehicle import Vehicle
from src.generators.traffic_generator import TrafficGenerator

class Intersection:
    def __init__(self, name = "Intersection_1", x = 0.0, y = 0.0):
        self.name = name
        self.x = x
        self.y = y
        self.stats = {"straight": 0, "left": 0}
        self.incoming_lanes = []
        self.outgoing_lanes = []

        self.phases = [
            "NS_Straight",  # 0: North and south straight
            "NS_Left",      # 1: North and south left
            "EW_Straight",  # 2: East and west straight
            "EW_Left"       # 3: East and west left
        ]

        self.current_phase_index = 0  # Default is from south north straight
        self.current_phase = self.phases[self.current_phase_index]
        self.phase_timer = 0.0        # How long has the current phase has lasted
        
        # 2. Hard rules
        self.min_green = 10.0
        self.max_green = 60.0

    def is_lane_allowed(self, lane_id):
        if self.current_phase == "NS_Straight":
        # 如果当前是南北直行相位：只有南北向的直行车道可以走
            if "Straight" in lane_id and ("North" in lane_id or "South" in lane_id):
                return True
            
        elif self.current_phase == "NS_Left":
            # 如果当前是南北左转相位：只有南北向的左转车道可以走
            if "Left" in lane_id and ("North" in lane_id or "South" in lane_id):
                return True
                
        elif self.current_phase == "EW_Straight":
            # 东西直行相位
            if "Straight" in lane_id and ("East" in lane_id or "West" in lane_id):
                return True
                
        elif self.current_phase == "EW_Left":
            # 东西左转相位
            if "Left" in lane_id and ("East" in lane_id or "West" in lane_id):
                return True
            
        # 其余所有车道一律红灯，不准通过
        return False

    def apply_action(self, rl_action, dt=1.0):
        """
        Process the action
        """
        requested_change = (rl_action != self.current_phase_index)

         # 1. Hard Rule: Max Green constraint (Force a change to prevent starvation)
        if self.phase_timer >= self.max_green:
            # Force transition to the next logical phase in sequence if we exceed max green
            next_phase = (self.current_phase_index + 1) % len(self.phases)
            self.current_phase_index = next_phase
            self.current_phase = self.phases[self.current_phase_index]
            self.phase_timer = 0.0
            return self.current_phase

        # 2. Check if the agent wants to switch the phase
        if requested_change:
            # Check if current phase has been active long enough
            if self.phase_timer >= self.min_green:
                # Allowed to switch directly to the chosen target phase
                self.current_phase_index = rl_action
                self.current_phase = self.phases[self.current_phase_index]
                self.phase_timer = 0.0
            else:
                # Forced to keep current phase because min_green is not satisfied
                self.phase_timer += dt
        else:
            # Agent decided to keep the current phase, increment the timer
            self.phase_timer += dt

        return self.current_phase

    def reset_stats(self):
        self.stats = {"straight": 0, "left": 0}