import random


class Vehicle:
    def __init__(self, start_pos=0, destination="Node_C"):
        # Physical properties of the vehicle
        self.position = float(start_pos)  # Position of the front bumper
        self.length = random.uniform(4.0, 5.0)  # Vehicle length in meters

        # Dynamic state
        self.speed = 0.0  # Current speed in m/s
        self.acceleration = 0.0  # Current acceleration in m/s^2

        # Cumulative waiting time (seconds this vehicle has been stopped)
        self.waiting_time = 0.0

        # Destination label (e.g., "Mayor_Magrath", "Mayor_Magrath_left")
        self.destination = destination

        # Movement type for lane assignment (e.g., "straight", "left", "right", "u_turn")
        self.movement_type = self._infer_movement_type(destination)

    def _infer_movement_type(self, destination):
        """
        Infer the movement type from the destination string.

        :param destination: Destination label string.
        :return: Movement type string: "straight", "left", "right", or "u_turn".
        """
        dest = str(destination).lower()

        if dest.endswith("_left"):
            return "left"
        elif dest.endswith("_right"):
            return "right"
        elif dest.endswith("_u_turn"):
            return "u_turn"
        else:
            return "straight"

    def move(self, is_green):
        """
        Legacy discrete movement method (kept for backward compatibility).

        :param is_green: If True, vehicle advances; if False, it waits.
        """
        if is_green:
            self.position += 1
        else:
            self.waiting_time += 1

    @property
    def back_position(self):
        """
        Position of the rear bumper.

        :return: Rear bumper position in meters.
        """
        return self.position - self.length

    def move_continuous(self, dt=1.0):
        """
        Advance vehicle physics by one time step using continuous dynamics.

        :param dt: Time step duration in seconds.
        """
        # Update speed based on current acceleration
        self.speed = max(0.0, self.speed + self.acceleration * dt)

        # Update position based on current speed
        self.position += self.speed * dt

        # Accumulate waiting time when the vehicle is fully stopped
        if self.speed == 0.0:
            self.waiting_time += dt