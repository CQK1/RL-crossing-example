class Lane:
    def __init__(self, length=150.0, speed_limit=40, lane_type="all"):
        """
        Initialize the Lane segment representing a single road path.

        :param length: Total length of the lane in meters.
        :param speed_limit: Maximum allowed speed on this lane in m/s.
        :param lane_type: Type of movements allowed on this lane.
                          Options:
                            - "all": All movements allowed (legacy behavior)
                            - "straight_right": Straight and right-turn vehicles only
                            - "left_uturn": Left-turn and U-turn vehicles only
        """
        self.length = length
        self.speed_limit = speed_limit
        self.vehicles = []
        self.approach_direction = "Unknown"
        self.lane_type = lane_type  # Determines which vehicle movements are allowed here
        self.virtual_queue_count = 0  # Overflow vehicles that couldn't enter due to capacity

    def update_vehicles_physics(self, dt=1.0, stop_line=None, is_red_func=None):
        """
        Update the physical positions and velocities of all vehicles on the lane.

        :param dt: Time step duration in seconds.
        :param stop_line: Position coordinate (meters) where vehicles must stop at a red light.
        :param is_red_func: Callable that takes a Vehicle and returns True if the signal is red for it.
        :return: List of vehicles that exited the lane in this time step.
        """
        # Sort vehicles by position in descending order (front vehicle first)
        self.vehicles.sort(key=lambda x: x.position, reverse=True)

        for i, car in enumerate(self.vehicles):
            target_acc = 1.5  # Default mild acceleration (m/s^2)

            # Check whether this specific vehicle sees a red light
            is_red = is_red_func(car) if is_red_func else False

            # A: Car-following logic (only applies when there is a vehicle ahead)
            if i > 0:
                front_car = self.vehicles[i - 1]
                gap = front_car.back_position - car.position

                # Anti-collision: if the gap is too small, force a full stop
                if gap <= 1.0:
                    car.speed = 0.0
                    target_acc = 0.0
                # Safe following distance: decelerate when too close
                elif gap < 8.0:
                    target_acc = -3.0

            # B: Red-light logic (only for the lead vehicle with no car in front)
            elif is_red and stop_line is not None:
                distance_to_stop = stop_line - car.position

                # Approaching the stop line: brake to stop before it
                if 0.0 < distance_to_stop < 20.0:
                    safe_dist = max(distance_to_stop, 0.1)
                    target_acc = -(car.speed ** 2) / (2 * safe_dist)
                # Already crossed the stop line but light is red: force stop
                elif distance_to_stop <= 0.0:
                    car.speed = 0.0
                    target_acc = 0.0

            car.acceleration = target_acc

            # Update continuous physics position and speed
            car.move_continuous(dt)

            # Enforce speed limit
            if car.speed > self.speed_limit:
                car.speed = self.speed_limit

        # Separate vehicles that remain on the lane from those that have exited
        staying_vehicles = []
        leaving_vehicles = []

        for car in self.vehicles:
            if car.position < self.length:
                staying_vehicles.append(car)
            else:
                leaving_vehicles.append(car)

        self.vehicles = staying_vehicles

        # Vacancies opened up: promote vehicles from virtual queue into physical lane
        freed_spaces = len(leaving_vehicles)
        if freed_spaces > 0 and self.virtual_queue_count > 0:
            moved_in = min(freed_spaces, self.virtual_queue_count)
            self.virtual_queue_count -= moved_in
            for _ in range(moved_in):
                from src.entities.vehicle import Vehicle
                new_car = Vehicle(start_pos=0.0, destination="Mayor_Magrath")
                new_car.movement_type = (
                    "straight" if self.lane_type == "straight_right" else "left"
                )
                self.vehicles.append(new_car)

        return leaving_vehicles