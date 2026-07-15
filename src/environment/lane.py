from src.entities.vehicle import Vehicle

class Lane:
    def __init__(self, length=150.0, speed_limit=40):
        """
        Initialize the Lane segment representing a single road path.
        """
        self.length = length
        self.speed_limit = speed_limit
        self.vehicles = []
        self.approach_direction = "Unknown"

    def update_vehicles_physics(self, dt=1.0, stop_line=None, is_red_func=None):
        """
        Update the physical positions and velocities of all vehicles on the lane.
        
        :param dt: Time step duration.
        :param stop_line: Position coordinate of the intersection stop line.
        :param is_red_func: Callable checking if the traffic light is red for a specific vehicle.
        :return: List of vehicles that exited the lane in this step.
        """
        # Sort vehicles by position in descending order (closest to the end of the lane first)
        self.vehicles.sort(key=lambda x: x.position, reverse=True)

        for i, car in enumerate(self.vehicles):
            target_acc = 1.5 
            
            # Check the traffic light status for this specific vehicle
            is_red = is_red_func(car) if is_red_func else False

            # A: Physical queue constraint (car-following logic)
            if i > 0:
                front_car = self.vehicles[i-1]
                # Calculate the absolute physical gap between the two vehicles (front vehicle's rear - rear vehicle's front)
                gap = front_car.back_position - car.position
                
                # Anti-collision mechanism: if the gap is too small, force a stop (reflects real physical queue lengths)
                if gap <= 1.0:
                    car.speed = 0.0
                    target_acc = 0.0
                elif gap < 8.0:  # Start decelerating within the safe car-following distance
                    target_acc = -3.0
                    
            # B: If there is no leading vehicle and the current direction faces a red light
            elif is_red and stop_line is not None:
                distance_to_stop = stop_line - car.position
                if 0.0 < distance_to_stop < 20.0:
                    # Begin braking within 20 meters of the stop line
                    safe_dist = max(distance_to_stop, 0.1)
                    target_acc = - (car.speed ** 2) / (2 * safe_dist)
                elif distance_to_stop <= 0.0:
                    # Force a stop if the vehicle crossed the stop line but the light is still red
                    car.speed = 0.0
                    target_acc = 0.0
            
            car.acceleration = target_acc
            
            # 3. Move in continuous physical steps
            car.move_continuous(dt)
            
            # 4. Ensure vehicle speed does not exceed the road's speed limit
            if car.speed > self.speed_limit:
                car.speed = self.speed_limit
                
        # 5. Segment vehicles into those remaining and those exiting the lane
        staying_vehicles = []
        leaving_vehicles = []
        
        for car in self.vehicles:
            if car.position < self.length:
                staying_vehicles.append(car)
            else:
                leaving_vehicles.append(car)
                
        self.vehicles = staying_vehicles
        
        return leaving_vehicles