from src.entities.vehicle import Vehicle

class Lane:
    def __init__(self, length = 150.0, speed_limit = 40):
        self.length = length
        self.speed_limit = speed_limit
        self.vehicles = []

    def update_vehicles_physics(self, dt = 1.0, stop_line = None, is_red = False):
        """
        Update all the positions of the vehicles on the lane
        stop_line: where car stops
        is_red: whether the current intersection is red
        """

        self.vehicles.sort(key=lambda x: x.position, reverse = True)

        for i, car in enumerate(self.vehicles):
            # 2. acc for follwing the car
            target_acc = 1.5 
            
            # A: if there's more than one car, need to decelarate behind the front car
            if i > 0:
                front_car = self.vehicles[i-1]
                gap = front_car.back_position - car.position
                if gap < 6.0:  # if gap distance is less than 6 meters, begin to decelarate
                    target_acc = -3.0
                    
            # B: there's no car in the front and there's red
            elif is_red and stop_line is not None:
                distance_to_stop = stop_line - car.position
                if 0.0 < distance_to_stop < 20.0:
                    # stop before the line within distance of 20
                    target_acc = - (car.speed ** 2) / (2 * distance_to_stop + 0.1)
            
            car.acceleration = target_acc
            
            # 3. move in continuous movement
            car.move_continuous(dt)
            
            # 4. make sure the speed is less the road speed limit
            if car.speed > self.speed_limit:
                car.speed = self.speed_limit
                
        # 5. remove the car if they are out of the line
        staying_vehicles = []
        leaving_vehicles = []
        
        for car in self.vehicles:
            if car.position < self.length:
                staying_vehicles.append(car)
            else:
                leaving_vehicles.append(car)
                
        # Update vehicles
        self.vehicles = staying_vehicles
        
        return leaving_vehicles

