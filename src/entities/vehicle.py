import random

class Vehicle:
    def __init__(self, start_pos = 0, destination = "Node_C"):
        # property of the car
        self.position = float(start_pos) # let this be the position of the head of the car
        self.length = random.uniform(4.0, 5.0)

        self.speed = 0.0 # current speed
        self.acceleration = 0.0 

        self.waiting_time = 0
        # assign destination when car is generated
        self.destination = destination
    
    def move(self, is_green):
        if is_green:
            self.position += 1
        else:
            self.waiting_time += 1

    @property
    def back_position(self):
        """ position of the back of the car """
        return self.position - self.length
    
    def move_continuous(self, dt=1.0):
        self.speed = max(0.0, self.speed + self.acceleration * dt)

        self.position += self.speed * dt

        if self.speed == 0.0:
            self.waiting_time += dt