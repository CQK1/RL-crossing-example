class Vehicle:
    def __init__(self, start_pos = 0):
        self.position = start_pos
        self.waiting_time = 0
    
    def move(self, is_green):
        if is_green:
            self.position += 1
        else:
            self.waiting_time += 1

class Pedestrian:
    def __init__(self, start_pos = 0):
        self.position = start_pos
        self.waiting_time = 0
    
    def move(self, is_green):
        if is_green:
            self.position += 1
        else:
            self.waiting_time += 1