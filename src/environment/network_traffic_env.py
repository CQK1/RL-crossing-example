# src/environment/network_traffic_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.environment.map import TrafficMap
from src.generators.traffic_generator import TrafficGenerator

class NetworkTrafficEnv:
    def __init__(self, controlled_nodes):
        super(NetworkTrafficEnv, self).__init__()

        self.traffic_map = TrafficMap()
        self.traffic_generator = TrafficGenerator()
        self.time_step = 0
        self.dt = 1.0
        self.controlled_nodes = controlled_nodes

        #generate action spaces
        self.action_space = spaces.Dict({
            node_id: spaces.Discrete(2) for node_id in self.controlled_nodes
        })

        #generate observation spaces
        self.observation_space = spaces.Dict({
            node_id: spaces.Dict({
                "light_state": spaces.Discrete(2),
                "waiting_vehicles": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
            }) for node_id in self.controlled_nodes
        })

    def get_state(self):
        """
        Get the state of each intersection in the map
        """
        state_dict = {}

        for inter_id, intersection in self.traffic_map.intersections.items():
            total_waiting_vehicles = 0
            for lane in intersection.incoming_lanes:
                total_waiting_vehicles += sum(1 for car in lane.vehicles if car.speed == 0.0)

            state_dict[inter_id] = {
                "light_state": intersection.light_state,
                "waiting_vehicles": total_waiting_vehicles
            }
        
        return state_dict

    def calculate_reward(self):
        """All the controlled intersections and cars that are waiting as total sum"""
        total_penalty = 0.0
        for inter_id, intersection in self.traffic_map.intersections.items():
            for lane in intersection.incoming_lanes:
                total_penalty += sum(car.waiting_time for car in lane.vehicles if car.speed == 0.0)
        return -total_penalty

    def step(self, action_dict):
        self.time_step += 1

        # 1. 执行动作
        for intersection_id, action in action_dict.items():
            if intersection_id in self.traffic_map.intersections:
                self.traffic_map.intersections[intersection_id].apply_action(action, dt=self.dt)

        # 2. 车辆生成与物理步进
        for lane in self.traffic_map.lanes:
            if lane.from_node_id == "Start_Node":
                new_car = self.traffic_generator.generate_vehicle()
                if new_car:
                    lane.vehicles.append(new_car)
        
        self.traffic_map.step(dt=self.dt)

        # 3. return reward
        observation = self.get_state()
        reward = self.calculate_reward()
        
        # Terminated 通常指任务成功或失败，Truncated 通常指时间到了
        terminated = False 
        truncated = self.time_step >= 300
        
        info = {}

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        self.time_step = 0
        
        # clear the environment
        for lane in self.traffic_map.lanes:
            lane.vehicles.clear() 
        for inter in self.traffic_map.intersections.values():
            inter.light_state = 0 
            
        # 获取初始状态并确保格式与 observation_space 一致
        observation = self.get_state()
        info = {} # 可以存放额外的调试信息
        
        return observation, info