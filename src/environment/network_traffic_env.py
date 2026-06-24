# src/environment/network_traffic_env.py
from src.environment.map import TrafficMap
from src.generators.traffic_generator import TrafficGenerator

class NetworkTrafficEnv:
    def __init__(self, target_id="Node_A"):
        self.traffic_map = TrafficMap()
        self.traffic_generator = TrafficGenerator()
        self.time_step = 0
        self.target_id = target_id

    def get_state(self):
        """
        收集地图上每一个路口的完整状态
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
        """所有受控路口的总排队惩罚之和，作为协同奖励"""
        total_penalty = 0.0
        for inter_id, intersection in self.traffic_map.intersections.items():
            for lane in intersection.incoming_lanes:
                total_penalty += sum(1.0 for car in lane.vehicles if car.speed == 0.0)
        return -total_penalty

    def step(self, action_dict):
        """
        接收多路口动作字典，执行物理步进
        action_dict: {"Node_A": 1, "Node_B": 0, "Node_C": 0}
        """
        self.time_step += 1

        # 1. 更新所有路口的红绿灯
        for intersection_id, action in action_dict.items():
            if action == 1 and intersection_id in self.traffic_map.intersections:
                self.traffic_map.intersections[intersection_id].toggle_light()

        # 2. 修正车流生成漏洞：新车只允许从没有前驱节点（源头）的车道注入
        for lane in self.traffic_map.lanes:
            # 只有当车道的起点是 "Start_Node" 时，才允许生成新车注入系统
            if lane.from_node_id == "Start_Node":
                new_car = self.traffic_generator.generate_vehicle()
                if new_car:
                    lane.vehicles.append(new_car)
            
        # 3. 驱动整个地图的物理时钟
        self.traffic_map.step(dt=1.0)

        # 4. 多智能体架构：返回完整的、无损的全局状态字典
        state = self.get_state()
        reward = self.calculate_reward()
        done = self.time_step >= 100

        return state, reward, done
    
    def reset(self):
        """真正的多智能体重置：不返回单路口的扁平状态"""
        self.time_step = 0
        
        # 清空所有车道上的车辆
        for lane in self.traffic_map.lanes:
            lane.vehicles.clear() 
            
        # 重置所有路口的红绿灯状态
        for inter in self.traffic_map.intersections.values():
            inter.light_state = 0 
            
        # 这里返回 None 即可，main.py 会自己显式调用 env.get_state() 获取初始状态
        return None