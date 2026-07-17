import random
from typing import Any
from src.entities.pedestrian import Pedestrian
from src.entities.vehicle import Vehicle

class TrafficGenerator:
    def __init__(self, rate_model: Any):
        """
        Traffic Generator Engine.
        
        This class acts as the "factory" that creates physical vehicle/pedestrian instances.
        It is decoupled from the mathematical logic of *when* to spawn them.
        
        :param rate_model: A mathematical model instance (e.g., InhomogeneousPoissonProcess) 
                           that provides a method to query the arrival probability $\lambda(t)$ 
                           for any given time step and direction.
        """
        # 依赖注入：注入数学模型引擎 (配方师)
        self.rate_model = rate_model

        # 映射：物理上的四大进口 (North, South, East, West) 
        # 对应 Excel 数据集（数学模型）中具体的流通量列名
        self.movements_mapping = {
            "North": {
                "straight": "north_thru",
                "left": "north_left",
                "right": "north_right",
                "u_turn": "north_u_turn",
                "pedestrian": "north_peds_cw"  # 简化：南北向合并为一个行人通道
            },
            "South": {
                "straight": "south_thru",
                "left": "south_left",
                "right": "south_right",
                "u_turn": "south_u_turn",
                "pedestrian": "south_peds_cw"
            },
            "East": {
                "straight": "east_thru",
                "left": "east_left",
                "right": "east_right",
                "u_turn": "east_u_turn",
                "pedestrian": "east_peds_cw"
            },
            "West": {
                "straight": "west_thru",
                "left": "west_left",
                "right": "west_right",
                "u_turn": "west_u_turn",
                "pedestrian": "west_peds_cw"
            }
        }

    def generate_entities(self, time_in_seconds: float):
        """
        Generates physical vehicles and pedestrians based on the probabilities 
        provided by the external rate_model at the current simulation time.
        
        :param time_in_seconds: Current global simulation time.
        :return: A dictionary containing lists of generated entities for each approach direction.
                 Format: {"North": [Vehicle, Pedestrian, ...], "South": [...], ...}
        """
        new_entities = {"North": [], "South": [], "East": [], "West": []}
        
        # 遍历四个入口方向
        for direction, movements in self.movements_mapping.items():
            
            # 1. 生成机动车 (Vehicles)
            for intent, column_name in movements.items():
                if intent == "pedestrian":
                    continue  # 行人单独处理
                
                # 核心逻辑：向外部数学模型查询当前时间、当前转向的生成概率 (\lambda)
                # 你可以在这里一键切换为 get_rate_interpolated 来让车流更加平滑
                probability = self.rate_model.get_rate_interpolated(time_in_seconds, column_name)
                
                # 掷骰子模拟泊松到达
                if random.random() < probability:
                    # 组装目的地标签，匹配 intersection.py 里的 is_movement_allowed 格式
                    # 例如：intent 为 "left"，目的地为 "Mayor_Magrath_left"
                    dest = f"Mayor_Magrath_{intent}" if intent != "straight" else "Mayor_Magrath"
                    
                    car = Vehicle(start_pos=0.0, destination=dest)
                    new_entities[direction].append(car)
            
            # 2. 生成行人 (Pedestrians)
            ped_column = movements.get("pedestrian")
            if ped_column:
                ped_prob = self.rate_model.get_rate_interpolated(time_in_seconds, ped_column)
                if random.random() < ped_prob:
                    pedestrian = Pedestrian(start_pos=0.0)
                    # 将行人也暂时混入列表中，后续 Environment 在分发时需要区别对待
                    # 或者你可以在物理引擎中扩展一个专属的斑马线排队区
                    new_entities[direction].append(pedestrian)
                    
        return new_entities