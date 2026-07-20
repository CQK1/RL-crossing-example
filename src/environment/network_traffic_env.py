import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.environment.map import TrafficMap
from src.generators.traffic_generator import TrafficGenerator
from src.data_reader import TrafficDataReader

try:
    from src.generators.poisson_process import InhomogeneousPoissonProcess
except ImportError:
    from src.generators.traffic_generator import InhomogeneousPoissonProcess

class NetworkTrafficEnv(gym.Env):
    def __init__(self, data_file_path=None):
        super(NetworkTrafficEnv, self).__init__()

        self.dt = 1.0          # 步进时间为 1 秒
        self.time_step = 0     # 记录当前运行的总秒数 (Simulation Time)

        # ---------------------------------------------------------
        # 第一步：数据与数学引擎初始化 (Data & Mathematical Engine)
        # ---------------------------------------------------------
        if data_file_path is None:
            data_file_path = "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx"
            if not os.path.exists(data_file_path):
                data_file_path = os.path.join("data", "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx")
        
        self.data_reader = TrafficDataReader(data_file_path)
        self.traffic_data = self.data_reader.load_data()
        
        self.poisson_engine = InhomogeneousPoissonProcess(self.traffic_data, cyclic=True)
        self.traffic_generator = TrafficGenerator(rate_model=self.poisson_engine)

        # ---------------------------------------------------------
        # 第二步：物理引擎与拓扑组装 (Physics & Map Topology)
        # ---------------------------------------------------------
        self.traffic_map = TrafficMap()
        self.controlled_nodes = ["Mayor_Magrath"]
        self.num_nodes = 1
        
        # 动作空间：1个路口，4个离散动作 (NS直行, NS左转, EW直行, EW左转)
        self.action_space = spaces.MultiDiscrete([4])
        # 状态空间：5个特征 (当前相位ID + 四个方向停止等待的车辆/行人总数)
        self.observation_space = spaces.Box(low=0, high=500, shape=(5,), dtype=np.float32)

        # 建立中心节点
        self.traffic_map.add_intersection("Mayor_Magrath", 0.0, 0.0)
        
        # 定义东南西北 4 个发车源(Spawn) 和 4 个驶出点(Exit)
        spawns = {"North": (0, 200), "South": (0, -200), "East": (200, 0), "West": (-200, 0)}
        
        for dir_name, (x, y) in spawns.items():
            spawn_node = f"{dir_name}_Spawn"
            exit_node = f"{dir_name}_Exit"
            
            self.traffic_map.add_intersection(spawn_node, x, y)
            self.traffic_map.add_intersection(exit_node, x * 2, y * 2) 
            
            # 驶入路口的车道 (Speed limit 15 m/s ≈ 54 km/h)
            self.traffic_map.add_line(spawn_node, "Mayor_Magrath", speed_limit=15.0)
            # 驶离路口的车道
            self.traffic_map.add_line("Mayor_Magrath", exit_node, speed_limit=15.0)

    def get_state(self):
        """
        观测状态提取：将当前路口的排队情况汇总成一维向量供强化学习使用。
        """
        obs = []
        intersection = self.traffic_map.intersections["Mayor_Magrath"]
        ns_straight_count, ns_left_count = 0, 0
        ew_straight_count, ew_left_count = 0, 0

        for lane in intersection.incoming_lanes:
            is_ns_lane = lane.approach_direction in ["North", "South"]
            
            # 统计停止线前排队的物理车辆数量 (速度为0即视为排队)
            for car in lane.vehicles:
                if car.speed == 0.0:
                    is_left_turn = str(car.destination).endswith("_left")
                    if is_ns_lane:
                        if is_left_turn: ns_left_count += 1
                        else: ns_straight_count += 1
                    else:
                        if is_left_turn: ew_left_count += 1
                        else: ew_straight_count += 1

        # 构建状态数组：[当前相位, 南北直行排队, 南北左转排队, 东西直行排队, 东西左转排队]
        obs.extend([
            intersection.current_phase_index,
            ns_straight_count, 
            ns_left_count,
            ew_straight_count, 
            ew_left_count
        ])
        return np.array(obs, dtype=np.float32)

    def calculate_reward(self):
        """
        奖励函数：优化目标是最小化整个路口的延误时间 (Delay)。
        """
        total_penalty = 0.0
        intersection = self.traffic_map.intersections["Mayor_Magrath"]
        for lane in intersection.incoming_lanes:
            total_penalty += sum(car.waiting_time for car in lane.vehicles if car.speed == 0.0)
        return -total_penalty

    def step(self, action_array):
        """
        核心物理与时间步进
        """
        # 1. 信号灯动作下发
        action = action_array[0]
        self.traffic_map.intersections["Mayor_Magrath"].apply_action(action, dt=self.dt)

        # 2. 调用制造工厂生成实体
        new_entities_dict = self.traffic_generator.generate_entities(float(self.time_step))
        
        # 3. 将生成的实体注入到对应的物理车道中
        for lane in self.traffic_map.lanes:
            # 只有指向路口中心的车道才接收发车源的流量
            if lane.to_node_id == "Mayor_Magrath":
                direction = lane.approach_direction
                entities_to_add = new_entities_dict.get(direction, [])
                if entities_to_add:
                    # 【核心修改点】：限制单车道最大积压车辆数为 40 辆。
                    # 当车道满了之后不强制塞车，防止 CPU 处理上千辆车导致运算性能崩溃卡死
                    if len(lane.vehicles) < 40:
                        lane.vehicles.extend(entities_to_add)
        
        # 将清空缓冲移到整个循环外面，确保多条车道在分配完车辆前，发车数据不会被提前抹掉
        for direction in new_entities_dict.keys():
            new_entities_dict[direction] = []

        # 4. 驱动物理沙盒时间流逝
        self.traffic_map.step(dt=self.dt)

        # 5. 计算反馈信号
        observation = self.get_state()
        reward = self.calculate_reward()
        
        # 时间推进
        self.time_step += 1
        
        # 终止条件：每天有 24 小时 = 86400 秒
        terminated = False 
        truncated = self.time_step >= 86400 
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 将仿真时间重置为凌晨 00:00:00 (0秒)
        self.time_step = 0
        
        # 清空物理地图上的所有残留车辆
        for lane in self.traffic_map.lanes:
            lane.vehicles.clear() 
            
        # 重置路口相位到默认状态
        for inter in self.traffic_map.intersections.values():
            inter.current_phase_index = 0
            inter.current_phase = inter.phases[0]
            inter.phase_timer = 0.0
            inter.reset_stats()
            
        return self.get_state(), {}