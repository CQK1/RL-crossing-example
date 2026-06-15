# src/data_reader.py
import pandas as pd
import os

class TrafficDataReader:
    def __init__(self, file_path):
        """
        初始化数据读取器
        :param file_path: Excel 文件的相对或绝对路径
        """
        self.file_path = file_path
        self.data = None
        
        # 预定义 25 列清晰、独立的列名
        self.columns = [
            "start_time",
            # 北侧进口
            "north_right", "north_thru", "north_left", "north_u_turn", "north_peds_cw", "north_peds_ccw",
            # 东侧进口
            "east_right", "east_thru", "east_left", "east_u_turn", "east_peds_cw", "east_peds_ccw",
            # 南侧进口
            "south_right", "south_thru", "south_left", "south_u_turn", "south_peds_cw", "south_peds_ccw",
            # 西侧进口
            "west_right", "west_thru", "west_left", "west_u_turn", "west_peds_cw", "west_peds_ccw"
        ]
    def load_data(self):
            """
            解析并清洗 Excel 数据
            """
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"找不到数据文件: {self.file_path}")

            # 读取 Excel 文件
            df = pd.read_excel(self.file_path, skiprows=11, names=self.columns, engine='openpyxl')
            
            # 1. 核心修复：errors='coerce' 会把无法解析的文本（如 "Single-Unit Trucks"）变成 NaT
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
            
            # 2. 清除所有 start_time 为 NaT 的行（也就是彻底过滤掉末尾的统计摘要行）
            df = df.dropna(subset=['start_time'])
            
            # 3. 将其余流量列的空值填充为 0
            df = df.fillna(0)
            
            # 4. 转换为整型流量
            for col in self.columns[1:]:
                df[col] = df[col].astype(int)
                
            self.data = df
            return self.data

    def get_time_series_data(self):
        if self.data is None:
            self.load_data()
        return self.data


# ==========================================
# 本地测试代码
# ==========================================
if __name__ == "__main__":
    filename = "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx"
    
    # 兼容处理：检查它是否在 data 目录下
    if not os.path.exists(filename) and os.path.exists(f"data/{filename}"):
        filename = f"data/{filename}"
        
    try:
        reader = TrafficDataReader(filename)
        traffic_df = reader.load_data()
        
        print("Excel Read")
        print("-" * 50)
        print("Dimension:", traffic_df.shape)
        print("-" * 50)
        print("First five rows:")
        print(traffic_df[["start_time", "north_thru", "south_thru", "east_thru", "west_thru"]].head())
        
    except Exception as e:
        print(f"Failed to read: {e}")