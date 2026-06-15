# src/data_reader.py
import pandas as pd
import os

class TrafficDataReader:
    def __init__(self, file_path):
        """
        初始化数据读取器
        :param file_path: 文件的相对或绝对路径
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
        自动识别格式并清洗数据
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"找不到数据文件: {self.file_path}")

        try:
            # 核心：因为报了 0x89 错误，说明大概率是二进制 Excel，我们直接用 read_excel
            # sheet_name=0 表示读取第一个工作表
            df = pd.read_excel(self.file_path, skiprows=11, names=self.columns, sheet_name=0)
        except Exception:
            # 万一以后真的转成了纯文本 CSV，保留一个降级备选方案
            df = pd.read_csv(self.file_path, skiprows=11, names=self.columns)
        
        # 清洗数据
        df['start_time'] = pd.to_datetime(df['start_time'])
        df = df.fillna(0)
        
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
    # 根据你控制台里报错的那个真实名字填写
    csv_filename = "Mayor Magrath Drive & 5 Avenue S_Binned_20260524170346-1.xlsx"
    
    # 兼容处理：检查它是否在 data 目录下
    if not os.path.exists(csv_filename) and os.path.exists(f"data/{csv_filename}"):
        csv_filename = f"data/{csv_filename}"
        
    try:
        reader = TrafficDataReader(csv_filename)
        traffic_df = reader.load_data()
        
        print("🎉 数据成功绕过编码异常，解析成功！")
        print("-" * 50)
        print("数据表行列尺寸:", traffic_df.shape)
        print("-" * 50)
        print("前 5 行 Lethbridge 本地交通流明细：")
        print(traffic_df[["start_time", "north_thru", "south_thru", "east_thru", "west_thru"]].head())
        
    except Exception as e:
        print(f"❌ 读取依然失败: {e}")