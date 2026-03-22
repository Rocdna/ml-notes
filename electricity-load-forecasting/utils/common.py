import pandas as pd
import numpy as np
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 对数据做预处理  --> 时间格式化，按照时间升序排序，且数据去重
# 数据集 data 目录下 train.csv 文件 --> 拆分训练集 和 测试集
# test train.csv  --> 模拟项目上线后真实数据

# 定义函数，对数据做预处理
def preprocess_data():
    df_train = pd.read_csv(DATA_DIR / "raw" / "train.csv")
    df_test = pd.read_csv(DATA_DIR / "raw" / "test.csv")
    # 时间格式化，转为：YYYY-MM-DD HH:mm:ss 格式
    df_train['time'] = pd.to_datetime(df_train['time'])
    df_test['time'] = pd.to_datetime(df_test['time'])
    # 按照时间升序排序
    df_train = df_train.sort_values(by='time')
    df_test = df_test.sort_values(by='time')
    # 去重
    df_train = df_train.drop_duplicates(subset=['time'], keep='first')
    df_test = df_test.drop_duplicates(subset=['time'], keep='first')
    return df_train, df_test

if __name__ == '__main__':
    print('hello common!')
