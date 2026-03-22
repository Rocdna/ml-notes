# 电力负荷预测模型训练模块

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, root_mean_squared_error
import joblib

from utils.log import Logger

logger = Logger('train')
from utils.common import preprocess_data

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


# 加载数据
def load_data():
    """训练电力负荷预测模型"""
    # 1. 加载并预处理数据
    return preprocess_data()

# 查看数据的整体分布情况
def ana_data(data):
    '''
    查看数据的整体分布情况
    负荷整体分布
    各小时平均负荷趋势，负荷在一天中的变化
    各月平均负荷趋势，负荷在一年中的变化
    工作日与周末的平均负荷情况，工作日与周末的负荷是否有区别
    :param data: 输入数据
    :return: None
    '''

    ana_data = data.copy()
    # 创建画布
    fig = plt.figure(figsize=(15, 25), constrained_layout=True)

    # 添加子图
    # 这会创建一个 4行1列 的子图布局，当前子图占据第 1个位置（即最上方）。
    ax1 = fig.add_subplot(411)
    ax1.hist(ana_data["power_load"], bins=100)
    ax1.set_title("Power Load Distribution")
    ax1.set_xlabel("Power Load (kW)")
    ax1.set_ylabel("Frequency")

    # 新增加一列，当 Hour 列
    # 从时间列中提取小时
    ana_data['hour'] = ana_data['time'].dt.hour
    
    # 根据小时分组计算平均值
    hourly = ana_data.groupby('hour', as_index=False)['power_load'].mean().reset_index()
    ax2 = fig.add_subplot(412)
    # 绘制折线图
    ax2.plot(hourly['hour'], hourly['power_load'])
    ax2.set_title("Hourly Average Power Load")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Power Load (kW)")
    ax2.set_xticks(range(24))
    

    # 从时间中提取月份
    ana_data['month'] = ana_data['time'].dt.month
    # 根据月份分组计算平均值
    monthly = ana_data.groupby('month', as_index=False)['power_load'].mean().reset_index()
    ax3 = fig.add_subplot(413)
    ax3.plot(monthly['month'], monthly['power_load'])
    ax3.set_title("Monthly Average Power Load")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Power Load (kW)")
    ax3.set_xticks(range(1, 13))


    # 从时间中提取工作日
    # 0=周一, 1=周二, ..., 5=周六, 6=周日
    ana_data['weekday'] = ana_data['time'].dt.weekday

    # True=周末, False=工作日
    ana_data['is_weekend'] = ana_data['weekday'].isin([5, 6])

    # 根据工作日分组计算平均值
    # 工作日
    weekday_hourly = ana_data[~ana_data['is_weekend']].groupby('hour')['power_load'].mean()
    # 周末
    weekend_hourly = ana_data[ana_data['is_weekend']].groupby('hour')['power_load'].mean()
    ax4 = fig.add_subplot(414)

    ax4.plot(weekday_hourly.index, weekday_hourly.values, label='Workday', marker='o')
    ax4.plot(weekend_hourly.index, weekend_hourly.values, label='Weekend', marker='s')
    ax4.set_title("Workday and Weekend Average Power Load")
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Power Load (kW)")
    ax4.legend()
    ax4.set_xticks(range(24))

    # 保存图片到文件
    plt.savefig(PROJECT_ROOT / "data" / "analysis" / "data_distribution.png", dpi=300)
    # plt.show()

    return ana_data

# 3. 特征工程
def feature_engineering(data, logger = logger):
    '''
    特征工程
    提取小时、月份特征
    提取出相近时间窗口中的负荷值
    提取昨日同时刻的负荷值
    删除空的样本值
    整理时间特征

    :param data: 输入数据
    :param logger: 日志记录器
    :return: 特征工程后的数据
    '''
    # 提取小时、月份特征
    # 提取出相近时间窗口中的负荷值
    # 提取昨日同时刻的负荷值
    # 删除空的样本值
    # 整理时间特征
    
    f_data = data.copy()
    f_data['hour'] = f_data['time'].dt.hour
    f_data['month'] = f_data['time'].dt.month

    # 热编码处理 hour 和 month
    f_data = pd.get_dummies(f_data, columns=['hour', 'month'])

    # 提取出相近时间窗口中的负荷特征
    load_1h_data = f_data['power_load'].shift(1)
    load_2h_data = f_data['power_load'].shift(2)
    load_3h_data = f_data['power_load'].shift(3)
    
    load_3h_df = pd.concat([load_1h_data, load_2h_data, load_3h_data], axis=1)
    load_3h_df.columns = ['前1小时', '前2小时', '前3小时']
    f_data = pd.concat([f_data, load_3h_df], axis=1)

    # 昨日同时刻的负荷值
    f_data['yesterday_time'] = f_data['time'] - pd.Timedelta(days=1)
    # 把所有的 日期 和 负荷拼接成字典，方便查找
    time_load_dict = dict(zip(f_data['time'], f_data['power_load']))
    
    # 获取昨日的负荷
    f_data['yesterday_load'] = f_data['yesterday_time'].map(time_load_dict)

    # 处理昨日负荷的空值，直接删除
    f_data = f_data.dropna(subset=['yesterday_load'])
    
    # 整理时间特征并返回，把字段提取出来
    f_data_col = f_data.columns.tolist()
    f_data_col.remove('time')
    f_data_col.remove('yesterday_time')
    f_data_col.remove('power_load')

    return f_data, f_data_col


# 4. 训练模型 评估模型 保存模型
def train_model(data, featuress, logger = logger):
    '''
    训练模型
    :param data: 输入数据
    :param featuress: 特征列
    :param logger: 日志记录器
    :return: 训练好的模型
    '''
    # 获取数据集
    x = data[featuress]
    y = data['power_load']

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 网格搜索和交叉验证得出 最优参数：{'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}
    estimator = XGBRegressor(
        learning_rate=0.1, 
        max_depth=6, 
        n_estimators=200
    )
    estimator.fit(x_train, y_train)

    # 评估
    y_pred = estimator.predict(x_test)
    print(f'模型在测试机上的均方误差：{mean_squared_error(y_test, y_pred)}')
    print(f'模型在测试机上的均方根误差：{root_mean_squared_error(y_test, y_pred)}')
    print(f'模型在测试机上的平均绝对误差：{mean_absolute_error(y_test, y_pred)}')
    print(f'模型在测试机上的平均绝对百分误差：{mean_absolute_percentage_error(y_test, y_pred)}')

    # 保存模型
    joblib.dump(estimator, MODEL_DIR / "xgb_model.pkl")
    logger.info(f"模型已保存到：{MODEL_DIR / 'xgb_model.pkl'}")

    return estimator


if __name__ == '__main__':
    df = load_data()
