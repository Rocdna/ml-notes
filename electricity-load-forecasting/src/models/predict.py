import pandas as pd
import datetime
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.log import Logger
logger = Logger('predict')

from utils.common import preprocess_data

# 解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 黑体、微软雅黑
plt.rcParams['axes.unicode_minus'] = False 


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "model"


# 预测数据解析特征
def pred_feature_extract(data_dict, time, logger):
    '''
    预测数据解析特征
    :param data_dict: 输入数据字典
    :param time: 预测时间
    :param logger: 日志记录器
    :return: 
    '''

    feature_names = ['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 
                    'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13',
                    'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 
                    'hour_20', 'hour_21', 'hour_22', 'hour_23', 'month_1', 'month_2', 
                    'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 
                    'month_9', 'month_10', 'month_11', 'month_12', '前1小时', '前2小时', 
                    '前3小时', 'yesterday_load']

    # 截取要预测的小时信息
    pred_hour = time.hour
    hour_list = [1 if hour == pred_hour else 0 for hour in range(24)]

    # 截取月份信息
    pred_month = time.month
    month_list = [1 if month == pred_month else 0 for month in range(1, 13)]

    # 解析窗口特征
    last_1h = data_dict.get(time - pd.Timedelta(hours=1), 500)
    last_2h = data_dict.get(time - pd.Timedelta(hours=2), 500)
    last_3h = data_dict.get(time - pd.Timedelta(hours=3), 500)

    # 解析昨日同时刻的负荷值
    yesterday_load = data_dict.get(time - pd.Timedelta(days=1), 500)

    # 合并特征
    feature_list = hour_list + month_list + [last_1h, last_2h, last_3h, yesterday_load]

    # 与训练时的特征保持一致，之后可以开始预测了
    feature_list = pd.DataFrame([feature_list], columns=feature_names)

    return feature_list

# 可视化预测结果
def visualize_pred_result(evaluate_df):
    """
    可视化预测结果
    :param evaluate_df: 包含预测时间、预测负荷、真实负荷的DataFrame
    :return: None
    """
    plt.figure(figsize=(30, 20))
    
    # 创建子图
    ax = plt.subplot()

    # 绘制折线图
    ax.plot(evaluate_df['预测时间'], evaluate_df['预测负荷'], color='red', label='预测负荷')
    ax.plot(evaluate_df['预测时间'], evaluate_df['真实负荷'], color='blue', label='真实负荷')
    ax.set_xlabel('预测时间')
    ax.set_ylabel('负荷值')
    ax.set_title('真实与预测负荷对比')
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.5)
    # 添加图例
    ax.legend(loc='upper left')

    # 保存图片
    plt.savefig(PROJECT_ROOT / "data" / "analysis" / "真实负荷与预测对比结果图.png", dpi=300, bbox_inches='tight')

    plt.show()



# 加载模型预测
def load_model_predict():
    """加载训练好的模型"""
     # 加载数据
    _, df_test = preprocess_data()

    model_path = MODEL_DIR / "xgb_model.pkl"
    model = joblib.load(model_path)

    # 确定要预测的时间段 获取2015年8月1号以后的记录
    pre_times = df_test[df_test['time'] >= '2015-08-01 00:00:00']

    time_load_dict = dict(zip(pre_times['time'], pre_times['power_load']))

    # 定义列表，用于保存预测结果，方便后续进行预测结果评价
    evaluate_list = []

    # 为了模拟实际场景，把要预测时间及以后的负荷掩盖掉，只保存预测时间之前的数据字典
    for pre_time in pre_times['time']: 
        print(f'正在预测 {pre_time} 时间的负荷......')
        time_load_dict_maskd = {k: v for k, v in time_load_dict.items() if k <= pre_time}
        # 提取特征
        feature_df = pred_feature_extract(time_load_dict_maskd, pre_time, logger)
        
        # 利用加载的模型预测
        pred_load = model.predict(feature_df)

        # 真实值
        true_load = time_load_dict[pre_time]

        # 保存预测结果，绘制结果图 
        # 预测时间、预测值、真实值
        evaluate_list.append([pre_time, pred_load[0], true_load])

    # evaluate_list 转换为 DataFrame
    evaluate_df = pd.DataFrame(evaluate_list, columns=['预测时间', '预测负荷', '真实负荷'])

    # 预测结果评价
    print(f'平均绝对误差：{mean_absolute_error(evaluate_df["真实负荷"], evaluate_df["预测负荷"])}')


    # 绘制预测结果图
    visualize_pred_result(evaluate_df)

    return model


if __name__ == '__main__':
    # 加载模型
    model = load_model_predict()
