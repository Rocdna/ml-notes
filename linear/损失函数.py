# 损失函数
# 误差 = 预测值 - 真实值
# 均方误差 = (误差的平方和) / 样本数量
# 求最佳的 k 和 b 使得均方误差最小

# L1 范数  向量中各元素的绝对值之和
# L2 范数  向量中各元素的平方和的平方根
# Lp 范数  向量中各元素的绝对值的 p 次幂和的 p 次根

# 梯度下降

# 什么是梯度
# 单变量中，梯度是某一点切线斜率，有方向则为函数增长最快的方向
# 多变量中，梯度是某一点的偏导数，指向偏导数分量的向量方向

# 下降步骤阈值，当梯度的模小于该值时，停止迭代
# 迭代次数阈值，当迭代次数超过该值时，停止迭代
# 两次迭代之间的损失函数值的差值小于该值时，停止迭代

# 学习率，控制每次更新的步长，过大可能导致震荡，过小可能导致收敛过慢

# 梯度下降法计算  银行贷款
# 损失函数采用均方误差

# 梯度下降算法分类

## 全梯度下降（Full Gradient Descent）FGD
# 每次迭代使用整个训练集计算梯度，更新参数
# 优点：每次迭代都朝着全局最优方向更新参数 
# 缺点：每次迭代计算成本高，尤其是大规模数据集

## 随机梯度下降（Stochastic Gradient Descent）SGD
# 每次迭代随机选择一个样本计算梯度，更新参数
# 优点：每次迭代计算成本低，适合大规模数据集
# 缺点：每次迭代更新方向不稳定，可能导致震荡

## 小批量梯度下降（Mini-batch Gradient Descent）MBGD
# 每次迭代随机选择一个小批量样本计算梯度，更新
# 优点：每次迭代计算成本适中，适合大规模数据集，更新方向相对稳定
# 缺点：需要选择合适的批量大小，过大可能导致计算
# 成本高，过小可能导致震荡

## 随机平均梯度下降（Stochastic Average Gradient Descent）SAG
# 每次迭代随机选择一个样本计算梯度，并将其存储起来，更新参数时使用所有存储的梯度的平均值
# 优点：每次迭代更新方向相对稳定，适合大规模数据集
# 缺点：需要存储所有样本的梯度，内存消耗
# 适合线性模型，非线性模型可能表现不佳


#t 回归模型的评估方法
# 均方误差（Mean Squared Error）MSE
# 平均绝对误差（Mean Absolute Error）MAE
# 均方根误差（Root Mean Squared Error）RMSE

# 线性回归 监督学习 有标签 有分类 且标签连续

# from sklearn.datasets import load_boston              # 波士顿房价数据集 已废弃
from sklearn.datasets import fetch_california_housing   # 加利福尼亚房价数据集
from sklearn.preprocessing import StandardScaler        # 数据标准化
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 线性回归模型
from sklearn.linear_model import SGDRegressor           # 随机梯度下降回归模型
from sklearn.metrics import mean_squared_error          # 均方误差评估指标
from sklearn.linear_model import Ridge, RidgeCV         # 岭回归模型和交叉验证岭回归模型


import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# print(f'数据集特征维度：{data.shape}, 标签维度：{target.shape}')
# print(f'数据集特征示例：\n{data[:5]}, 标签示例：\n{target[:5]}')

# 数据预处理
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=24)


# 特征工程
transfer = StandardScaler()
# 训练模型前对训练数据进行标准化处理，并将同样的变换应用于测试数据
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 模型训练
# fit_intercept: 是否计算截距项，默认为 True
# estimator = LinearRegression(fit_intercept=True)    # 正规方程求解线性回归模型参数

# 随机梯度下降求解线性回归模型参数    
# max_iter: 最大迭代次数，默认为 1000
# tol: 迭代停止的阈值，默认为 1e-3
# learning_rate: 学习率模式，控制每次更新的步长，默认为 'invscaling'，即随着迭代次数增加而逐渐减小
# constant: 固定学习率，
# 固定学习率为 0.01
estimator = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.01, random_state=24)      
estimator.fit(x_train, y_train)

# 打印模型参数
print(f'权重：{estimator.coef_}, 偏置：{estimator.intercept_}')

# 模型预测
y_pred = estimator.predict(x_test)
print(f'预测值：{y_pred[:5]}, 真实值：{y_test[:5]}')

# 模型评估
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差：{mse}')
print(f'均方根误差：{np.sqrt(mse)}')
print(f'平均绝对误差：{np.mean(np.abs(y_test - y_pred))}')