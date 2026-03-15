# 欠拟合  过拟合
# L1 L2 解决过拟合问题
# L1 范数  向量中各元素的绝对值之和
# L2 范数  向量中各元素的平方和的平方根

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV

def dm01_under_fitting():
    np.random.seed(23)

    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    # 数据预处理，把x轴数据转换成二维数组
    X = x.reshape(-1, 1)

    print(f'x轴数据：{X[:5]}, y轴数据：{y[:5]}')


    # 模型训练
    estimator = LinearRegression()
    estimator.fit(X, y)

    # 模型预测
    y_pred = estimator.predict(X)
    print(f'预测值：{y_pred[:5]}, 真实值：{y[:5]}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pred)}')
    print(f'均方根误差：{root_mean_squared_error(y, y_pred)}')
    print(f'平均绝对误差：{mean_absolute_error(y, y_pred)}')

    # 绘图，散点图显示真实数据，线图显示预测数据
    plt.scatter(x, y, color='blue')
    plt.plot(x, y_pred, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# 正好拟合
def dm02_perfect_fitting():
    # 生成数据
    np.random.seed(23)

    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    print(f'特征：{x[:5]}')
    print(f'标签：{y[:5]}')

    # 数据预处理，把x轴数据转换成二维数组
    X = x.reshape(-1, 1)
    print(f'处理后的数据：{X[:5]}')

    X2 = np.hstack([X, X ** 2])
    print(f'处理后的数据：{X2[:5]}')


    # 模型训练
    estimator = LinearRegression()
    estimator.fit(X2, y)

    # 模型预测
    y_pred = estimator.predict(X2)
    print(f'预测值：{y_pred[:5]}, 真实值：{y[:5]}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pred)}')
    print(f'均方根误差：{root_mean_squared_error(y, y_pred)}')
    print(f'平均绝对误差：{mean_absolute_error(y, y_pred)}')

    # 绘图，散点图显示真实数据，线图显示预测数据
    plt.scatter(x, y, color='blue')

    # plt.plot(x, y_pred, color='red')

    # np.sort(x) 对x轴数据进行排序，
    # y_pred[np.argsort(x)] 根据排序后的x轴数据重新排列y_pred，使其与x轴数据对应
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.show()


# 过拟合
def dm03_over_fitting():
    # 生成数据
    np.random.seed(23)

    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    # 数据预处理，把x轴数据转换成二维数组
    X = x.reshape(-1, 1)

    # 模拟过拟合，增加特征维度，增加模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])

    print(f'特征：{X3[:5]}')

    # 模型训练
    estimator = LinearRegression()
    estimator.fit(X3, y)

    # 模型预测
    y_pred = estimator.predict(X3)
    print(f'预测值：{y_pred[:5]}, 真实值：{y[:5]}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pred)}')
    print(f'均方根误差：{root_mean_squared_error(y, y_pred)}')
    print(f'平均绝对误差：{mean_absolute_error(y, y_pred)}')

    # 绘图，散点图显示真实数据，线图显示预测数据
    plt.scatter(x, y, color='blue')
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# L1 正则化  Lasso回归
#t L1 惩罚系数 让部分权重为0，实现特征选择
# L2 正则化  Ridge回归
#t L2 惩罚系数 让权重尽可能小，但不为0，适合处理多重共线性问题

# L1正则化
def dm04_lasso_regularization():
    # 生成数据
    np.random.seed(23)

    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    # 数据预处理，把x轴数据转换成二维数组
    X = x.reshape(-1, 1)

    # 模拟过拟合，增加特征维度，增加模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])

    print(f'特征：{X3[:5]}')

    # 创建 Lasso 回归模型，alpha: 正则化强度，默认为1.0，值越大正则化强度越大
    estimator = Lasso(alpha=0.2)   
    # 模型训练
    estimator.fit(X3, y)

    # 模型预测
    y_pred = estimator.predict(X3)
    print(f'预测值：{y_pred[:5]}, 真实值：{y[:5]}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pred)}')
    print(f'均方根误差：{root_mean_squared_error(y, y_pred)}')
    print(f'平均绝对误差：{mean_absolute_error(y, y_pred)}')

    # 绘图，散点图显示真实数据，线图显示预测数据
    plt.scatter(x, y, color='blue')
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# L2 正则化
def dm05_ridge_regularization():
    # 生成数据
    np.random.seed(23)

    x = np.random.uniform(-3, 3, 100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100)

    # 数据预处理，把x轴数据转换成二维数组
    X = x.reshape(-1, 1)

    # 模拟过拟合，增加特征维度，增加模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])

    print(f'特征：{X3[:5]}')

    # 创建 Ridge 回归模型，alpha: 正则化强度，默认为1.0，值越大正则化强度越大
    estimator = Ridge(alpha=100)   
    # 模型训练
    estimator.fit(X3, y)

    # 模型预测
    y_pred = estimator.predict(X3)
    print(f'预测值：{y_pred[:5]}, 真实值：{y[:5]}')

    # 模型评估
    print(f'均方误差：{mean_squared_error(y, y_pred)}')
    print(f'均方根误差：{root_mean_squared_error(y, y_pred)}')
    print(f'平均绝对误差：{mean_absolute_error(y, y_pred)}')

    # 绘图，散点图显示真实数据，线图显示预测数据
    plt.scatter(x, y, color='blue')
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# dm01_under_fitting()
# dm02_perfect_fitting()
# dm03_over_fitting()

# 用 L1，L2正则解决过拟合
# dm04_lasso_regularization()
dm05_ridge_regularization()

