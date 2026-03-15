## 线性回归

# 一元线性回归 
# h = wx + b
# 其中 w 是权重，b 是偏置，h 是预测值，x 是输入特征
# 多元线性回归
# h = w1x1 + w2x2 + ... + wnxn + b
# 可用矩阵表示


# 案例演示线性回归API

from sklearn.linear_model import LinearRegression
import numpy as np

# 生成数据
x_train = np.array([[160], [166], [172], [174], [180], [182], [188], [190], [195], [200]])
y_train = np.array([50, 55, 60, 62, 68, 70, 75, 78, 82, 85])
x_test = np.array([[172]])
y_test = np.array([60])

# 模型训练
estimator = LinearRegression()
estimator.fit(x_train, y_train)

# 线性回归模型，查看权重，偏置
print(f'权重：{estimator.coef_}, 偏置：{estimator.intercept_}')

# 模型预测
y_pred = estimator.predict(x_test)
print(f'预测值：{y_pred}, 真实值：{y_test}')