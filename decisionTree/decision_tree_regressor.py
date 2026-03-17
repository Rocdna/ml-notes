#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# 读取数据
x_train = np.array(list(range(1,11))).reshape(-1, 1)
y_train = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

# 创建决策树回归模型
estimator1 = LinearRegression()
estimator2 = DecisionTreeRegressor(max_depth=3)
estimator3 = DecisionTreeRegressor(max_depth=5)

estimator1.fit(x_train, y_train)
estimator2.fit(x_train, y_train)
estimator3.fit(x_train, y_train)


# 测试集的数据
x_test = np.arange(0, 10, 0.1).reshape(-1, 1)

y_pred1 = estimator1.predict(x_test)
y_pred2 = estimator2.predict(x_test)
y_pred3 = estimator3.predict(x_test)

# 绘图
plt.scatter(x_train, y_train, c='blue', label='train data')
plt.plot(x_test, y_pred1, color='red', label='Linear Regression')
plt.plot(x_test, y_pred2, color='green', label='Decision Tree (depth=3)')
plt.plot(x_test, y_pred3, color='orange', label='Decision Tree (depth=5)')
plt.legend()
plt.show()


