# GBDT boost
'''
Gradient Boosting Decision Tree (梯度 提升决策树) 残差提升树
    是一种集成学习方法，它通过迭代地训练弱学习器（通常是决策树）来提高模型的性能。
    每个新的学习器都是基于前一个学习器的残差进行训练的，从而逐步减少模型的误差。

    通过拟合残差（真实值 - 预测值）来弥补错误。
    每棵树学习前面的错误（残差），多棵树累加，逐步逼近真实值

    -梯度 = 真实值 - 预测值 = 残差

    梯度提升树不再拟合残差，而是利用梯度下降的近似方法，利用损失函数的负梯度作为提升树算法中的残差近似值 


    
    1. 采用所有目标值的均值 作为第一个弱学习器的预测值
    2. 计算每个样本的残差，残差 = 真实值 - 预测值  作为第二个学习器的 目标值
    3. 针对第一个弱学习器，依次计算每个分割点的 最小平方和，找到最佳分割点，
    4. 把上述分割点带入第二个弱学习器，计算它的预测值
'''


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier          # 决策树分类器
from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升树分类器
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV        # 网格搜索


# 1. 数据准备
df = pd.read_csv('./centralizedLearning/data/titanic_train.csv')

x = df[['Pclass', 'Age', 'Sex']].copy()
y = df['Survived']



# 处理缺失列值
x['Age'] = x['Age'].fillna(x['Age'].mean())
x = pd.get_dummies(x)

# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. 模型训练

#* 单个决策树回归器
estimator1 = DecisionTreeClassifier()
estimator1.fit(x_train, y_train)
y_pred1 = estimator1.predict(x_test)
print(f'单个决策树的准确率: {estimator1.score(x_test, y_test)}')
print('-' * 30)

#* 梯度提升树回归器
estimator2 = GradientBoostingClassifier()
estimator2.fit(x_train, y_train)
y_pred1 = estimator2.predict(x_test)

print(f'梯度提升树的准确率: {accuracy_score(y_test, y_pred1)}')
print('-' * 30)


## 针对GBDT模型 进行参数调优
params = {
    'n_estimators': [10, 50, 60, 90],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.1, 0.01, 0.001]
}

estimator3 = GradientBoostingClassifier()
# estimator3.fit(x_train, y_train)
# 网格搜索
grid = GridSearchCV(estimator3, params, cv=3)
grid.fit(x_train, y_train)
y_pred3 = grid.predict(x_test)

print(f'网格搜索后最佳模型: { grid.best_estimator_ }')
print(f'网格搜索后最佳准确率: { grid.best_score_ }')
print(f'网格搜索后最佳参数: { grid.best_params_ }')
