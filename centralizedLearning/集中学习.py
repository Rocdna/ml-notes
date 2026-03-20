
# 集中学习
'''
    通过多个模型的组合形成一个精度更高的模型，解决单一预测

    Bagging:  随机森林
        生成多个不同的训练集，训练多个模型，最终投票/平均得到结果。
    
        流程：
            有放回的抽样，平权投票，并行训练
            数据集，特征列，都随机抽取
    
            随机森林算法：用CART树
            
    Boosting:   AdaBoost GBDT XGBoost LightGBM
         多个"弱分类器"依次学习，串行训练，每次重点关注前面分错的样本，最终把所有弱分类器组合起来得到强分类器。
        
        
        全数据集训练，串行训练，加权投票，后一个模型根据前一个模型分错的样本进行训练。
    
    adaboost：
        预测对了权重降低，预测错了权重增加，权重高的样本在下一轮训练中更重要。

        
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# 加载数据
df = pd.read_csv('./centralizedLearning/data/titanic_train.csv')


# 数据预处理
x = df[[ 'Pclass', 'Age', 'Sex' ]].copy()
y = df['Survived']

# print(x.head())
# print(y.head())

# 缺失值处理
x['Age'] = x['Age'].fillna(x['Age'].mean())
x = pd.get_dummies(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 模型训练 单一决策树
estimator = DecisionTreeClassifier()
estimator.fit(x_train, y_train) 


# 模型预测
y_pred = estimator.predict(x_test)

print('准确率：', accuracy_score(y_test, y_pred))
print('-' * 40)



## 随机森林，多个决策树 --> 默认参数
#* n_estimators: 决策树的数量
#* max_depth: 决策树的最大深度
estimator2 = RandomForestClassifier(n_estimators=100)
estimator2.fit(x_train, y_train) 

y_pred2 = estimator2.predict(x_test)

print('准确率：', accuracy_score(y_test, y_pred2))
print('-' * 40)


## 随机森林，多个决策树 --> 网格搜索
estimator3 = RandomForestClassifier()
# estimator3.fit(x_train, y_train)        # 训练一次
params = {
    'n_estimators': [10, 50, 60, 90],
    'max_depth': [3, 5, 7, 9]
}
grid = GridSearchCV(estimator3, params, cv=3)
grid.fit(x_train, y_train)

y_pred3 = grid.predict(x_test)

print('最佳参数：', grid.best_params_)
print('最佳准确率：', grid.best_score_)


print('获取随机森林模型的准确率：', accuracy_score(y_test, y_pred3))


