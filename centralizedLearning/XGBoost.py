# XGBoost (Extreme Gradient Boosting) 极端梯度提升
'''
XGBoost 全称：Extreme Gradient Boosting（极端梯度提升）

作者：陈天奇（Tianqi Chen）在2014年提出

本质：是对 GBDT 的工程优化和增强版

核心改进：
    1. 正则化防止过拟合
    2. 二阶导数优化（比GBDT的一阶更精准）
    3. 并行计算（特征分裂时可以并行）
    4. 缺失值自动处理
    5. 剪枝策略

与 GBDT 的区别：
    GBDT：用一阶导数（梯度）
    XGBoost：用一阶导数 + 二阶导数（Hessian矩阵）

    更精准地找到最优分裂点
'''

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV       # 分层K折交叉验证，网格搜索
from sklearn.utils import class_weight                                  # 类权重，用来平衡权重

# 1. 数据准备
def dm01_data_split():
    df = pd.read_csv('./centralizedLearning/data/红酒品质分类.csv')

    # df.info()

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1] - 3

    # print(x)
    # print(Counter(y))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 把数据集，标签拼接到一起，写到文件中 axis = 1 表示按列拼接
    pd.concat([x_train, y_train], axis=1).to_csv('./centralizedLearning/data/红酒品质分类_train.csv', index=False)
    pd.concat([x_test, y_test], axis=1).to_csv('./centralizedLearning/data/红酒品质分类_test.csv', index=False)

# 2. 模型训练
def dm02_train_model():
    # 读取训练集和测试集
    train_data = pd.read_csv('./centralizedLearning/data/红酒品质分类_train.csv')
    test_data = pd.read_csv('./centralizedLearning/data/红酒品质分类_test.csv')

    # 提取特征数据和标签数据
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]    # 最后一列

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]


    # 创建模型对象
    estimator = xgb.XGBClassifier(
        learning_rate=0.1,           # 学习率
        max_depth=3,                 # 树的最大深度
        n_estimators=100,            # 树的数量
        random_state=42,             # 随机种子
        objective='multi:softmax',   # 多分类问题
    )

    # 平衡权重
    sample_weights = class_weight.compute_sample_weight('balanced', y_train)

    # 模型训练
    estimator.fit(x_train, y_train, sample_weight = sample_weights)

    # 模型预测
    y_pred = estimator.predict(x_test)

    # 模型评估
    print('准确率：', accuracy_score(y_test, y_pred))

    # 模型保存
    # 没有model文件夹，创建model文件夹
    joblib.dump(estimator, './centralizedLearning/model/红酒品质分类.pkl')
    print('模型保存成功！')


# 测试模型 
def dm03_use_model():
    # 加载模型
    # estimator = joblib.load('./centralizedLearning/model/红酒品质分类.pkl')

    # 读取测试集
    train_data = pd.read_csv('./centralizedLearning/data/红酒品质分类_train.csv')
    test_data = pd.read_csv('./centralizedLearning/data/红酒品质分类_test.csv')

    # 提取特征数据和标签数据
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]


    # 创建模型
    estimator = xgb.XGBClassifier(
        learning_rate=0.1,           # 学习率
        max_depth=3,                 # 树的最大深度
        n_estimators=100,            # 树的数量
        random_state=42,             # 随机种子
        objective='multi:softmax',   # 多分类问题
    )

    # 创建网格搜索，交叉验证 分层采样数据
    params = {
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [50, 100, 150]
    }

    # 创建分层采样对象
    # n_splits：分成几份
    # shuffle：是否打乱数据
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator, params, cv=cv, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # 获取最优参数
    best_params = grid_search.best_params_
    print('最优参数：', best_params)

    # 获取最优模型
    best_estimator = grid_search.best_estimator_
    # print('最优模型：', best_estimator)

    # 获取最优评分
    best_score = grid_search.best_score_
    print('最优评分：', best_score)

    # 模型预测
    y_pred = best_estimator.predict(x_test)
    print('准确率：', accuracy_score(y_test, y_pred))


# dm02_train_model()
dm03_use_model()

