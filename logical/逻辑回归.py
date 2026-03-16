"""
逻辑回归
    有监督学习，有特征，有标签，标签离散（分类）
    适用于二分类问题，扩展到多分类问题

    sigmoid函数：S(x) = 1 / (1 + e^(-x)))
    逻辑回归，概率

    最大似然估计
    似然函数到底在问什么

    不是问"及格率是多少"，而是：
    "如果及格率是 θ，观察到 8及格2不及格 这个具体结果的可能性有多大？"

    逻辑回归原理
    对数损失函数：
        最大似然估计，求出最大可能的概率，损失函数是越小越好，通过最小化损失函数来求解模型参数
        所以需要取负号，变成最小化问题，也是方便梯度下降计算

"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('./logical/data/breast-cancer-wisconsin.csv', na_values=['?'])
# data.info() 

# 替换缺失值
# 参数1：要替换的值
# 参数2：替换成什么值
# 参数3：是否修改原数据，默认False，返回一个新的DataFrame
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)           # 删除包含缺失值的行


# 特征工程（提取，预处理）
x = data.iloc[:, 1:-1]  # 特征  从第一列到倒数第二列
y = data.iloc[:, -1]    # 标签  最后一列
# y = data['Class']     # 效果等同上下
# y = data.Class


# print(f'特征：\n{x[:5]}')
# print(f'标签：\n{y[:5]}')


# 训练集和测试集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 特征预处理
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 模型训练
estimator = LogisticRegression()
estimator.fit(x_train, y_train)

# 模型预测
y_predict = estimator.predict(x_test)
print(f'预测结果：\n{y_predict}, 真实结果：\n{y_test}')

# 模型评估
print(f'预测前评估，正确率：{estimator.score(x_test, y_test)}')
print(f'预测后评估，正确率：{estimator.score(x_test, y_predict)}')
print(f'准确率：{accuracy_score(y_test, y_predict)}')


# 逻辑回归不能用准确率来评估，结果不够精确，适用于不平衡数据集
# 例如：癌症数据集中，阳性样本（患病）远少于阴性样本（健康），如果模型总是预测为阴性，那么准确率可能很高，但模型没有实际意义
# 逻辑回归主要用于 二分类问题
# 评估指标可以使用混淆矩阵、精确率、召回率、F1分数等更适合分类问题的指标来评估模型性能.

## 样本少的当正例
#t 分类评估方法， 混淆矩阵
# 真实值，预测值
# TP（True Positive）：真正例，模型正确预测为正类的样本数量
# TN（True Negative）：真反例，模型正确预测为负类的样本数量
# FP（False Positive）：假正例，模型错误预测为正类的样本
# FN（False Negative）：假反例，模型错误预测为负类的样本数量

#* 精确率（Precision）= TP / (TP + FP) 预测为正类的样本中，真正例的比例
#* 召回率（Recall）= TP / (TP + FN) 实际为正类的样本中，真正例的比例
#* F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率) 精确率和召回率的调和平均数，综合考虑了两者的表现

#* 混淆矩阵  默认使用分类少的当正例

'''
                    预测标签（正例）          预测标签（反例）

真实标签（正例）        真正例（TP）           假反例（FN）

真实标签（反例）        假正例（FP）           真反例（TN）

'''


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# 需求：已知有10个样本，6个恶性，4个良性
# 模型A：预测结果是3个恶性，4个良性
# 模型B：预测结果是5个恶性，5个良性

y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']

y_predict_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']
y_predict_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性']

# 标签
labels = ['恶性', '良性']
dflables = ['恶性（正例）', '良性（反例）']

# 针对于模型A 搭建混淆矩阵
confusion_matrix_A = confusion_matrix(y_train, y_predict_A, labels=labels)
print(f'模型A的混淆矩阵：\n{confusion_matrix_A}')


# 针对于模型B 搭建混淆矩阵
confusion_matrix_B = confusion_matrix(y_train, y_predict_B, labels=labels)
print(f'模型B的混淆矩阵：\n{confusion_matrix_B}')


# 为了让测试结果更清晰，用DataFrame来展示混淆矩阵
confusion_matrix_A_df = pd.DataFrame(confusion_matrix_A, index=dflables, columns=dflables)
print(f'模型A的混淆矩阵（DataFrame展示）：\n{confusion_matrix_A_df}')

confusion_matrix_B_df = pd.DataFrame(confusion_matrix_B, index=dflables, columns=dflables)
print(f'模型B的混淆矩阵（DataFrame展示）：\n{confusion_matrix_B_df}')


# 计算A模型的精确率、召回率、F1分数
# print(f'模型A的精确率、召回率、F1分数：\n{classification_report(y_train, y_predict_A, labels=labels)}')
print(f'模型A的精确率：{precision_score(y_train, y_predict_A, pos_label="恶性")}')
print(f'模型A的召回率：{recall_score(y_train, y_predict_A, pos_label="恶性")}')
print(f'模型A的F1分数：{f1_score(y_train, y_predict_A, pos_label="恶性")}')

# 计算B模型的精确率、召回率、F1分数
print(f'模型B的精确率：{precision_score(y_train, y_predict_B, pos_label="恶性")}')
print(f'模型B的召回率：{recall_score(y_train, y_predict_B, pos_label="恶性")}')
print(f'模型B的F1分数：{f1_score(y_train, y_predict_B, pos_label="恶性")}')



## ROC曲线和AUC值
# ROC曲线：Receiver Operating Characteristic Curve，受试者工作特征曲线
# AUC值：Area Under the Curve，曲线下面积，AUC值越大，模型性能越好
# ROC曲线是通过改变分类阈值来计算不同的TPR（真正率）和FPR（假正率）来绘制的曲线，AUC值则是ROC曲线下面积的大小，AUC值越接近1，模型性能越好，AUC值为0.5表示模型没有区分能力，AUC值小于0.5表示模型性能较差。


print(f'模型A的ROC曲线和AUC值：\n{classification_report(y_train, y_predict_A, labels=labels)}')



