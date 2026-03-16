# 电信客户流失案例
"""


"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def dm01_data_preprocess():
    # 数据
    data = pd.read_csv('./logical/data/churn.csv')

    # churn 和 gender 是字符串，需要转换成数值类型，需要进行one-hot编码
    data = pd.get_dummies(data, columns=['Churn', 'gender'])
    # 处理后的数据集
    # data.info()

    # 删除冗余特征，列删除 axis=1，行删除 axis=0
    data.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)  
    data.rename(columns={'Churn_Yes': 'Flag'}, inplace=True)

    data.info()

# 数据可视化
def dm02_data_visualization():
    churn_df = pd.read_csv('./logical/data/churn.csv')
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    churn_df.rename(columns={'Churn_Yes': 'Flag'}, inplace=True)

    # 统计流失客户和非流失客户的数量
    churn_counts = churn_df['Flag'].value_counts()
    print(f'流失客户数量：{churn_counts}, 非流失客户数量：{churn_counts}')

    # 数据可视化，绘制柱状图，Contract_Month 是否开月度会员
    # 注意：countplot 的第一个参数应为 data=churn_df
    sns.countplot(data=churn_df, x='Contract_Month', hue='Flag')
    plt.show()


# 逻辑回归算法的模型训练
def dm03_logistic_regression():
    # 获取数据 + 数据处理
    churn_df = pd.read_csv('./logical/data/churn.csv')
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
    churn_df.rename(columns={'Churn_Yes': 'Flag'}, inplace=True)

    # 提取特征列和标签列
    x = churn_df[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = churn_df['Flag']    # False -> 不流失  True -> 流失
    # 数据划分
    # 划分训练集和测试集，测试集占比30%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    # 创建逻辑回归模型并训练
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 在测试集上进行预测
    y_pred = estimator.predict(x_test)

    print(y_pred[:5])

    # 输出模型评估指标
    print("准确率：", accuracy_score(y_test, y_pred))
    print("精确率：", precision_score(y_test, y_pred))
    print("召回率：", recall_score(y_test, y_pred))
    print("F1分数：", f1_score(y_test, y_pred))
    print("分类报告：\n", classification_report(y_test, y_pred))

    # macro avg 宏平均，数据均衡的情况下
    # weighted avg 加权平均，数据不均衡的情况下








# dm01_data_preprocess()
# dm02_data_visualization()
dm03_logistic_regression()

