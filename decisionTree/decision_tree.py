'''
    决策树

    ID3决策树： 信息增益大的特征列来充当节点
        熵：表示随机变量的不确定性，熵越大，不确定性越大，数据的混乱程度越高
        熵的计算公式：H(X) = - ∑ P(x) * log2(P(x))
        解释：X是随机变量，P(x)是随机变量X取值为x的概率，log2是2为底的对数
        
        信息增益：表示划分前后熵的变化，信息增益越大，划分越有效
        信息增益的计算公式：g(D,A) = H(D) - ∑ P(t) * H(Dt)
        解释：D是数据集，A是属性，t是属性A的取值，Dt是数据集D中属性A取值为t的数据子集，P(t)是属性A取值为t的概率，H(Dt)是数据子集Dt的熵

        信息增益 = 划分后熵的减少量
        - 划分前熵 - 划分后熵
        - 增益越大，说明这个特征越好用    

        ID3数的不足：偏向于选择取值多的特征列，容易过拟合

    C4.5决策树： 信息增益率大的特征列来充当节点，惩罚取值多的特征列
        信息增益率 = 信息增益 / 特征熵
        1/特征熵 = 惩罚系数
        特征越多，惩罚系数越小，信息增益率越小


    CART决策树： 回归树和分类树
        基尼指数：表示数据集D的不确定性，基尼指数越小，数据集D的纯度越高
        基尼指数的计算公式：Gini(D) = 1 - ∑ P(t) * P(t)
        解释：D是数据集，t是数据集D的类别，P(t)是数据集D中类别t的概率
        
        基尼值 = 1 - 类别概率的平方和
        基尼指数 = 各分类占比 * 当前分类的基尼值 求和   

        CART回归树： 基尼指数小的特征列来充当节点
        CART分类树： 基尼指数小的特征列来充当节点

        连续值可通过排序，二分法，找到最佳分割点，将连续值转化为离散值
        回归树：预测连续值   使用平方损失

        
    决策树剪枝： 防止过拟合
        是一种防止决策树过拟合的一种正则化方法；提高泛化能力

    剪枝：把子树的节点全部删掉，使用叶子节点来替换
        预剪枝：
            规定树的最大深度
            树生长过程中，提前停止分裂，限制深度，样本数，信息增益阈值
            优点：减少决策树的训练时间，测试时间开销
            缺点：容易欠拟合

        后剪枝：
            树完全成长后，长出来后再剪掉
            优点：泛化能力比预剪枝更好
            缺点：训练时间开销大
'''


# 案例：泰坦尼克号乘客生存预测


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 加载数据
data = pd.read_csv('./decisionTree/data/titanic_train.csv')
data.info()


# 数据预处理
x = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']


x = x.copy()
x['Age'].fillna(x['Age'].mean(), inplace=True)  # 有警告，不让直接更改原数据

# print(x.info())

x['Sex'] = x['Sex'].map({'male': True, 'female': False})

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 模型训练
#* max_depth：树的最大深度，如果为None，则节点会一直扩展，直到所有叶子都是纯净的，或者所有叶子包含少于min_samples_split个样本
estimator = DecisionTreeClassifier(max_depth=10, criterion='entropy')
estimator.fit(x_train, y_train)

# 模型预测
y_pred = estimator.predict(x_test)
print(f'预测值为：{y_pred}')

# 模型评估
print(f'分类评估报告：{classification_report(y_test, y_pred)}')

# 绘制决策树
plt.figure(figsize=(30,20))     # 设置图片大小

#* filled=True，颜色填充
#* feature_names=x.columns, class_names=['0', '1']，设置特征名和类别名
#* max_depth=10，绘制决策树的层数
plot_tree(estimator, filled=True, feature_names=x.columns, max_depth=10, class_names=['0', '1'])  # filled=True，颜色填充

plt.savefig('./decisionTree/data/titanic.png')
plt.show()


