# adaptive boosting       自适应增强
#* adaboost 预测正确，权重减小；预测错误，权重增大

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder          # 标签编码器
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


df_wine = pd.read_csv('./centralizedLearning/data/wine.csv')

# df_wine.info()

# print(df_wine['Class label'])    # 葡萄总类有三种

# 从标签 Class label  中过滤掉 1 类别
df_wine = df_wine[df_wine['Class label'] != 1]


x = df_wine[['Alcohol', 'Hue']]     # 酒精  色泽
y = df_wine['Class label']          # 标签列

# print(x[:5])
# print(y[:5])


# 通过标签编码器把标签列转化为数值列
le = LabelEncoder()
y = le.fit_transform(y)
print(y[:5])

# stratify 参数保证训练集和测试集中各类别的比例与原始数据集相同
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# 模型训练
# 默认单一决策树
estimator1 = DecisionTreeClassifier(max_depth=3)

estimator1.fit(x_train, y_train)

y_pred1 = estimator1.predict(x_test)

print('单一决策树正确率：', accuracy_score(y_test, y_pred1))


# adaboost  -->  集成学习，CART树，200颗
# 参数一：决策树基分类器
# 参数二：树的数量
# 参数三：学习率
# 参数四：算法（该参数已过时），SAMME.R SAMME.R算法是SAMME算法的改进版，采用概率分类，效果更好
estimator2 = AdaBoostClassifier(estimator=estimator1, n_estimators=300, learning_rate=0.1)

estimator2.fit(x_train, y_train)

y_pred2 = estimator2.predict(x_test)

print('adaboost正确率：', accuracy_score(y_test, y_pred2))


