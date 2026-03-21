'''
    朴素贝叶斯算法：
        利用概率值分类，数一数给定特征时每个类别出现了多少次，算出概率，选最大的
        朴素：特征独立
        拉普拉斯平滑：防止概率为0

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import CountVectorizer     # 文本向量化
from sklearn.naive_bayes import MultinomialNB                   # 朴素贝叶斯
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 读取数据
df = pd.read_csv('./Kmeans/data/书籍评价.csv', encoding='gbk')
# df.info()

df['labels'] = np.where(df['评价'] == '好评', 1, 0)  # 好评为1，差评为0

# df.info()
# print(df[:5])

y = df['labels']
# print(jieba.lcut('这本书真的很好看，值得推荐')) # 分词

# 对用户的评论进行切词
comment_list = [','.join(jieba.lcut(comment)) for comment in df['内容'] ]
# print(comment_list[:5])

# 读取停用词表，删除停用词
with open('./Kmeans/data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read()
    stopwords = stopwords.splitlines()
    stopwords = list(set([line.strip() for line in stopwords]))
    print(stopwords[:5])

# 向量化文本
#* 参数：停用词列表
transfer = CountVectorizer(stop_words=stopwords)

# 统计词频矩阵 先训练 后转化 再转数组
x = transfer.fit_transform(comment_list).toarray()
# print(x)

# 查看词汇表（所有不重复的词）
print(transfer.get_feature_names_out())

# 查看词频矩阵
# 矩阵形状：(评论数, 词数) = (13, 37)
# 含义：13条评论经过分词、去除停用词后的结果
# 每行代表一条评论，每列代表一个词
# 矩阵中的值表示该词在该评论中出现的次数
# print(x)

# 切分训练集和测试集
x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

# 数据多的时候用更合适的切分方法
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 训练模型
estimator = MultinomialNB()
estimator.fit(x_train, y_train)

# 预测
y_pred = estimator.predict(x_test)
print('模型预测结果', y_pred)
# 模型评估
print('准确率：', accuracy_score(y_test, y_pred))







