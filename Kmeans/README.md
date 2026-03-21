# K-Means 聚类学习笔记

## 概述

本文件夹包含 **K-Means 聚类算法** 和 **朴素贝叶斯分类器** 的学习代码及相关数据。

---

## 文件说明

### 代码文件

| 文件 | 说明 |
|------|------|
| `K-means.py` | K-Means 聚类算法完整代码 |
| `朴素贝叶斯.py` | 朴素贝叶斯分类器代码 |

### 数据文件 (`data/`)

| 文件 | 说明 |
|------|------|
| `customers.csv` | 客户数据（200条），包含性别、年龄、年收入、消费评分 |
| `书籍评价.csv` | 书籍评价数据（13条），用于朴素贝叶斯文本分类 |
| `stopwords.txt` | 中文停用词表（1700+ 停用词） |

---

## K-Means 算法

### 算法步骤

1. 初始化 K 个中心点
2. 计算每个样本到每个中心点的距离
3. 将每个样本分配到距离最近的中心点
4. 更新每个中心点的位置（取该簇所有样本的均值）
5. 重复步骤 2-4，直到中心点位置不再变化
6. 输出每个样本所属的聚类中心

### 评估方法

| 方法 | 说明 | 评判标准 |
|------|------|----------|
| **SSE**（肘部法） | 均方误差，考虑内聚程度 | 越小越好 |
| **SC**（轮廓系数） | 考虑内聚和耦合程度 | 越大越好 |
| **CH**（Calinski-Harabasz） | 综合评估 | 越大越好 |

### 核心参数

- `n_clusters`: 聚类数量 K
- `max_iter`: 最大迭代次数
- `random_state`: 随机种子，确保可复现

### 实战案例：客户分群

基于客户的 **年收入** 和 **消费评分** 进行聚类分析，将客户分为 5 个群体，可用于精准营销。

---

## 朴素贝叶斯分类器

### 核心思想

利用贝叶斯定理，通过概率进行分类：
- 计算给定特征条件下每个类别出现的概率
- 选择概率最大的类别作为预测结果

### 算法特点

- **朴素**：假设各特征之间相互独立
- **拉普拉斯平滑**：防止概率为 0 的问题

### 文本分类流程

1. 读取文本数据
2. 使用 jieba 分词
3. 去除停用词
4. CountVectorizer 向量化
5. 训练 MultinomialNB 模型
6. 评估预测结果

---

## 使用方法

### K-Means 聚类

```python
from sklearn.cluster import KMeans

estimator = KMeans(n_clusters=5, random_state=42)
y_pred = estimator.fit_predict(X)
```

### 朴素贝叶斯

```python
from sklearn.naive_bayes import MultinomialNB

estimator = MultinomialNB()
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
```
