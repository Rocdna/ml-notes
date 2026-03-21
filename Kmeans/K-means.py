'''
    聚类：无监督学习方法，没有标签，通过相似度进行分类

    K-means算法：
        1. 初始化K个中心点
        2. 计算每个样本到每个中心点的距离
        3. 将每个样本分配到距离最近的中心点
        4. 更新每个中心点的位置
        5. 重复以上步骤，直到中心点位置不再变化
        6. 输出每个样本所属的聚类中心
        7. 可视化聚类结果
        8. 评估聚类结果
        9. 可视化评估聚类结果

    K-means评估方法：
        1. SSE：均方误差，越小越好，表示聚类效果越好
        2. SC：轮廓系数，越大越好，表示聚类效果越好
        3. CH：卡inski-Harabasz系数，越大越好，表示聚类效果越好
        4. 肘方法：根据SSE变化率来选择K值，变化率越小越好，表示聚类效果越好

'''

from sklearn.cluster import KMeans                     # 导入KMeans类 采用质心分类
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs                # 默认按照高斯分布生成数据
from sklearn.metrics import calinski_harabasz_score    # 轮系数评估聚类结果
from sklearn.metrics import silhouette_score              # 轮系数评估聚类结果
import pandas as pd



# 准备数据
# 参数1：样本数量
# 参数2：特征数量
# 参数3：每个簇的中心位置
# 参数4：每个簇的标准差
# 参数5：随机种子
x, y = make_blobs(
    n_samples=1000, 
    n_features=2, 
    centers=[[-2, -2], [0, 0], [1.5, 1.5], [3, 3]],
    cluster_std=[0.6, 0.6, 0.6, 0.6], 
    random_state=42
)

# print(x)
# print(y)

# # 绘制图形
# plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()


# # 创建KMeans模型
# # 参数1：聚类数量
# # 参数2：随机种子
# estimator = KMeans(n_clusters=4, random_state=42)

# # 训练模型
# y_pred = estimator.fit_predict(x)
# print(y_pred)

# # 可视化聚类结果
# plt.scatter(x[:, 0], x[:, 1], c=y_pred)
# plt.show()

# # 评价指标
# # 轮系数评估聚类结果
# print(calinski_harabasz_score(x, y_pred))


# 可视化评估聚类结果
# plt.scatter(x[:, 0], x[:, 1], c=y_pred)
# plt.show()



## SSE + 肘部法   SSE 内聚，考虑内部距离，越小越好

def dm01_sse():
    # 生成数据集
    x, y = make_blobs(
        n_samples=1000, 
        n_features=2, 
        centers=[[-2, -2], [0, 0], [1, 1], [3, 3]],
        cluster_std=[0.6, 0.4, 0.3, 0.6], 
        random_state=42
    )

    sse_list = []

    # for 循环，遍历不同的K值
    for k in range(1, 100):
        # 创建KMeans模型
        # 参数1：聚类数量 参数2：最大迭代次数 参数3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=42)
        # 训练模型
        y_pred = estimator.fit_predict(x)
        # 计算SSE
        sse = estimator.inertia_
        # 打印K值和SSE
        print(f"K={k}, SSE={sse}")

        sse_list.append(sse)

    # print(sse_list)
    # 绘制曲线找到最优K值
    # 修改X轴尺寸
    plt.xticks(range(1, 100, 3))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.grid()
    plt.plot(range(1, 100), sse_list)
    plt.show()


# SC 轮系数评估聚类结果   SC耦合，考虑外部距离，越大越好
def dm02_sc():
    # 生成数据集
    x, y = make_blobs(
        n_samples=1000, 
        n_features=2, 
        centers=[[-2, -2], [0, 0], [1, 1], [3, 3]],
        cluster_std=[0.6, 0.4, 0.3, 0.6], 
        random_state=42
    )

    sse_list = []

    # for 循环，遍历不同的K值
    #* 簇外，至少需要2个簇 
    for k in range(2, 100):
        # 创建KMeans模型
        # 参数1：聚类数量 参数2：最大迭代次数 参数3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=42)
        # 训练模型
        y_pred = estimator.fit_predict(x)
        # 计算SC
        sc = silhouette_score(x, y_pred)
        # 打印K值和SSE
        print(f"K={k}, SC={sc}")
        sse_list.append(sc)

    # print(sse_list)
    # 绘制曲线找到最优K值
    # 修改X轴尺寸
    plt.xticks(range(1, 100, 3))
    plt.xlabel("K")
    plt.ylabel("SC")
    plt.grid()
    plt.plot(range(2, 100), sse_list)
    plt.show()

# CH 卡inski-Harabasz系数评估聚类结果   CH 结合 内聚和耦合，越大越好
def dm03_ch():
    # 生成数据集
    x, y = make_blobs(
        n_samples=1000, 
        n_features=2, 
        centers=[[-2, -2], [0, 0], [1, 1], [3, 3]],
        cluster_std=[0.6, 0.4, 0.3, 0.6], 
        random_state=42
    )

    sse_list = []

    # for 循环，遍历不同的K值
    #* 簇外，至少需要2个簇 
    for k in range(2, 100):
        # 创建KMeans模型
        # 参数1：聚类数量 参数2：最大迭代次数 参数3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=42)
        # 训练模型
        y_pred = estimator.fit_predict(x)
        # 计算CH
        ch = calinski_harabasz_score(x, y_pred)
        # 打印K值和CH
        print(f"K={k}, CH={ch}")
        sse_list.append(ch)

    # print(sse_list)
    # 绘制曲线找到最优K值
    # 修改X轴尺寸
    plt.xticks(range(1, 100, 3))
    plt.xlabel("K")
    plt.ylabel("CH")
    plt.grid()
    plt.plot(range(2, 100), sse_list)
    plt.show()


# 用户分群案例，基于用户的年收入 和 消费指数，相似性分类
def dm04_find_k():
    pd_data = pd.read_csv("./KMeans/data/customers.csv")
    # pd_data.info()

    # 定义 sse_list sc_list 记录不同K值的评估效果
    sse_list = []
    sc_list = []

    # 抽取特征
    x = pd_data[['Annual Income (k$)', 'Spending Score (1-100)']]

    for k in range(2, 20):
        # 创建KMeans模型
        # 参数1：聚类数量 参数2：最大迭代次数 参数3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=42)
        # 训练模型
        y_pred = estimator.fit_predict(x)
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x, y_pred))

    # 绘制曲线找到SC最优K值
    plt.xticks(range(1, 20, 3))
    plt.xlabel("K")
    plt.ylabel("SC")
    plt.grid()
    plt.plot(range(2, 20), sc_list)
    plt.show()

    # 绘制曲线找到SSE最优K值
    plt.xticks(range(1, 20, 3))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.grid()
    plt.plot(range(2, 20), sse_list)
    plt.show()

    # k = 5 为最优K值


# 用户分群案例，基于用户的年收入 和 消费指数，相似性分类
def dm05_train_predit():
    pd_data = pd.read_csv("./KMeans/data/customers.csv")
    # pd_data.info()

    # 抽取特征
    x = pd_data[['Annual Income (k$)', 'Spending Score (1-100)']]

    # 训练模型  5 是测试出来最佳k值
    estimator = KMeans(n_clusters=5, max_iter=100, random_state=42)
    y_pred = estimator.fit_predict(x)

    # 打印预测结果
    # print(y_pred)

    # 绘制5个簇 样本点
    plt.scatter(x.values[y_pred == 0, 0], x.values[y_pred == 0, 1], c='r')
    plt.scatter(x.values[y_pred == 1, 0], x.values[y_pred == 1, 1], c='g')
    plt.scatter(x.values[y_pred == 2, 0], x.values[y_pred == 2, 1], c='b')
    plt.scatter(x.values[y_pred == 3, 0], x.values[y_pred == 3, 1], c='y')
    plt.scatter(x.values[y_pred == 4, 0], x.values[y_pred == 4, 1], c='m')

    # 绘制5个簇 质心

    print(estimator.cluster_centers_)

    plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1], c='k', marker='x', s=100, alpha=0.5)
    plt.title("K-means Clustering")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"])
    plt.show()



# dm01_sse()
# dm02_sc() 
# dm03_ch()
# dm04_find_k()
dm05_train_predit()



## 时间序列预测
# 多变量单步 --> 单变量单步