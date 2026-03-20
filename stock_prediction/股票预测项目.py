"""
股票预测项目：预测明天涨跌
=============================

项目流程：
1. 数据获取
2. 特征工程（构建技术指标）
3. 数据预处理
4. 多种算法对比
5. 模型评估与可视化

作者：机器学习笔记
日期：2026-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 数据获取 ==========
def get_stock_data():
    """
    获取股票数据
    这里用模拟数据演示，实际可用akshare获取真实数据
    """
    try:
        import akshare as ak
        # 获取上证指数数据
        df = ak.stock_zh_index_daily(symbol="sh000001")
        df = df.tail(500)  # 取最近500天
        df.to_csv('./股票预测项目/data/stock_data.csv', index=False)
        print("数据获取成功！")
        return df
    except:
        # 如果没有akshare，生成模拟数据
        print("使用模拟数据演示...")
        np.random.seed(42)
        n = 500
        data = {
            'date': pd.date_range('2024-01-01', periods=n),
            'open': np.random.uniform(3000, 3500, n),
            'high': np.random.uniform(3050, 3600, n),
            'low': np.random.uniform(2950, 3450, n),
            'close': np.random.uniform(3000, 3500, n),
            'volume': np.random.uniform(1e9, 5e9, n)
        }
        df = pd.DataFrame(data)
        # 添加趋势
        trend = np.cumsum(np.random.randn(n) * 20)
        df['close'] = df['close'] + trend
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 50, n)
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 50, n)
        return df

# ========== 2. 特征工程 ==========
def create_features(df):
    """
    构建技术指标特征
    """
    df = df.copy()

    # 基础价格特征
    df['price_change'] = df['close'].pct_change()  # 涨跌幅
    df['high_low_ratio'] = df['high'] / df['low']  # 波动幅度

    # 移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()

    # MA交叉信号
    df['ma5_ma10_cross'] = (df['ma5'] - df['ma10']) / df['ma10']
    df['ma5_ma20_cross'] = (df['ma5'] - df['ma20']) / df['ma20']

    # RSI (相对强弱指标)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 布林带
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # KDJ指标
    low14 = df['low'].rolling(window=14).min()
    high14 = df['high'].rolling(window=14).max()
    df['kdj_k'] = 100 * (df['close'] - low14) / (high14 - low14)
    df['kdj_k'] = df['kdj_k'].ewm(span=3, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(span=3, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    df['kdj_cross'] = (df['kdj_k'] - df['kdj_d'])

    # 成交量特征
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']

    # 目标变量：明天涨跌 (1=涨, 0=跌)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # 删除NaN
    df = df.dropna()

    return df

# ========== 3. 数据预处理 ==========
def preprocess_data(df):
    """
    特征选择和数据分割
    """
    # 选择特征列
    feature_cols = [
        'price_change', 'high_low_ratio',
        'ma5_ma10_cross', 'ma5_ma20_cross',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'kdj_cross',
        'volume_ratio'
    ]

    X = df[feature_cols]
    y = df['target']

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # 时序数据不打乱
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

# ========== 4. 模型训练与对比 ==========
def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    训练多种模型并对比效果
    """
    # 定义模型
    models = {
        '逻辑回归': LogisticRegression(max_iter=1000),
        '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        '梯度提升': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = {}

    print("=" * 60)
    print("模型训练与对比")
    print("=" * 60)

    for name, model in models.items():
        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 评估
        accuracy = accuracy_score(y_test, y_pred)

        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        print(f"\n{name}:")
        print(f"  测试集准确率: {accuracy:.4f}")
        print(f"  交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return results

# ========== 5. 评估最佳模型 ==========
def evaluate_best_model(results, y_test, model_name='XGBoost'):
    """
    评估最佳模型
    """
    print("\n" + "=" * 60)
    print(f"最佳模型详细评估: {model_name}")
    print("=" * 60)

    model_info = results[model_name]
    y_pred = model_info['y_pred']
    y_pred_proba = model_info['y_pred_proba']

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['跌', '涨']))

    # 计算各项指标
    tn, fp, fn, tp = cm.ravel()
    print(f"真正例 (TP): {tp}")
    print(f"真负例 (TN): {tn}")
    print(f"假正例 (FP): {fp}")
    print(f"假负例 (FN): {fn}")

    return cm, y_pred_proba

# ========== 6. 可视化 ==========
def plot_results(results, y_test, cm, feature_cols):
    """
    可视化结果
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. 模型准确率对比
    ax1 = fig.add_subplot(2, 2, 1)
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    cv_means = [results[n]['cv_mean'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracies, width, label='测试集准确率', color='steelblue')
    bars2 = ax1.bar(x + width/2, cv_means, width, label='交叉验证准确率', color='lightgreen')

    ax1.set_ylabel('准确率')
    ax1.set_title('各模型准确率对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45)
    ax1.legend()
    ax1.set_ylim(0.4, 0.8)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    # 2. 混淆矩阵热力图
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.figure.colorbar(im, ax=ax2)
    ax2.set(xticks=[0, 1], yticks=[0, 1],
            xticklabels=['跌', '涨'], yticklabels=['跌', '涨'],
            title='混淆矩阵', ylabel='真实标签', xlabel='预测标签')

    # 在格子中添加数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # 3. ROC曲线
    ax3 = fig.add_subplot(2, 2, 3)
    for name, info in results.items():
        fpr, tpr, _ = roc_curve(y_test, info['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    ax3.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    ax3.set_xlabel('假正率 (FPR)')
    ax3.set_ylabel('真正率 (TPR)')
    ax3.set_title('ROC曲线对比')
    ax3.legend(loc='lower right')

    # 4. 特征重要性 (用XGBoost)
    ax4 = fig.add_subplot(2, 2, 4)
    xgb_model = results['XGBoost']['model']
    importance = xgb_model.feature_importances_
    indices = np.argsort(importance)[::-1]

    ax4.barh(range(len(feature_cols)), importance[indices])
    ax4.set_yticks(range(len(feature_cols)))
    ax4.set_yticklabels([feature_cols[i] for i in indices])
    ax4.set_xlabel('特征重要性')
    ax4.set_title('XGBoost 特征重要性排名')
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig('./股票预测项目/result/模型评估结果.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存到: ./股票预测项目/result/模型评估结果.png")
    plt.show()

# ========== 7. 策略回测 ==========
def backtest(results, y_test, model_name='XGBoost'):
    """
    简单回测策略
    """
    print("\n" + "=" * 60)
    print("策略回测")
    print("=" * 60)

    y_pred_proba = results[model_name]['y_pred_proba']
    y_pred = results[model_name]['y_pred']

    # 策略1：模型预测
    correct = (y_pred == y_test).sum()
    total = len(y_test)
    print(f"\n策略1 - 模型预测:")
    print(f"  准确率: {correct/total:.4f}")

    # 策略2：随机猜测作为对比
    np.random.seed(42)
    random_correct = np.random.choice([0, 1], size=total) == y_test
    print(f"  随机猜测准确率: {random_correct.mean():.4f}")

    # 胜率分析
    print(f"\n胜率分析:")
    print(f"  预测涨的正确率: {(y_pred[y_test==1]==1).mean():.4f}" if (y_pred==1).sum() > 0 else "  无预测涨")
    print(f"  预测跌的正确率: {(y_pred[y_test==0]==0).mean():.4f}" if (y_pred==0).sum() > 0 else "  无预测跌")

# ========== 主函数 ==========
def main():
    print("=" * 60)
    print("股票预测项目：预测明天涨跌")
    print("=" * 60)

    # 1. 获取数据
    print("\n[1] 获取股票数据...")
    df = get_stock_data()
    print(f"数据形状: {df.shape}")

    # 2. 特征工程
    print("\n[2] 构建技术指标特征...")
    df = create_features(df)
    print(f"特征数量: {df.shape[1]}")
    print(f"样本数量: {df.shape[0]}")

    # 3. 数据预处理
    print("\n[3] 数据预处理...")
    X_train, X_test, y_train, y_test, feature_cols = preprocess_data(df)
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    print(f"特征: {feature_cols}")

    # 4. 模型训练与对比
    print("\n[4] 训练多种模型...")
    results = train_and_compare_models(X_train, X_test, y_train, y_test)

    # 5. 评估最佳模型
    cm, y_pred_proba = evaluate_best_model(results, y_test, 'XGBoost')

    # 6. 策略回测
    backtest(results, y_test, 'XGBoost')

    # 7. 可视化
    print("\n[7] 生成可视化图表...")
    plot_results(results, y_test, cm, feature_cols)

    print("\n" + "=" * 60)
    print("项目完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
