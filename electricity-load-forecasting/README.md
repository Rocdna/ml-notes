# 电力负荷预测项目

使用机器学习方法对电力负荷进行预测，训练 XGBoost 模型，实现多变量单步预测。

## 项目结构

```
electricity-load-forecasting/
├── src/models/
│   ├── train.py      # 模型训练
│   └── predict.py    # 模型预测
├── utils/
│   ├── common.py     # 数据预处理
│   └── log.py        # 日志工具
├── data/
│   ├── raw/          # 原始数据
│   └── analysis/     # 分析结果
├── model/            # 保存的模型
├── log/              # 日志文件
└── README.md
```

## 模块说明

### 1. 数据预处理 (utils/common.py)
- 读取训练集和测试集 CSV 文件
- 时间格式转换
- 按时间排序
- 数据去重

### 2. 模型训练 (src/models/train.py)
- **数据分析**：负荷分布、小时/月份趋势、工作日与周末对比
- **特征工程**：
  - 小时、月份热编码
  - 前1/2/3小时负荷特征
  - 昨日同时刻负荷特征
- **模型训练**：XGBoost 回归
- **评估指标**：MSE、RMSE、MAE、MAPE

### 3. 模型预测 (src/models/predict.py)
- 加载训练好的模型
- 滚动预测（模拟真实场景）
- 预测结果可视化与误差评估

## 模型参数

| 参数 | 值 |
|------|-----|
| learning_rate | 0.1 |
| max_depth | 6 |
| n_estimators | 200 |

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python src/models/train.py
```

### 运行预测
```bash
python src/models/predict.py
```

## 依赖

- pandas
- numpy
- matplotlib
- xgboost
- scikit-learn
- joblib
- loguru
