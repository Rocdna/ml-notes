# 决策树

## 数据说明

本项目使用的数据来自 **Kaggle 泰坦尼克号生存预测比赛**。

### 数据来源

- 训练集: `titanic_train.csv`
- 测试集: `titanic_test.csv`

### 获取方式

1. **Kaggle 官网下载**:
   - 访问 https://www.kaggle.com/competitions/titanic
   - 注册/登录后进入 Data 页面下载

2. **直接使用**:
   - Kaggle Titanic 数据集是经典机器学习入门数据集
   - 包含乘客信息（舱位、性别、年龄等）和生存状态

### 数据字段

| 字段 | 说明 |
|------|------|
| PassengerId | 乘客ID |
| Survived | 是否生存 (0=否, 1=是) |
| Pclass | 舱位等级 (1=一等, 2=二等, 3=三等) |
| Name | 姓名 |
| Sex | 性别 |
| Age | 年龄 |
| SibSp | 船上兄弟姐妹/配偶数量 |
| Parch | 船上父母/子女数量 |
| Ticket | 票号 |
| Fare | 票价 |
| Cabin | 舱位号 |
| Embarked | 登船港口 (S/C/Q) |

## 文件说明

- `decision_tree.py` - 分类决策树（泰坦尼克号生存预测）
- `decision_tree_regressor.py` - 回归决策树（简单数值预测）
- `decision_tree_note.md` - 学习笔记
