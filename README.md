### 该数据集包含 10000 个数据点，以行的形式存储，每列包含 14 个特征。

- UID：唯一标识符，范围从 1 到 10000
- product ID：由字母 L、M 或 H 组成，分别代表低（占所有产品的 50%）、中（占 30%）和高（占 20%）三种产品质量等级，并包含一个特定于该等级的序列号。
- air temperature [K]：采用随机游走过程生成，随后归一化至标准偏差为 2 K，约为 300 K。
- process temperature [K]：使用归一化为标准偏差为 1 K 的随机游走过程生成，加上空气温度加 10 K。
- rotational speed [rpm]：根据 2860 W 的功率计算得出，并叠加正态分布噪声。
- torque [Nm]：扭矩值呈正态分布，中心距为 40 Nm，标准差为 10 Nm，没有负值。
- tool wear [min]：质量等级 H/M/L 会使加工过程中使用的刀具磨损时间分别增加 5/3/2 分钟。

“机器故障”标签表示，在此特定数据点中，机器是否出现以下任何故障模式的故障。
机器故障由五种独立的故障模式组成。

- 刀具磨损失效（tool wear failure/TWF）：刀具将在随机选择的 200 至 240 分钟之间的刀具磨损时间点发生失效或需要更换（在我们的数据集中为 120 次）。此时，刀具已被更换 69 次，失效 51 次（随机分配）。
- 散热失效（heat dissipation failure/HDF）：如果空气温度与工艺温度之差低于 8.6 K，且刀具转速低于 1380 rpm，则散热会导致工艺失效。共有 115 个数据点符合此情况。
- 功率故障 (power failure/PWF)：扭矩与转速（单位为弧度/秒）的乘积等于该过程所需的功率。如果该功率低于 3500 瓦或高于 9000 瓦，则过程失败，这种情况在我们的数据集中出现了 95 次。
- 过应变失效 (overstrain failure/OSF)：对于 L 型产品，如果刀具磨损和扭矩的乘积超过 11,000 minNm（M 型为 12,000 minNm，H 型为 13,000 minNm），则工艺因过应变而失效。98 个数据点均符合此规律。
- 随机故障（random failures/RNF）：每个过程都有 0.1% 的概率发生故障，而与过程参数无关。这种情况仅出现在 5 个数据点上，远低于我们数据集中 10,000 个数据点的预期数量。

如果上述故障模式中至少有一种成立，则流程失败，“机器故障”标签将被设置为 1。因此，机器学习方法无法得知是哪种故障模式导致了流程失败。

---

# 机器故障预测性维护分析报告

## 1. 数据加载与探索

首先加载数据集并初步观察数据结构以及分布情况

**代码片段：**

```python
def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

data = read_data('./ai4i2020.csv')
# 前两列是编号(UDI, Product ID)，中间6列是特征，最后5列是目标
data = data.iloc[:, 2:]
```

## 2. 特征分布可视化与标准化

为了使模型训练更稳定，对数值型特征进行了可视化检查和归一化/标准化处理。

### 2.1 特征分布

定义了一个 `pt_feature` 类来绘制特征的直方图并拟合正态分布曲线。

**代码片段：**

```python
class pt_feature():
    def __init__(self, data, ...):
        # 初始化绘图布局
        ...
    def plot_feature(self, feature_name):
        # 绘制直方图与正态拟合曲线
        ...
```

**解释：**

- 该类用于批量展示 `Air temperature`, `Process temperature`, `Rotational speed`, `Torque` 等特征的分布，帮助识别数据偏态或异常值。

### 2.2 数据标准化

针对不同类型的物理量采用了不同的缩放策略

**代码片段：**

```python
def numeric(x): # 标准化
    return (x - x.mean()) / (x.std())

def uniform(x): # 归一化
    return (x - x.min()) / (x.max() - x.min())

ns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
data[ns] = numeric(data[ns])
data["Tool wear [min]"] = uniform(data["Tool wear [min]"])
```

**解释：**

- **标准化 (Z-score)**: 应用于温度、转速和扭矩，使其均值为 0，方差为 1
- **归一化 (Min-Max)**: 应用于 `Tool wear [min]`（工具磨损），将其缩放到 [0, 1] 区间。

---

### 2.3 特征工程

**代码片段：**

```python
data = pd.get_dummies(data) # 独热编码
features = data.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
labels = data['Machine failure']
```

**解释：**

- `get_dummies`: 将非数值型的类别特征转换为数值型的 One-Hot 编码。
- **标签分离**: 将具体的故障类型移除，仅保留 `Machine failure` 作为预测目标。

---

## 3. 不平衡学习

由于故障样本远少于正常样本，即使模型全部预测“正常”也能达到 97%的正确率。

**代码片段：**

```python
# 逻辑回归
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced'
)

# 决策树
clf = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=5,
    random_state=42,
    min_samples_leaf=10,
)
```

**解释：**

- `class_weight='balanced'`: 增加少数类（故障）的权重，让模型更加关注故障样本
- 结果评估使用了 `Precision-Recall Curve`，这对于不平衡数据集比单纯的 Accuracy 更有参考价值。

**结果**

```
#逻辑回归
Accuracy: 0.8174
Confusion Matrix:
 [[3952  885]
 [  28  135]]
Report:
               precision    recall  f1-score   support

         0.0       0.99      0.82      0.90      4837
         1.0       0.13      0.83      0.23       163

    accuracy                           0.82      5000
   macro avg       0.56      0.82      0.56      5000
weighted avg       0.96      0.82      0.87      5000
```

- 模型对正类非常敏感，误报率极高。

```
#决策树
[[4455  382]
 [  13  150]]
              precision    recall  f1-score   support

         0.0       1.00      0.92      0.96      4837
         1.0       0.28      0.92      0.43       163

    accuracy                           0.92      5000
   macro avg       0.64      0.92      0.69      5000
weighted avg       0.97      0.92      0.94      5000
```

- 相比于逻辑回归，precision, recall, f1 上升了，同时误报率也下降了。

---

## 4. 小样本学习

模拟极端情况，仅使用 **1%** 的数据进行训练

### 4.1 逻辑回归模型

**代码片段：**

```python
model = LogisticRegression(
    class_weight={0:1, 1:3},
    max_iter=1000
)
model.fit(train_features, train_labels)
y_prob = model.predict_proba(test_features)[:, 1]
y_pred = (y_prob > 0.3).astype(int)
```

**结果**

```
Accuracy: 0.9165656565656566
Confusion Matrix:
 [[8895  671]
 [ 155  179]]
Report:
               precision    recall  f1-score   support

         0.0       0.98      0.93      0.96      9566
         1.0       0.21      0.54      0.30       334

    accuracy                           0.92      9900
   macro avg       0.60      0.73      0.63      9900
weighted avg       0.96      0.92      0.93      9900
```

**解释：**

- 调整阈值 `(y_prob > 0.3)` 对概率大于 0.3 的判断为 1

### 4.2 集成模型构建=

采用了投票集成策略

**代码片段：**

```python
# 划分数据集：99% 测试，仅 1% 训练
train_features, test_features, train_labels, test_labels = train_test_split(..., test_size=0.99, ...)

# 定义基模型，手动设置类别权重
lr = LogisticRegression(class_weight={0:1, 1:4}, max_iter=1000)
svm = SVC(kernel='rbf', class_weight={0:1, 1:4}, probability=True)
rf = RandomForestClassifier(n_estimators=300, class_weight={0:1, 1:3}, random_state=42)

# 投票
voting = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
    voting='soft',
    weights=[0.45, 0.35, 0.2]
)
voting.fit(train_features, train_labels)
```

**解释：**

- 手动设置权重`{0:1, 1:4}` 告诉模型故障样本的重要性是正常样本的 4 倍
- 对结果进行加权平均

```
Accuracy: 0.9392929292929293
Confusion Matrix:
 [[9126  440]
 [ 161  173]]
Report:
               precision    recall  f1-score   support

         0.0       0.98      0.95      0.97      9566
         1.0       0.28      0.52      0.37       334

    accuracy                           0.94      9900
   macro avg       0.63      0.74      0.67      9900
weighted avg       0.96      0.94      0.95      9900
```

- 样本数量变少后，recall 有所下降，当时 f1 和 precision 提升了

### 4.3 阈值优化

使用默认的 0.5 阈值可能不是最优的，因此通过 F1 Score 寻找最佳判定阈值。

**代码片段：**

```python
precision, recall, thresholds = precision_recall_curve(test_labels, y_prob)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = np.argmax(f1)
best_th = thresholds[best_idx]
print(best_th, precision[best_idx], recall[best_idx], f1[best_idx])
```

**解释：**

- 通过遍历所有可能的阈值，计算对应的 F1 分数
- 选择 F1 分数最高的阈值作为最终模型的判定标准（使用了 `y_prob > 0.30`）

---

## 5. 结论

1.  **特征处理**: 对不同物理属性分别进行标准化和归一化，有助于模型收敛。
2.  **不平衡处理**: `class_weight='balanced'` 参数在 Logistic Regression 和 Decision Tree 中均能有效提升对故障样本的召回率。
3.  **小样本策略**: 在仅有 1% 训练数据的情况下，单模型往往表现不稳定。通过集成三种差异较大的模型（LR, SVM, RF）并进行加权软投票，显著提升了模型的鲁棒性。
