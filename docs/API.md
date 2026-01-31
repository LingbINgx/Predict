# API 摘要

本节列出项目中常用的类与函数（简要说明与示例）。

1) `models/model.py` - `BaseMod`

- 用途：封装 scikit-learn 风格的模型，提供统一的 `fit`、`predict`、`score` 等方法。
- 主要方法：`fit(X,y)`, `predict(X)`, `predict_proba(X)`, `score(X,y)`, `confusion_matrix(X,y)`, `classification_report(X,y)`, `f1_score(X,y)`。

示例：

```python
from models.model import BaseMod
from sklearn.ensemble import RandomForestClassifier

mod = BaseMod(RandomForestClassifier())
mod.fit(X_train, y_train)
acc = mod.score(X_test, y_test)
```

2) `utils/activeLearn.py` - `BasicActiveLearner` & `instant_launch`

- `BasicActiveLearner`:
  - 方法：`load_data(X,y,test_size=0.5,imbalance_handle='None')`, `initialize_data(initial_size)`, `create_learner()`, `active_learning_cycle(n_queries, query_size, threshold)`，`plot_precision_recall_curve(metrics)`。
  - 功能：实现基于 `modAL` 的主动学习循环，并封装不平衡处理（SMOTE/SMOTEENN）。

- `instant_launch(X,y,...)`:
  - 功能：一键式运行主动学习流程，返回 `(metrics, train_instance)`。
  - 常用参数：`estimator`, `query_strategy`, `initial_size`, `n_queries`, `query_size`, `threshold`, `test_size`, `imbalance_handle`。

示例：

```python
from utils.activeLearn import instant_launch
metrics, learner = instant_launch(X, y, n_queries=100)
```
