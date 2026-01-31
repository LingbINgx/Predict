# 架构与目录

该仓库主要文件夹：

- `data/`：原始与处理后的数据文件（如 `ai4i2020.csv`、`ai4i2020_processed.csv`）。
- `data_preprocess/`：数据预处理的 Notebook。
- `models/`：模型封装（`model.py` 包含 `BaseMod` 封装器）。
- `utils/`：工具函数与主动学习实现（`activeLearn.py`, `wraps.py`, `pt_feature.py` 等）。
- `train/`：训练/实验脚本（例如 `al.py` 演示主动学习流程）。
- `pictures/`：输出图片示例。
- `docs/`：本说明文档所在目录。

主要模块责任划分：

- 数据加载/预处理：`data_preprocess/`、`utils/pt_feature.py`
- 模型封装与评估：`models/model.py`（`BaseMod`）
- 主动学习：`utils/activeLearn.py`（`BasicActiveLearner`、`instant_launch`）
