# NBA比赛预测项目

这是一个使用多任务神经网络预测NBA比赛结果的项目。通过分析球队和球员的统计数据，模型能够预测比赛的胜负以及分差。

## 项目结构

```
prediction_of_NBA_game/
├── main.py                         # 主入口文件（没用别点）
├── predict_games.py                # 预测脚本（可以用来测试你自己的数据）
├── train_multitask_nn.py           # 核心训练代码（ 训练完的日志显示训练集和测试集的准确率
├── requirements.txt                # 项目依赖
├── artifacts/
│   └── multitask_nn/               # 训练产物***
│       ├── feature_columns.joblib  # 特征列信息
│       ├── feature_scaler.joblib   # 特征缩放器
│       ├── margin_scaler.joblib    # 分差缩放器
│       ├── metrics.json            # 评估指标
│       ├── multitask_model.pt      # 训练好的模型
│       └── test_predictions.csv    # 测试集预测结果
└── data_preparation/
    ├── DATA_DICTIONARY.md          # 数据字典
    ├── eda.py                      # 探索性数据分析
    ├── feature_engineering.py      # 特征工程
    ├── fetch_data.py               # 数据获取
    └── data/
        ├── processed/
        │   └── features_2023_24.csv # 处理后的特征数据
        └── raw/
            └── raw_games_2023_24.csv # 原始数据
```

## 安装依赖

确保你已经安装了Python 3.10，然后运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- torch: 深度学习框架
- pandas: 数据处理
- numpy: 数值计算
- scikit-learn: 机器学习工具
- joblib: 模型序列化
- tqdm: 进度条
- matplotlib/seaborn: 数据可视化

## 数据准备

1. 运行数据获取脚本：
```bash
python data_preparation/fetch_data.py
```

2. 进行特征工程：
```bash
python data_preparation/feature_engineering.py
```

处理后的数据将保存在 `data_preparation/data/processed/features_2023_24.csv`。

## 训练模型

运行训练脚本以训练多任务神经网络：

```bash
python train_multitask_nn.py
```

训练过程包括：
- 数据加载和预处理
- 时间顺序的数据分割（训练/验证/测试）
- 特征和标签的缩放
- 模型训练和验证
- 保存模型和预处理器到 `artifacts/multitask_nn/`

## 预测比赛

使用训练好的模型进行预测：

```bash
python predict_games.py
```

可选参数：
- `--input`: 输入特征CSV文件路径（默认：`data_preparation/data/processed/features_2023_24.csv`）
- `--artifacts`: 模型产物目录（默认：`artifacts/multitask_nn`）
- `--output`: 输出预测结果路径（默认：`artifacts/multitask_nn/predictions.csv`）

## 模型架构

多任务神经网络同时预测：
- 比赛胜负（二分类）
- 得分差（回归）

模型使用PyTorch实现，包括全连接层和Dropout正则化。

## 评估指标

训练后会在 `artifacts/multitask_nn/metrics.json` 中保存评估指标，包括准确率、AUC等。

## 注意事项

- 数据按比赛时间顺序分割，避免未来数据泄露
- 模型支持GPU加速
- 所有预处理器（scaler）都会保存以确保预测时的一致性

    