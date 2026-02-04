# NFL Big Data Bowl 2026: 轨迹预测系统

本项目面向 NFL Big Data Bowl 2026 预测赛道，目标是利用 Next Gen Stats (NGS) 追踪数据预测球员的运动轨迹。系统基于时空 Transformer 架构，支持从数据加载、特征工程、模型训练到推理与可视化的完整流程。

## 项目结构

nfl-bdb-2026/
├── README.md               # 项目文档
├── QUICKSTART.md           # 快速开始指南
├── requirements.txt        # 依赖库
├── checkpoints/            # 模型保存路径
│   ├── best.pth            # 验证集损失最低的模型
│   └── last.pth            # 训练结束时的模型 (断点续训未实现)
├── data/                   # 数据目录
│   ├── processed/          # 预处理后的张量数据
│   │   ├── inputs.pt       # 模型输入特征
│   │   ├── metadata.pt     # 比赛元数据
│   │   ├── stats.pt        # 统计特征
│   │   └── targets.pt      # 预测目标
│   └── raw/                # 原始 CSV 数据
│       ├── input_*.csv     # 训练输入
│       ├── output_*.csv    # 训练目标
│       ├── test_input.csv  # 测试输入
│       └── test_output.csv # 测试目标
├── logs/                   # 日志目录
│   └── train.log           # 训练过程日志
├── src/                    # 源代码
│   ├── __init__.py
│   ├── config.py           # 全局配置 (路径、超参)
│   ├── dataset.py          # PyTorch Dataset 定义
│   ├── main.py             # 统一入口
│   ├── model.py            # Spatiotemporal Transformer 模型
│   ├── preprocess.py       # 高性能数据清洗与序列化 (Polars)
│   ├── train.py            # 训练核心逻辑 (带 AMP)
│   ├── utils.py            # 工具函数 (Seed, Logger)
│   ├── verify.py           # 环境验证脚本
│   └── visualize.py        # 可视化工具
└── visualize/              # 可视化结果
    └── *.gif               # 动画

## 数据集表头

game_id,
play_id,
player_to_predict,
nfl_id,
frame_id,
play_direction,
absolute_yardline_number,
player_name,
player_height,
player_weight,
player_birth_date,
player_position,
player_side,
player_role,
x,
y,
s,
a,
dir,
o,
num_frames_output,
ball_land_x,
ball_land_y
