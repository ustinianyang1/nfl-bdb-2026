快速开始
1. 环境安装
确保已安装 CUDA 12.x 驱动。

# 安装依赖 (建议使用虚拟环境)
pip install -r requirements.txt
注意: torch 版本需根据你的 CUDA 版本选择，建议访问 pytorch.org 获取安装命令。

2. 数据准备
请确保数据位于 `data/raw/` 目录下，文件命名格式为 `input_*.csv` (例如 `input_2023_w01.csv`)。系统会自动扫描并合并这些文件。

3. 运行项目
脚本会自动检查数据是否已预处理，若无则自动执行清洗逻辑。

# 执行完整流程
python -m src.main --mode all

# 仅执行环境验证
python -m src.main --mode verify

# 仅执行预处理
python -m src.main --mode preprocess

# 仅执行训练
python -m src.main --mode train

# 仅执行可视化
python -m src.main --mode visualize
