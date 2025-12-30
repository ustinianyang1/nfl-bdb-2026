import torch
import os
import glob

class Config:
    # --- 路径配置 ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_PATTERN = os.path.join(DATA_DIR, 'raw', 'input_*.csv') 
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

    # --- 特征配置 ---
    # 必须标准化的数值列
    NUMERIC_COLS = ['x', 'y', 's', 'a', 'dir', 'o', 'player_height', 'player_weight']
    FEATURE_COLS = NUMERIC_COLS # 输入特征
    TARGET_COLS = ['x', 'y']    # 输出特征
    
    # --- 序列参数 ---
    PAST_FRAMES = 10     
    FUTURE_FRAMES = 10   
    
    # --- 模型参数 ---
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1

    # --- 训练参数 ---
    BATCH_SIZE = 512     
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    PATIENCE = 10        # 早停耐心值
    NUM_WORKERS = 4      
    SEED = 42
    
    # --- 设备 ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True       # 混合精度

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)