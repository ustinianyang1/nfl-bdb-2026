import polars as pl
import numpy as np
import torch
import os
import glob
from tqdm import tqdm
from .config import Config

def clean_and_process_data():
    """读取 CSV，执行标准化，生成滑动窗口序列 Tensor 及 元数据"""
    save_path_inputs = os.path.join(Config.PROCESSED_DATA_DIR, 'inputs.pt')
    save_path_targets = os.path.join(Config.PROCESSED_DATA_DIR, 'targets.pt')
    save_path_meta = os.path.join(Config.PROCESSED_DATA_DIR, 'metadata.pt') # 新增
    save_path_stats = os.path.join(Config.PROCESSED_DATA_DIR, 'stats.pt')

    # 1. 幂等性检查 (加上 metadata 检查)
    if os.path.exists(save_path_inputs) and os.path.exists(save_path_meta):
        print(f"[Info] 跳过预处理 (Data & Metadata found)")
        return True

    # 2. 扫描所有 CSV 文件
    files = glob.glob(Config.RAW_DATA_PATTERN)
    if not files:
        print(f"[Error] 未找到任何数据文件: {Config.RAW_DATA_PATTERN}")
        return False

    print(f"[Info] 找到 {len(files)} 个文件，Polars 加载中...")
    q = pl.scan_csv(files, null_values=['NA', 'nan', ''], ignore_errors=True)
    q = q.filter(pl.col("nfl_id").is_not_null())
    df = q.collect()
    print(f"[Info] 原始数据行数: {df.height}")
    
    # --- 数据清洗逻辑保持不变 ---
    if "player_height" in df.columns:
        if df["player_height"].dtype == pl.String:
            df = df.with_columns(
                pl.when(pl.col("player_height").str.contains("-"))
                .then(
                    pl.col("player_height").str.split("-").list.get(0).cast(pl.Int32, strict=False) * 12 + 
                    pl.col("player_height").str.split("-").list.get(1).cast(pl.Int32, strict=False)
                )
                .otherwise(pl.col("player_height").cast(pl.Float32, strict=False))
                .alias("player_height")
            )

    for col in Config.NUMERIC_COLS:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float32, strict=False))
    
    df = df.drop_nulls(subset=Config.NUMERIC_COLS)
    df = df.sort(['game_id', 'play_id', 'nfl_id', 'frame_id'])

    # --- 标准化 ---
    print("[Info] 标准化...")
    feature_stats = {}
    for col in Config.NUMERIC_COLS:
        mean = df[col].mean()
        std = df[col].std()
        if std is None or std == 0: std = 1.0 
        df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
        feature_stats[col] = {'mean': mean, 'std': std}
    torch.save(feature_stats, save_path_stats)

    # --- 构建序列与元数据 ---
    print("[Info] 生成序列与元数据...")
    min_seq_len = Config.PAST_FRAMES + Config.FUTURE_FRAMES
    
    # GroupBy 聚合 (新增保留 game_id, play_id)
    df_grouped = df.group_by(['game_id', 'play_id', 'nfl_id'], maintain_order=True).agg([
        pl.col(c) for c in Config.FEATURE_COLS
    ])

    all_inputs = []
    all_targets = []
    all_metadata = [] # 存储 (game_id, play_id, nfl_id)

    rows = df_grouped.iter_rows(named=True)
    target_indices = [Config.FEATURE_COLS.index(c) for c in Config.TARGET_COLS]

    for row in tqdm(rows, total=df_grouped.height, desc="Processing"):
        seq_len = len(row[Config.FEATURE_COLS[0]])
        if seq_len < min_seq_len: continue

        raw_seq_list = [row[c] for c in Config.FEATURE_COLS]
        raw_seq = np.array(raw_seq_list, dtype=np.float32).T 
        
        # 获取该轨迹的元信息
        meta_info = (row['game_id'], row['play_id'], row['nfl_id'])
        
        num_windows = seq_len - min_seq_len + 1
        for i in range(num_windows):
            past_end = i + Config.PAST_FRAMES
            future_end = past_end + Config.FUTURE_FRAMES
            
            input_window = raw_seq[i:past_end, :]
            target_window = raw_seq[past_end:future_end, target_indices]
            
            all_inputs.append(input_window)
            all_targets.append(target_window)
            all_metadata.append(meta_info) # 同步保存

    if not all_inputs:
        print("[Error] 未生成序列")
        return False

    print(f"[Info] 保存 Tensor...")
    tensor_inputs = torch.from_numpy(np.stack(all_inputs))
    tensor_targets = torch.from_numpy(np.stack(all_targets))
    
    torch.save(tensor_inputs, save_path_inputs)
    torch.save(tensor_targets, save_path_targets)
    # 保存元数据 (List of tuples)
    torch.save(all_metadata, save_path_meta)
    
    print(f"[Info] 完成. 样本数: {len(all_inputs)}")
    return True