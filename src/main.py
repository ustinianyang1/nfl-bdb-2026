import argparse
import sys
import os
import torch
import numpy as np
import random
import secrets
from datetime import datetime
from collections import defaultdict
from src.config import Config
from src.utils import seed_everything
from src.preprocess import clean_and_process_data
from src.train import main_train
from src.verify import main as verify_main
# 导入新的可视化函数
from src.visualize import animate_full_play
from src.model import TimeSeriesTransformer

def run_verify():
    print(f"[INFO] --- 环境验证 ---")
    return verify_main()

def run_preprocess():
    print(f"[INFO] --- 数据预处理 ---")
    return clean_and_process_data()

def run_train():
    print(f"[INFO] --- 模型训练 ---")
    inputs_path = os.path.join(Config.PROCESSED_DATA_DIR, 'inputs.pt')
    if not os.path.exists(inputs_path):
        print(f"[WARN] 数据未找到，尝试运行预处理...")
        if not run_preprocess():
            return False
    main_train()
    return True

def run_visualize():
    viz_dir = os.path.join(Config.PROJECT_ROOT, 'visualize')
    os.makedirs(viz_dir, exist_ok=True)
    print(f"[INFO] --- 全场可视化 (Saved to {viz_dir}) ---")
    
    # 1. 加载所有数据
    inputs_path = os.path.join(Config.PROCESSED_DATA_DIR, 'inputs.pt')
    targets_path = os.path.join(Config.PROCESSED_DATA_DIR, 'targets.pt')
    meta_path = os.path.join(Config.PROCESSED_DATA_DIR, 'metadata.pt')
    stats_path = os.path.join(Config.PROCESSED_DATA_DIR, 'stats.pt')

    if not os.path.exists(meta_path):
        print("[ERROR] 未找到 metadata.pt。请先运行 'python src/main.py --mode preprocess' 重新生成数据。")
        return False
    
    inputs = torch.load(inputs_path, weights_only=True)
    targets = torch.load(targets_path, weights_only=True)
    metadata = torch.load(meta_path, weights_only=True) # list of (game_id, play_id, nfl_id)
    
    # 2. 对数据进行分组：Key=(game_id, play_id), Value=[indices]
    play_mapping = defaultdict(list)
    for idx, (gid, pid, nid) in enumerate(metadata):
        play_mapping[(gid, pid)].append(idx)
    
    # 3. 筛选出人数足够的回合
    valid_plays = [k for k, v in play_mapping.items() if len(v) >= 10]
    if not valid_plays:
        print("[ERROR] 未找到有效回合")
        return False

    # 4. 随机选择一个回合
    chosen_key = secrets.choice(valid_plays)
    chosen_indices = play_mapping[chosen_key]
    print(f"[INFO] 选中回合: Game={chosen_key[0]}, Play={chosen_key[1]}, 帧数={len(chosen_indices)}")

    # 5. 准备批量数据
    batch_inputs = inputs[chosen_indices].to(Config.DEVICE)   # [N_Players, Past, Feats]
    batch_targets = targets[chosen_indices] # [N_Players, Future, 2]

    # 6. 加载模型推理
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'best.pth')
    if not os.path.exists(ckpt_path):
        print("[WARN] 未找到模型权重，使用随机初始化模型演示")
        
    model = TimeSeriesTransformer().to(Config.DEVICE)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
            batch_preds = model(batch_inputs) # [N_Players, Future, 2]

    # 7. 反归一化并打包数据
    batch_inputs = batch_inputs.cpu().numpy()
    batch_targets = batch_targets.cpu().numpy()
    batch_preds = batch_preds.float().cpu().numpy()
    
    stats = {}
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, weights_only=True)
    
    def denorm(vec, col_name):
        if col_name not in stats: return vec
        return vec * stats[col_name]['std'] + stats[col_name]['mean']

    idx_x = Config.FEATURE_COLS.index('x')
    idx_y = Config.FEATURE_COLS.index('y')

    play_viz_data = []
    for i in range(len(chosen_indices)):
        # Input (Past)
        px = denorm(batch_inputs[i, :, idx_x], 'x')
        py = denorm(batch_inputs[i, :, idx_y], 'y')
        past_xy = np.stack([px, py], axis=1)
        
        # Target (Future)
        tx = denorm(batch_targets[i, :, 0], 'x')
        ty = denorm(batch_targets[i, :, 1], 'y')
        target_xy = np.stack([tx, ty], axis=1)
        
        # Pred (Future)
        prx = denorm(batch_preds[i, :, 0], 'x')
        pry = denorm(batch_preds[i, :, 1], 'y')
        pred_xy = np.stack([prx, pry], axis=1)
        
        play_viz_data.append({
            'past': past_xy,
            'target': target_xy,
            'pred': pred_xy
        })

    # 8. 生成动画
    save_name = f"game_{chosen_key[0]}_play_{chosen_key[1]}.gif"
    save_path = os.path.join(viz_dir, save_name)
    animate_full_play(play_viz_data, save_path)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'verify', 'preprocess', 'train', 'visualize'])
    args = parser.parse_args()
    
    Config.ensure_dirs()
    seed_everything(Config.SEED)

    if args.mode == 'verify':
        sys.exit(run_verify())
    elif args.mode == 'preprocess':
        run_preprocess()
    elif args.mode == 'train':
        run_train()
    elif args.mode == 'visualize':
        run_visualize()
    elif args.mode == 'all':
        if run_verify() == 0 and run_preprocess() and run_train():
            run_visualize()

if __name__ == "__main__":
    main()