import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os

def create_football_field(figsize=(12, 6.33)):
    """创建 NFL 球场底图"""
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    # 绘制码线
    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white', alpha=0.5)
    
    # 码数文字
    for x in range(20, 110, 10):
        numb = x if x <= 50 else 120 - x
        plt.text(x, 5, str(numb - 10), horizontalalignment='center', fontsize=15, color='white')
        plt.text(x - 0.95, 53.3 - 5, str(numb - 10), horizontalalignment='center', fontsize=15, color='white', rotation=180)
    
    # 端区
    ax.add_patch(patches.Rectangle((0, 0), 10, 53.3, color='blue', alpha=0.2))
    ax.add_patch(patches.Rectangle((110, 0), 10, 53.3, color='blue', alpha=0.2))
    
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.axis('off')
    return fig, ax

def animate_full_play(play_data, save_path=None):
    """
    可视化全场22人
    
    play_data: list of dict, 每个 dict 包含一名球员的数据:
      {
        'past': (Past, 2),
        'target': (Future, 2),
        'pred': (Future, 2)
      }
    """
    num_players = len(play_data)
    print(f"[INFO] 正在渲染...")
    
    fig, ax = create_football_field()
    ax.set_title(f"Full Play Prediction", color='black')

    # 为每名球员创建绘图对象
    # dots: 当前位置点
    # lines_pred: 预测轨迹线
    player_dots = []
    player_lines_pred = []
    player_lines_gt = []
    
    # 使用不同颜色区分球员，或者统一样式
    # 这里简单处理：蓝色点，红色虚线预测，白色实线真值
    for _ in range(num_players):
        dot, = ax.plot([], [], marker='o', markersize=6, markeredgecolor='black', markerfacecolor='blue')
        line_p, = ax.plot([], [], color='red', linewidth=1.5, linestyle='--', alpha=0.8)
        line_g, = ax.plot([], [], color='white', linewidth=1, linestyle='-', alpha=0.4)
        
        player_dots.append(dot)
        player_lines_pred.append(line_p)
        player_lines_gt.append(line_g)

    # 获取时间步长度
    sample = play_data[0]
    len_past = len(sample['past'])
    len_future = len(sample['pred'])
    total_frames = len_past + len_future

    def init():
        for i in range(num_players):
            player_dots[i].set_data([], [])
            player_lines_pred[i].set_data([], [])
            player_lines_gt[i].set_data([], [])
        return player_dots + player_lines_pred + player_lines_gt

    def update(frame):
        for i in range(num_players):
            p_data = play_data[i]
            past = p_data['past']   # [10, 2]
            target = p_data['target'] # [10, 2]
            pred = p_data['pred']     # [10, 2]

            # 1. 历史阶段
            if frame < len_past:
                # 只显示当前点
                cur_x = past[frame, 0]
                cur_y = past[frame, 1]
                player_dots[i].set_data([cur_x], [cur_y])
                
                # 隐藏预测线
                player_lines_pred[i].set_data([], [])
                player_lines_gt[i].set_data([], [])

            # 2. 预测阶段
            else:
                pred_idx = frame - len_past
                if pred_idx < len_future:
                    cur_x = pred[pred_idx, 0]
                    cur_y = pred[pred_idx, 1]
                    player_dots[i].set_data([cur_x], [cur_y])
                    
                    # 显示预测轨迹
                    player_lines_pred[i].set_data(pred[:pred_idx+1, 0], pred[:pred_idx+1, 1])
                    # 显示真实轨迹
                    player_lines_gt[i].set_data(target[:pred_idx+1, 0], target[:pred_idx+1, 1])

        return player_dots + player_lines_pred + player_lines_gt

    ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=120)

    if save_path:
        try:
            ani.save(save_path, writer=PillowWriter(fps=8))
            print(f"[INFO] GIF 保存成功: {save_path}")
        except Exception as e:
            print(f"[ERROR] GIF 保存失败: {e}")
    else:
        plt.show()
    plt.close(fig)