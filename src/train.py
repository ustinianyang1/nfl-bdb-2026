import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import warnings
import numpy as np
from .config import Config
from .model import TimeSeriesTransformer
from .dataset import get_dataloader
from .utils import get_logger

# 屏蔽 Flash Attention 警告 (因为 Windows 下通常不可用)
warnings.filterwarnings("ignore", message=".*1Torch was not compiled with flash attention.*")

class RMSELoss(nn.Module):
    """RMSE Loss: sqrt(MSE)"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, targets):
        mse = self.mse(outputs, targets)
        return torch.sqrt(mse)

class EarlyStopping:
    """早停机制与最佳模型自动保存"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', logger=None):
        """
        Args:
            patience (int): 上次验证集 loss 改善后等待的 epoch 数
            verbose (bool): 是否打印日志
            delta (float): 视为改善的最小变化量
            path (str): 保存最佳模型的文件路径
            logger: 日志记录器实例
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.logger = logger

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'[INFO] 早停计数器: {self.counter} / {self.patience}')
                
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''当验证集 loss 减少时保存模型'''
        if self.verbose and self.logger:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True) 
        
        with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        total_loss += loss.item()
        
    return total_loss / len(loader)

def main_train():
    logger = get_logger('train', Config.LOG_DIR)
    logger.info(f"Device: {Config.DEVICE}")
    
    train_loader = get_dataloader('train')
    val_loader = get_dataloader('val')
    logger.info(f"Batches - Train: {len(train_loader)}, Val: {len(val_loader)}")
    
    model = TimeSeriesTransformer().to(Config.DEVICE)

    criterion = RMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)
    
    # --- 初始化早停对象 ---
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best.pth')
    last_model_path = os.path.join(Config.CHECKPOINT_DIR, 'last.pth')
    
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, 
        verbose=True, 
        path=best_model_path,
        logger=logger
    )
    
    for epoch in range(Config.EPOCHS):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, Config.DEVICE)
        v_loss = validate(model, val_loader, criterion, Config.DEVICE)
        
        # 手动获取 LR 用于打印
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_loss)
        
        logger.info(f"Epoch {epoch+1:02d}/{Config.EPOCHS} | LR: {current_lr:.2e} | Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        
        torch.save(model.state_dict(), last_model_path)

        early_stopping(v_loss, model)
        
        if early_stopping.early_stop:
            logger.info(f"early stopping triggered")
            break

    logger.info(f"Training Complete. Best Validation Loss: {early_stopping.val_loss_min:.4f}")

if __name__ == "__main__":
    main_train()