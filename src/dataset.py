import torch
from torch.utils.data import Dataset, DataLoader
import os
from .config import Config

class NFLTrajectoryDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.mode = mode
        
        inputs_path = os.path.join(Config.PROCESSED_DATA_DIR, 'inputs.pt')
        targets_path = os.path.join(Config.PROCESSED_DATA_DIR, 'targets.pt')
        
        self.inputs = torch.load(inputs_path, weights_only=True)
        self.targets = torch.load(targets_path, weights_only=True)
        
        total_len = len(self.inputs)
        train_len = int(0.8 * total_len)
        
        if mode == 'train':
            self.inputs = self.inputs[:train_len]
            self.targets = self.targets[:train_len]
        else:
            self.inputs = self.inputs[train_len:]
            self.targets = self.targets[train_len:]
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_dataloader(mode='train'):
    dataset = NFLTrajectoryDataset(mode=mode)
    shuffle = (mode == 'train')
    
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )