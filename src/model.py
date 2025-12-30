import torch
import torch.nn as nn
from .config import Config

class TimeSeriesTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.d_model = Config.D_MODEL
        
        # 1. Input Projection
        self.embedding = nn.Sequential(
            nn.Linear(len(Config.FEATURE_COLS), self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )
        
        # 2. Positional Encoding (Learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1, Config.PAST_FRAMES, self.d_model) * 0.02)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=Config.NHEAD, 
            dim_feedforward=self.d_model * 4,
            dropout=Config.DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)
        
        # 4. Prediction Head
        # Flatten [Batch, Past, D] -> [Batch, Past*D]
        self.flatten_dim = Config.PAST_FRAMES * self.d_model
        
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, Config.FUTURE_FRAMES * len(Config.TARGET_COLS))
        )
        
        self._init_weights()

    def _init_weights(self):
        # 简单的权重初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [Batch, Past, Features]
        
        # Embed
        x = self.embedding(x) + self.pos_encoder
        
        # Transform
        x = self.transformer(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Predict
        out = self.head(x)
        
        # Reshape: [Batch, Future, 2]
        return out.view(-1, Config.FUTURE_FRAMES, len(Config.TARGET_COLS))