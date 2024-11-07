import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import torch
import torch.nn as nn

class gMLPLayer(nn.Module):
    def __init__(self, num_patches: int, embedding_dim: int, dropout_rate: float = 0.1):
        super(gMLPLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.channel_projection1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.channel_projection2 = nn.Linear(embedding_dim, embedding_dim)
        self.normalize1 = nn.LayerNorm(embedding_dim)
        self.normalize2 = nn.LayerNorm(embedding_dim)
        self.spatial_projection = nn.Linear(num_patches, num_patches, bias=True)

    def spatial_gating_unit(self, x: torch.Tensor) -> torch.Tensor:
        u, v = torch.chunk(x, chunks=2, dim=-1)
        v = self.normalize2(v)
        v_projected = self.spatial_projection(v.transpose(-1, -2)).transpose(-1, -2)
        return u * v_projected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize1(x)
        x_projected = self.channel_projection1(x)
        x_spatial = self.spatial_gating_unit(x_projected)
        x_projected = self.channel_projection2(x_spatial)
        return x + x_projected

class MLPMixerLayer(nn.Module):
    def __init__(self, num_patches: int, embedding_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super(MLPMixerLayer, self).__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_patches),
            nn.Linear(num_patches, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_patches),
            nn.Dropout(dropout_rate)
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing
        x = x + self.token_mixing(x.transpose(-1, -2)).transpose(-1, -2)
        # Channel mixing
        x = x + self.channel_mixing(x)
        return x

class ResMLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, expansion_factor: int = 4, dropout_rate: float = 0.1):
        super(ResMLPBlock, self).__init__()
        hidden_dim = embedding_dim * expansion_factor
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.channel_mlp1 = nn.Linear(embedding_dim, hidden_dim)
        self.channel_mlp2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection for first MLP block
        residual = x
        x = self.norm1(x)
        x = self.channel_mlp1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.channel_mlp2(x)
        x = self.dropout(x)
        x = x + residual

        # Second residual connection
        x = x + self.dropout(self.norm2(x))
        return x
