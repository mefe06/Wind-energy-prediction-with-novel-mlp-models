import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from base_models import ModelBase, NeuralNetworkModel
from mlp_blocks import gMLPLayer, MLPMixerLayer, ResMLPBlock
import torch
import torch.nn as nn

class LightGBMModel(ModelBase):
    """LightGBM implementation with hyperparameter tuning."""
    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state
        
    def get_param_grid(self, max_depth: int) -> Dict[str, Any]:
        """Generate parameter grid for tuning."""
        return {
            'task': ['train'],
            'boosting_type': ['gbdt'],
            'objective': ['regression_l1'],
            'metric': ['l1'],
            'max_depth': [max_depth],
            'num_leaves': [2**i for i in range(4, max_depth)],
            'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3],
            'min_split_gain': [0, 1, 2, 3, 4, 5],
            'min_data_in_leaf': [50*i for i in range(1, 20)],
            'max_bin': [127, 255, 511, 1023],
            'n_estimators': [(100+400*i) for i in range(1, 7)],
            'verbose': [-1]
        }
    
    def tune_model(self, X: pd.DataFrame, y: pd.Series, cv_folds: List[Tuple]) -> Dict[str, Any]:
        """Perform hyperparameter tuning."""
        best_score = float('-inf')
        best_params = None
        best_depth = None
        
        for depth in range(7, 16):
            param_grid = self.get_param_grid(depth)
            model = lgbm.LGBMRegressor(n_jobs=-1)
            
            search = RandomizedSearchCV(
                model, 
                param_grid, 
                n_iter=15,
                cv=cv_folds, 
                scoring='neg_mean_absolute_error', 
                random_state=self.random_state
            )
            search.fit(X, y)
            
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_params = search.best_params_
                best_depth = depth
                
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_depth': best_depth
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> None:
        """Train the model with given parameters."""
        train_data = lgbm.Dataset(X, y)
        self.model = lgbm.train(params, train_data)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

class gMLPNetwork(nn.Module):
    def __init__(self, input_features_nb, num_patches: int, embedding_dim: int, num_layers: int, dropout_rate: float = 0.1):
        super(gMLPNetwork, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_features_nb, embedding_dim)
        
        # Stack of gMLP layers
        self.layers = nn.ModuleList([
            gMLPLayer(num_patches=num_patches, embedding_dim=embedding_dim, dropout_rate=dropout_rate) 
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.regressor = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x):
        # Project patches into embedding dimension
        x = self.embedding(x)
        
        # Apply stacked gMLP layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling and classification
        x = self.regressor(x)
        return x

class MLPMixerNetwork(nn.Module):
    def __init__(self, input_features_nb, num_patches: int, embedding_dim: int, num_layers: int, hidden_dim: int, dropout_rate: float = 0.1):
        super(MLPMixerNetwork, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_features_nb, embedding_dim)
        
        # Stack of MLP Mixer layers
        self.layers = nn.ModuleList([
            MLPMixerLayer(num_patches=num_patches, embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate) 
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.regressor = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x):
        # Project patches into embedding dimension
        x = self.embedding(x)
        
        # Apply stacked MLP Mixer layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling and classification
        x = self.regressor(x)
        return x

class ResMLPNetwork(nn.Module):
    def __init__(self, input_features_nb, embedding_dim: int, num_layers: int, expansion_factor: int, dropout_rate: float = 0.1):
        super(ResMLPNetwork, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_features_nb, embedding_dim)
        
        # Stack of ResMLP blocks
        self.layers = nn.ModuleList([
            ResMLPBlock(embedding_dim=embedding_dim, expansion_factor=expansion_factor, dropout_rate=dropout_rate) 
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.regressor = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x):
        # Project patches into embedding dimension
        x = self.embedding(x)
        
        # Apply stacked ResMLP blocks
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling and classification
        x = self.regressor(x)
        return x

