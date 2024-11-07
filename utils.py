import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from models import (gMLPNetwork, MLPMixerNetwork, ResMLPNetwork, LightGBMModel)  # From our model implementations
from base_models import NeuralNetworkModel  # From our base model implementations
from preprocessing import preprocess_wind_data
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor
def load_data(base_path: str, zone_range: range = range(1, 11)) -> Dict[str, Any]:
    base_path = Path(base_path)
    
    # Initialize data structures
    train_data = {}
    test_data = {}
    
    # Load test targets
    test_y = pd.read_csv(base_path / "TestTar_W.csv")
    test_y.TIMESTAMP = pd.to_datetime(test_y.TIMESTAMP)
    
    # Load zone data
    for zone in zone_range:
        # Load training data
        train_data[zone] = pd.read_csv(base_path / f"Train_W_Zone{zone}.csv")
        train_data[zone].TIMESTAMP = pd.to_datetime(train_data[zone].TIMESTAMP)
        
        # Load test data
        test_data[zone] = pd.read_csv(base_path / f"TestPred_W_Zone{zone}.csv")
        test_data[zone].TIMESTAMP = pd.to_datetime(test_data[zone].TIMESTAMP)
        
        # Merge test data with targets
        test_data[zone] = pd.merge(
            test_data[zone], 
            test_y, 
            on=['ZONEID', 'TIMESTAMP'], 
            how='left'
        )
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'test_y': test_y
    }

def get_model_params(model_type: str, args: argparse.Namespace) -> Dict[str, Any]:
    
    model_specific_params = {
        'lightgbm': {
            'nb_estimators': args.n_estimators,
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'random_state': args.random_state
        },
        'nn': {
            'hidden_layers': [int(x) for x in args.hidden_layers.split(',')],
            'dropout_rate': args.dropout_rate,
            'input_shape': args.input_features_nb
        },
        'gmlp': {
            'input_features_nb': args.input_features_nb, 
            'num_patches': args.num_patches,
            'embedding_dim': args.embedding_dim,
            'num_layers': args.num_blocks,  # Updated to match PyTorch gMLPNetwork
            'dropout_rate': args.dropout_rate,
        },
        'mlp_mixer': {
            'input_features_nb': args.input_features_nb, 
            'num_patches': args.num_patches,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_units,  # Updated to match PyTorch MLPMixerNetwork
            'num_layers': args.num_blocks,  # Updated to match PyTorch MLPMixerNetwork
            'dropout_rate': args.dropout_rate,
        },
        'resmlp': {
            'input_features_nb': args.input_features_nb, 
            'embedding_dim': args.embedding_dim,  # Added embedding_dim for ResMLPNetwork
            'expansion_factor': args.hidden_units // args.embedding_dim,  # Derived from hidden_units and embedding_dim
            'num_layers': args.num_blocks,  # Updated to match PyTorch ResMLPNetwork
            'dropout_rate': args.dropout_rate,
        }
    }
    
    return model_specific_params[model_type]


def get_model(model_type: str) -> Any:
    """Get model instance based on model type."""
    models = {
        'lightgbm': LGBMRegressor,#LightGBMModel,
        'nn': NeuralNetworkModel,
        'gmlp': gMLPNetwork,
        'mlp_mixer': MLPMixerNetwork,
        'resmlp': ResMLPNetwork
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                       f"Available types: {list(models.keys())}")
    return models[model_type]

def train_model(model, X_train, y_train, batch_size, epochs, learning_rate=0.001):
    """Train a PyTorch model with the given data and parameters."""
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_idx =0
        for batch_X, batch_y in dataloader:
            #print(f"Processing batch {batch_idx+1}/{len(dataloader)} with shape {batch_X.shape}")
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_idx+=1
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

