# Re-attempting with minimal implementation of ModelBase and NeuralNetworkModel to avoid memory issues

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
import torch.nn as nn

# Define a base model class in PyTorch, mimicking the ModelBase
class ModelBase:
    """Base class for all models."""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def preprocess(self, X):
        """Preprocess data before model training/prediction."""
        return self.scaler.fit_transform(X)

    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'mae': torch.mean(torch.abs(y_pred - y_true)).item(),
            'mse': torch.mean((y_pred - y_true) ** 2).item()
        }
        return metrics


# Define a Neural Network Model, similar to NeuralNetworkModel
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_shape: int, hidden_layers: List[int], dropout_rate: float = 0.1):
        super().__init__()
        self.model = self.create_mlp(input_shape, hidden_layers, dropout_rate)

    def create_mlp(self, input_shape: int, hidden_layers: List[int], dropout_rate: float = 0.1) -> nn.Sequential:
        """Create a basic MLP model."""
        layers = [nn.Linear(input_shape, hidden_layers[0]), nn.ReLU(), nn.Dropout(dropout_rate)]
        
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i-1], hidden_layers[i]), nn.ReLU(), nn.Dropout(dropout_rate)]
        
        layers.append(nn.Linear(hidden_layers[-1], 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)