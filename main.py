import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any
import logging
from utils import load_data, get_model, get_model_params, train_model  # From utility functions
from preprocessing import preprocess_wind_data  # From preprocessing implementation
from models import (LightGBMModel, gMLPNetwork, MLPMixerNetwork, ResMLPNetwork)  # PyTorch models
from base_models import NeuralNetworkModel  # PyTorch base model

def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def add_model_specific_args(parser: argparse.ArgumentParser):
    """Add model-specific arguments to parser."""
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for neural models')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for neural models')
    
    # LightGBM specific
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of boosting iterations for LightGBM')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for LightGBM')
    parser.add_argument('--max_depth', type=int, default=7, help='Maximum tree depth for LightGBM')
    
    # Neural Network specific
    parser.add_argument('--hidden_layers', type=str, default='64,32', help='Comma-separated list of hidden layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for neural models')
    
    # gMLP, MLP-Mixer, and ResMLP specific
    parser.add_argument('--num_patches', type=int, default=None, help='Number of patches for gMLP/MLP-Mixer')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension for gMLP/MLP-Mixer')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of blocks for gMLP/MLP-Mixer/ResMLP')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units for MLP-Mixer')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Wind Energy Prediction')
    parser.add_argument('--data_path', type=str, default="/Users/mefe/Documents/eee492_data/mlp-models-wind-prediction/Data",
                       help='Path to data directory')
    parser.add_argument('--model_type', type=str, default="mlp_mixer",
                       choices=['lightgbm', 'nn', 'gmlp', 'mlp_mixer', 'resmlp'],
                       help='Type of model to train')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to save outputs')
    
    # Add model specific arguments
    add_model_specific_args(parser)
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting wind energy prediction with {args.model_type} model")
    
    try:
        # Load data
        logger.info("Loading data...")
        data = load_data(args.data_path)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        train_processed, test_processed = preprocess_wind_data(
            data['train_data'],
            data['test_data'],
            data['test_y']
        )
        
        # Prepare data for model
        # Prepare data for model, ensuring only numerical columns are used
        X_train = train_processed.drop(['TIMESTAMP', 'TARGETVAR'], axis=1)
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)  # Convert to numeric, replace NaNs
        X_train = torch.tensor(X_train.values, dtype=torch.float32)

        y_train = train_processed['TARGETVAR'].astype(np.float32).values  # Convert target to float32
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        X_test = test_processed.drop(['TIMESTAMP', 'TARGETVAR'], axis=1)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

        y_test = test_processed['TARGETVAR'].astype(np.float32).values
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Get model-specific parameters
        # Set num_patches if not provided
        if args.num_patches is None:
            args.num_patches = args.batch_size
            args.input_features_nb = X_train.shape[1]
        model_params = get_model_params(args.model_type, args)
        # Initialize model
        logger.info(f"Initializing {args.model_type} model...")
        model_type = get_model(args.model_type)
        model = model_type(**model_params)
        # Train model
        logger.info("Training model...")
        if args.model_type != 'lightgbm':
            # Set up training parameters for neural models
            train_model(model, X_train, y_train, args.batch_size, args.epochs)
        else:
            model.fit(X_train, y_train)
        
        # Make predictions
        logger.info("Making predictions...")
        if args.model_type != 'lightgbm':
            # Handle neural network predictions in batches
            test_dataset = torch.utils.data.TensorDataset(X_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)
            
            predictions = []
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for batch_X in test_loader:
                    batch_pred = model(batch_X[0]).view(-1).numpy()
                    predictions.extend(batch_pred)
            predictions = np.array(predictions)
        else:
            # LightGBM can handle the full dataset at once
            predictions = model.predict(X_test).reshape(-1, 1)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - y_test.numpy()))
        mse = np.mean((predictions - y_test.numpy())**2)
        
        # Log results
        logger.info(f"Results:")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {np.sqrt(mse):.4f}")
        
        # # Save predictions if output path provided
        # if args.output_path:
        #     output_path = Path(args.output_path)
        #     output_path.mkdir(parents=True, exist_ok=True)
            
        #     pd.DataFrame({
        #         'actual': y_test.numpy(),
        #         'predicted': predictions
        #     }).to_csv(output_path / 'predictions.csv', index=False)
            
        #     logger.info(f"Saved predictions to {output_path}")
            
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
