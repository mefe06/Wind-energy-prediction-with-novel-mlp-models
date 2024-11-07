import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

def preprocess_wind_data(
    train_files: Dict[int, str],
    test_files: Dict[int, str],
    test_target_file: str,
    n_zones: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    def _calculate_wind_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic wind features from raw measurements."""
        # Wind speed calculations
        df['ws_10'] = np.sqrt(df['U10']**2 + df['V10']**2)
        df['ws_100'] = np.sqrt(df['U100']**2 + df['V100']**2)
        
        # Wind energy calculations
        df['we_10'] = df['ws_10']**3
        df['we_100'] = df['ws_100']**3
        
        # Wind direction calculations
        df['wd_10'] = (180/np.pi) * np.arctan2(df['V10'], df['U10'])
        df['wd_100'] = (180/np.pi) * np.arctan2(df['V100'], df['U100'])
        
        # Additional features
        df['wss_10'] = df['U10']**2 + df['V10']**2
        df['wss_100'] = df['ws_100']**2
        
        # Ratio features
        df['ws_ratio'] = df['ws_100'] / df['ws_10']
        df['we_ratio'] = df['we_100'] / df['we_10']
        df['wd_ratio'] = df['wd_100'] / df['wd_10']
        
        return df
    
    def _add_directional_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add wind direction quadrant and bearing features."""
        # Wind direction quadrants for 100m
        conditions_100 = [
            (df['wd_100'] > 0) & (df['wd_100'] <= 90),
            (df['wd_100'] > 90),
            (df['wd_100'] >= -90) & (df['wd_100'] < 0),
            (df['wd_100'] < -90)
        ]
        choices = [1, 2, 4, 3]
        
        df['wdq_100'] = np.select(conditions_100, choices)
        df['wdq_100_cos'] = np.cos(2*np.pi*df['wdq_100']/4)
        df['wdq_100_sin'] = np.sin(2*np.pi*df['wdq_100']/4)
        
        # Bearing calculations for 100m
        shifted_angles_100 = np.select([df['wd_100'] >= 0, df['wd_100'] <= 0],
                                     [df['wd_100'], 360 + df['wd_100']])
        df['wdb_100'] = shifted_angles_100
        df['wdb_100_cos'] = np.cos(2*np.pi*df['wdb_100']/360)
        df['wdb_100_sin'] = np.sin(2*np.pi*df['wdb_100']/360)
        
        # Repeat for 10m
        conditions_10 = [
            (df['wd_10'] > 0) & (df['wd_10'] <= 90),
            (df['wd_10'] > 90),
            (df['wd_10'] >= -90) & (df['wd_10'] < 0),
            (df['wd_10'] < -90)
        ]
        
        df['wdq_10'] = np.select(conditions_10, choices)
        df['wdq_10_cos'] = np.cos(2*np.pi*df['wdq_10']/4)
        df['wdq_10_sin'] = np.sin(2*np.pi*df['wdq_10']/4)
        
        shifted_angles_10 = np.select([df['wd_10'] >= 0, df['wd_10'] <= 0],
                                    [df['wd_10'], 360 + df['wd_10']])
        df['wdb_10'] = shifted_angles_10
        df['wdb_10_cos'] = np.cos(2*np.pi*df['wdb_10']/360)
        df['wdb_10_sin'] = np.sin(2*np.pi*df['wdb_10']/360)
        
        return df
    
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features and rolling statistics."""
        # Time components
        df['hour'] = df['TIMESTAMP'].dt.hour
        df['day'] = df['TIMESTAMP'].dt.day
        df['month'] = df['TIMESTAMP'].dt.month
        
        # Cyclical encoding
        df['cos_hour'] = np.cos(2*np.pi*df['hour']/24)
        df['sin_hour'] = np.sin(2*np.pi*df['hour']/24)
        df['cos_day'] = np.cos(2*np.pi*df['day']/7)
        df['sin_day'] = np.sin(2*np.pi*df['day']/7)
        df['cos_month'] = np.cos(2*np.pi*df['month']/12)
        df['sin_month'] = np.sin(2*np.pi*df['month']/12)
        
        # Drop original time columns
        df = df.drop(['hour', 'day', 'month'], axis=1)
        
        return df
    
    def _add_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add rolling statistics for various features."""
        for window in windows:
            # Target variable rolling stats
            df[f'roll_{window}_we(t-1)_mean'] = df['we(t-1)'].rolling(window).mean()
            df[f'roll_{window}_we(t-1)_std'] = df['we(t-1)'].rolling(window).std()
            
            # Wind speed rolling stats
            for height in ['10', '100']:
                base_cols = [f'ws_{height}', f'wd_{height}', f'we_{height}']
                for col in base_cols:
                    df[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window).std()
        
        return df
    
    def _add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Add lagged features for target variable and other measurements."""
        # Target variable lags
        for lag in lags:
            df[f'we(t-{lag})'] = df['TARGETVAR'].shift(lag)
            df[f'ew(t-{lag})'] = df['TARGETVAR'].shift(1) - df['TARGETVAR'].shift(lag + 1)
        
        # Wind measurements lags
        for height in ['10', '100']:
            base_cols = [f'ws_{height}', f'wd_{height}', f'we_{height}']
            for col in base_cols:
                for lag in range(1, 4):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    df[f'{col}_f_{lag}'] = df[col].shift(-lag)
        
        return df
    
    # Load and process data
    train_data = {}
    test_data = {}
    
    # Load test targets
    test_y = test_target_file#pd.read_csv(test_target_file)
    test_y['TIMESTAMP'] = pd.to_datetime(test_y['TIMESTAMP'])
    
    # Process each zone
    for zone in range(1, n_zones + 1):
        # Load data
        train_data[zone] = train_files[zone] #pd.read_csv(train_files[zone])
        test_data[zone] = test_files[zone] #pd.read_csv(test_files[zone])
        
        # Convert timestamps
        train_data[zone]['TIMESTAMP'] = pd.to_datetime(train_data[zone]['TIMESTAMP'])
        test_data[zone]['TIMESTAMP'] = pd.to_datetime(test_data[zone]['TIMESTAMP'])
        
        # Merge test data with targets
        test_data[zone] = pd.merge(test_data[zone], test_y, on=['ZONEID', 'TIMESTAMP', 'TARGETVAR'], how='left')
        
        # Apply feature engineering
        for dataset in [train_data[zone], test_data[zone]]:
            dataset = _calculate_wind_features(dataset)
            dataset = _add_directional_features(dataset)
            dataset = _add_temporal_features(dataset)
            dataset = _add_lag_features(dataset, lags=[1, 2, 3, 7, 14, 21])
            dataset = _add_rolling_features(dataset, windows=[4, 6, 12, 24])
    
    # Combine all zones
    train_combined = pd.concat([train_data[zone] for zone in range(1, n_zones + 1)], ignore_index=True)
    test_combined = pd.concat([test_data[zone] for zone in range(1, n_zones + 1)], ignore_index=True)
    
    # One-hot encode zone IDs
    train_combined = pd.get_dummies(train_combined, columns=['ZONEID'], drop_first=True)
    test_combined = pd.get_dummies(test_combined, columns=['ZONEID'], drop_first=True)
    
    # Handle missing values
    train_combined = train_combined.dropna().reset_index(drop=True)
    test_combined = test_combined.dropna().reset_index(drop=True)
    
    return train_combined, test_combined

def create_cv_folds(data: pd.DataFrame, timestamp_col: str, n_folds: int = 5) -> List[Tuple]:
    """Create time-based cross-validation folds."""
    def split_period_to_n(start_date, end_date, num_folds):
        total_duration = end_date - start_date
        fold_duration = total_duration/num_folds
        return [(start_date + (i * fold_duration), 
                start_date + ((i + 1) * fold_duration)) 
                for i in range(num_folds)]
    
    start = data[timestamp_col].min()
    end = data[timestamp_col].max()
    fold_dates = split_period_to_n(start, end, n_folds)
    
    indices = []
    for start_date, end_date in fold_dates:
        idx = data.index[
            (data[timestamp_col] >= start_date) & 
            (data[timestamp_col] <= end_date)
        ].tolist()
        indices.append(idx)
    
    folds = []
    for i in range(n_folds):
        test = np.array(indices[i])
        train_indices = indices[:i] + indices[i+1:]
        train = np.array([idx for sublist in train_indices for idx in sublist])
        np.random.shuffle(train)
        np.random.shuffle(test)
        folds.append((train, test))
    
    return folds
