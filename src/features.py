import pandas as pd
import numpy as np
import mlflow
import os
from utils import get_logger, load_config

# Initialize Logger
logger = get_logger('FeatureEngineering')

def engineer_temporal_features(df, window_size, lag_steps):
    """
    Rolling windows and lag variables grouped by drone_id.
    """
    logger.info(f"Engineering features: Window Size = {window_size}H, Lag Steps = {lag_steps}")
    
    # Ensure chronological order
    df = df.sort_values(by=['drone_id', 'timestamp']).reset_index(drop=True)
    
    # Isolate numeric columns for mathematical operations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'drone_id']
    
    engineered_dfs = []
    
    # Group by drone to prevent data leaking between different physical units
    for drone_id, group in df.groupby('drone_id'):
        group = group.copy()
        
        for col in numeric_cols:
            # 1. Rolling Mean (Sustained Context)
            group[f'{col}_rolling_mean_{window_size}H'] = group[col].rolling(window=window_size).mean()
            
            # 2. Rolling Standard Deviation (Volatility / Erratic behavior)
            group[f'{col}_rolling_std_{window_size}H'] = group[col].rolling(window=window_size).std()
            
            # 3. Lag Variable (Delta Shift)
            group[f'{col}_lag_{lag_steps}'] = group[col].shift(lag_steps)
            
        engineered_dfs.append(group)
        
    df_engineered = pd.concat(engineered_dfs).reset_index(drop=True)
    
    # Rolling windows introduce NaNs at the start of every drone's timeline. We must purge them.
    initial_shape = df_engineered.shape
    df_engineered = df_engineered.dropna().reset_index(drop=True)
    logger.info(f"Purged {initial_shape[0] - df_engineered.shape[0]} NaN rows introduced by temporal shifting.")
    
    return df_engineered

if __name__ == "__main__":
    # Load Configuration
    config = load_config('config/exp3_sustained.json')
    
    # Setup MLflow Tracking
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run(run_name="Temporal_Feature_Engineering"):
        logger.info("Starting Temporal Feature Engineering Pipeline...")
        
        # Log parameters to MLflow
        mlflow.log_param("rolling_window_hours", config['temporal_params']['rolling_window_hours'])
        mlflow.log_param("lag_steps", config['temporal_params']['lag_steps'])
        
        # Load Data
        input_path = config['data_paths']['input']
        logger.info(f"Loading matrix from {input_path}")
        df = pd.read_csv(input_path)
        
        # Engineer Features
        df_engineered = engineer_temporal_features(
            df, 
            window_size=config['temporal_params']['rolling_window_hours'], 
            lag_steps=config['temporal_params']['lag_steps']
        )
        
        # Save output
        output_path = config['data_paths']['output']
        df_engineered.to_csv(output_path, index=False)
        logger.info(f"Engineered matrix saved to {output_path} with shape {df_engineered.shape}")
        
        # Log matrix dimensions to MLflow as metrics
        mlflow.log_metric("final_feature_count", df_engineered.shape[1])
        mlflow.log_metric("final_row_count", df_engineered.shape[0])
        
        logger.info("Feature Engineering Pipeline completed successfully.")