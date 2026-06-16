import pandas as pd
import numpy as np
import optuna
import mlflow
import pickle
from sklearn.ensemble import IsolationForest
from utils import get_logger, load_config

logger = get_logger('OptunaOptimization')

def objective(trial, X, contamination):
    """Optuna objective function maximizing the boundary gap."""
    # Define Hyperparameter Search Space
    n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
    max_samples = trial.suggest_float('max_samples', 0.5, 1.0, step=0.1)
    max_features = trial.suggest_float('max_features', 0.5, 1.0, step=0.1)
    
    model = IsolationForest(
        n_estimators = n_estimators,
        max_samples = max_samples,
        max_features = max_features,
        contamination = contamination,
        random_state = 42,
        n_jobs = -1
    )
    model.fit(X)
    
    # Calculate boundary gap
    scores = model.decision_function(X)
    predictions = model.predict(X)
    
    normal_scores = scores[predictions == 1]
    anomaly_scores = scores[predictions == -1]
    
    if len(anomaly_scores) == 0:
        return 0.0
        
    return np.mean(normal_scores) - np.mean(anomaly_scores)

if __name__ == "__main__":
    config = load_config('config/optuna_config.json')
    mlflow.set_experiment(config['experiment_name'])
    
    logger.info("--- Starting Optuna Optimization on Champion Model ---")
    df = pd.read_csv(config['data_paths']['input'])
    
    features = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'drone_id']
    X = df[features]
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X, config['optuna_params']['contamination']),
        n_trials=config['optuna_params']['n_trials']
    )
    
    logger.info(f"Optimization Complete. Best Trial: {study.best_trial.number}")
    logger.info(f"Best Hyperparameters: {study.best_trial.params}")
    
    with mlflow.start_run(run_name = "Optuna_Best_Model"):
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("max_boundary_gap", study.best_value)
        
        # Train Final Optimized Model
        best_model = IsolationForest(
            **study.best_trial.params,
            contamination = config['optuna_params']['contamination'],
            random_state = 42,
            n_jobs = -1
        )
        best_model.fit(X)
        
        # Serialize and Save
        model_path = config['data_paths']['model_path']
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
            
        mlflow.sklearn.log_model(best_model, "champion_isolation_forest")
        logger.info(f"Final champion model saved to {model_path}")