import pandas as pd
import numpy as np
import pickle
import mlflow
import sys
from sklearn.ensemble import IsolationForest
from utils import get_logger, load_config

# Initialize Logger
logger = get_logger('ModelTraining')

def train_isolation_forest(df, params):
    """
    Trains an Isolation Forest model to isolate the top N% anomalous events.
    """
    logger.info(f"Initializing Isolation Forest with contamination={params['contamination']}")
    
    # Isolate only numeric features for the algorithm (drop timestamps and IDs)
    features = df.select_dtypes(include=[np.number]).columns
    features = [col for col in features if col not in ['drone_id']]
    X = df[features]
    
    # Initialize and fit the model
    model = IsolationForest(
        contamination=params['contamination'],
        random_state=params['random_state'],
        n_estimators=params['n_estimators'],
        n_jobs=-1 # Utilize all CPU cores
    )
    
    logger.info("Fitting model on feature space...")
    model.fit(X)
    
    # Predict anomalies (-1 = Anomaly, 1 = Normal)
    predictions = model.predict(X)
    anomaly_scores = model.decision_function(X)
    
    # Append results back to the original dataframe for forensic review
    df_results = df.copy()
    df_results['anomaly_label'] = predictions
    df_results['anomaly_score'] = anomaly_scores
    
    anomaly_count = len(df_results[df_results['anomaly_label'] == -1])
    logger.info(f"Training complete. Isolated {anomaly_count} anomalous events.")
    
    return model, df_results, features

if __name__ == "__main__":
    # Allow passing different config files via command line for later experiments
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config/exp1_baseline.json'
    config = load_config(config_file)
    
    # Setup MLflow Tracking
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run(run_name=f"Train_{config['model_params']['algorithm']}"):
        logger.info(f"--- Starting Experiment: {config['experiment_name']} ---")
        
        # Load Data
        input_path = config['data_paths']['input']
        logger.info(f"Loading training matrix from {input_path}")
        df = pd.read_csv(input_path)
        
        # Log Hyperparameters
        for key, value in config['model_params'].items():
            mlflow.log_param(key, value)
            
        # Train Model
        if config['model_params']['algorithm'] == "IsolationForest":
            model, df_predictions, used_features = train_isolation_forest(df, config['model_params'])
        else:
            logger.error("Algorithm not yet implemented in train.py")
            sys.exit(1)
            
        # Save Predictions for Forensic Evaluation
        output_path = config['data_paths']['output_predictions']
        df_predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Save Model via Pickle
        model_path = config['data_paths']['model_path']
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model successfully serialized and saved to {model_path}")
        
        # Log Model and Metrics to MLflow
        mlflow.sklearn.log_model(model, "isolation_forest_model")
        anomaly_ratio = len(df_predictions[df_predictions['anomaly_label'] == -1]) / len(df_predictions)
        mlflow.log_metric("actual_anomaly_ratio", anomaly_ratio)
        mlflow.log_metric("feature_count", len(used_features))
        
        logger.info("Experiment run finalized and tracked in MLflow.")