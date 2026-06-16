import pandas as pd
import numpy as np
import pickle
import mlflow
import sys
from sklearn.ensemble import IsolationForest
from utils import get_logger, load_config

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

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

def train_one_class_svm(df, params):
    """
    Trains a One-Class SVM on temporally engineered features.
    Standardization is mandatory for SVMs to prevent scale dominance.
    """
    logger.info(f"Initializing One-Class SVM with nu={params['nu']}")
    
    # Isolate numeric features
    features = df.select_dtypes(include=[np.number]).columns
    features = [col for col in features if col not in ['drone_id']]
    X = df[features]
    
    # SVMs use distance calculations and are highly sensitive to feature scales.
    # We MUST scale the data (mean=0, variance=1)
    logger.info("Standardizing temporal feature space for SVM compatibility...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize model
    model = OneClassSVM(
        nu=params['nu'],
        kernel=params['kernel'],
        gamma=params['gamma']
    )
    
    logger.info("Fitting OCSVM model on scaled feature space...")
    model.fit(X_scaled)
    
    # Predict (-1 = Anomaly, 1 = Normal)
    predictions = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)
    
    # Append results
    df_results = df.copy()
    df_results['anomaly_label'] = predictions
    df_results['anomaly_score'] = anomaly_scores
    
    anomaly_count = len(df_results[df_results['anomaly_label'] == -1])
    logger.info(f"OCSVM Training complete. Isolated {anomaly_count} anomalous temporal events.")
    
    return model, df_results, features

def train_lof_pca_pipeline(df, params):
    """
    Trains a Local Outlier Factor (LOF) model on a PCA-compressed manifold.
    This resolves the Curse of Dimensionality for distance-based algorithms.
    """
    logger.info(f"Initializing PCA ({params['pca_components']} components) + LOF Pipeline")
    
    # Isolate numeric features
    features = df.select_dtypes(include=[np.number]).columns
    features = [col for col in features if col not in ['drone_id']]
    X = df[features]
    
    logger.info("Standardizing temporal feature space for PCA projection...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize PCA
    pca = PCA(n_components=params['pca_components'], random_state=42)
    
    # Initialize LOF (Note: LOF does not have a standard 'predict' method for new data
    # when novelty=False, but fit_predict works for the training set).
    lof = LocalOutlierFactor(
        n_neighbors=params['n_neighbors'],
        contamination=params['contamination'],
        novelty=False,
        n_jobs=-1
    )
    
    logger.info("Projecting Manifold via PCA...")
    X_pca = pca.fit_transform(X_scaled)
    
    logger.info("Computing Local Density Distances via LOF...")
    predictions = lof.fit_predict(X_pca)
    
    # LOF's negative outlier factor is akin to a decision score (lower is more anomalous)
    anomaly_scores = lof.negative_outlier_factor_
    
    # We must package the Scaler and PCA into a single object for serialization
    # We don't include LOF in the pipeline here because LOF does not support the standard
    # transform/predict API well when saving it for future raw data prediction. 
    # For anomaly detection serialization, saving the fitted PCA space is critical.
    preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=params['pca_components'], random_state=42))
    ])
    preprocessing_pipeline.fit(X)
    
    # Append results
    df_results = df.copy()
    df_results['anomaly_label'] = predictions
    df_results['anomaly_score'] = anomaly_scores
    
    anomaly_count = len(df_results[df_results['anomaly_label'] == -1])
    logger.info(f"LOF Training complete. Isolated {anomaly_count} anomalous dense events.")
    
    # We return the preprocessing pipeline as the 'model' to save, 
    # as LOF novelty=False objects cannot be strictly used for future prediction.
    return preprocessing_pipeline, df_results, features

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
        elif config['model_params']['algorithm'] == "OneClassSVM":
            model, df_predictions, used_features = train_one_class_svm(df, config['model_params'])
        elif config['model_params']['algorithm'] == "LOF_PCA_Pipeline":
            model, df_predictions, used_features = train_lof_pca_pipeline(df, config['model_params'])
        else:
            logger.error(f"Algorithm {config['model_params']['algorithm']} not yet implemented.")
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