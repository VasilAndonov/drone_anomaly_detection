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

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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

def execute_strict_ensemble(params):
    """
    Executes a strict intersection (Consensus Tribunal) across multiple model predictions.
    Utilizes relational inner-joins to align matrices of different temporal lengths.
    """
    logger.info("Initializing Strict Ensemble Tribunal (Intersection Strategy)")
    
    # Load the prediction matrices
    logger.info("Loading prediction matrices from Exp 1, Exp 2, and Exp 3...")
    df1 = pd.read_csv(params['exp1_preds'])
    df2 = pd.read_csv(params['exp2_preds'])
    df3 = pd.read_csv(params['exp3_preds'])
    
    # Isolate the keys and labels from each experiment
    df1_sub = df1[['timestamp', 'drone_id', 'anomaly_label']].rename(columns={'anomaly_label': 'label_exp1'})
    df2_sub = df2[['timestamp', 'drone_id', 'anomaly_label']].rename(columns={'anomaly_label': 'label_exp2'})
    df3_sub = df3[['timestamp', 'drone_id', 'anomaly_label']].rename(columns={'anomaly_label': 'label_exp3'})
    
    # Execute Relational Inner-Joins to align the temporal events
    logger.info("Executing relational temporal alignment across matrices...")
    df_merged = df1_sub.merge(df2_sub, on=['timestamp', 'drone_id'], how='inner')
    df_merged = df_merged.merge(df3_sub, on=['timestamp', 'drone_id'], how='inner')
    
    # Strict Intersection: All three algorithms must output -1 (Anomaly)
    df_merged['ensemble_label'] = np.where(
        (df_merged['label_exp1'] == -1) & 
        (df_merged['label_exp2'] == -1) & 
        (df_merged['label_exp3'] == -1), 
        -1, 1
    )
    
    df_results = df3.copy()
    if 'anomaly_score' in df_results.columns:
        df_results = df_results.drop(columns=['anomaly_label', 'anomaly_score'])
    else:
        df_results = df_results.drop(columns=['anomaly_label'])
        
    # Map the ensemble decisions back to the full feature matrix
    df_results = df_results.merge(df_merged[['timestamp', 'drone_id', 'ensemble_label']], on=['timestamp', 'drone_id'], how='left')
    df_results.rename(columns={'ensemble_label': 'anomaly_label'}, inplace=True)
    
    # Synthesize a decision score for the forensic evaluation script
    df_results['anomaly_score'] = np.where(df_results['anomaly_label'] == -1, -100, 100)
    
    anomaly_count = len(df_results[df_results['anomaly_label'] == -1])
    logger.info(f"Ensemble Complete. The tribunal mathematically agreed on exactly {anomaly_count} high-confidence anomalies.")
    
    return None, df_results, []

def execute_ensemble_tribunal(params):
    """
    Executes a Multi-Model Tribunal.
    Supports 'strict_intersection' (unanimous) or 'majority_vote' (>= 2 models).
    """
    strategy = params.get('ensemble_params', {}).get('strategy', 'strict_intersection')
    logger.info(f"Initializing Meta-Ensemble Tribunal using strategy: {strategy}")
    
    logger.info("Loading prediction matrices from Exp 1, Exp 2, and Exp 3...")
    df1 = pd.read_csv(params['data_paths']['exp1_preds'])
    df2 = pd.read_csv(params['data_paths']['exp2_preds'])
    df3 = pd.read_csv(params['data_paths']['exp3_preds'])
    
    df1_sub = df1[['timestamp', 'drone_id', 'anomaly_label']].rename(columns={'anomaly_label': 'label_exp1'})
    df2_sub = df2[['timestamp', 'drone_id', 'anomaly_label']].rename(columns={'anomaly_label': 'label_exp2'})
    df3_sub = df3[['timestamp', 'drone_id', 'anomaly_label']].rename(columns={'anomaly_label': 'label_exp3'})
    
    logger.info("Executing relational temporal alignment...")
    df_merged = df1_sub.merge(df2_sub, on=['timestamp', 'drone_id'], how='inner')
    df_merged = df_merged.merge(df3_sub, on=['timestamp', 'drone_id'], how='inner')
    
    # Calculate total anomaly votes per timestamp (True = 1 vote)
    anomaly_votes = (df_merged['label_exp1'] == -1).astype(int) + \
                    (df_merged['label_exp2'] == -1).astype(int) + \
                    (df_merged['label_exp3'] == -1).astype(int)
    
    if strategy == "strict_intersection":
        # Requires all 3 votes
        df_merged['ensemble_label'] = np.where(anomaly_votes == 3, -1, 1)
    elif strategy == "majority_vote":
        # Requires 2 or more votes
        df_merged['ensemble_label'] = np.where(anomaly_votes >= 2, -1, 1)
    else:
        logger.error(f"Unknown ensemble strategy: {strategy}")
        sys.exit(1)
        
    df_results = df3.copy()
    if 'anomaly_score' in df_results.columns:
        df_results = df_results.drop(columns=['anomaly_label', 'anomaly_score'])
    else:
        df_results = df_results.drop(columns=['anomaly_label'])
        
    df_results = df_results.merge(df_merged[['timestamp', 'drone_id', 'ensemble_label']], on=['timestamp', 'drone_id'], how='left')
    df_results.rename(columns={'ensemble_label': 'anomaly_label'}, inplace=True)
    
    # Synthesize decision scores based on vote confidence
    # 3 votes = -100 (High Confidence Anomaly)
    # 2 votes = -50  (Moderate Confidence Anomaly)
    # <2 votes = 100 (Normal)
    score_mapping = np.where(anomaly_votes == 3, -100, np.where(anomaly_votes == 2, -50, 100))
    df_results['anomaly_score'] = score_mapping
    
    anomaly_count = len(df_results[df_results['anomaly_label'] == -1])
    logger.info(f"Ensemble Complete. The {strategy} tribunal escalated {anomaly_count} anomalies.")
    
    return None, df_results, []

def train_kmeans_hybrid(df, params):
    """
    Trains a K-Means clustering model to define distinct operating states.
    Anomalies are defined as the points furthest from their assigned cluster centroid.
    """
    logger.info(f"Initializing K-Means Hybrid with {params['n_clusters']} operating states...")
    
    features = df.select_dtypes(include=[np.number]).columns
    features = [col for col in features if col not in ['drone_id']]
    X = df[features]
    
    # Distance-based clustering strictly requires scaling
    logger.info("Standardizing temporal feature space for geometric clustering...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Fit K-Means to establish the Centroids (Normal Operating States)
    kmeans = KMeans(
        n_clusters=params['n_clusters'], 
        random_state=params['random_state'], 
        n_init='auto'
    )
    logger.info("Fitting K-Means centroids...")
    kmeans.fit(X_scaled)
    
    # 2. Calculate the distance from every point to its closest centroid
    logger.info("Calculating Euclidean distances to nearest operating states...")
    closest, distances = pairwise_distances_argmin_min(X_scaled, kmeans.cluster_centers_)
    
    # 3. Define the Anomaly Boundary based on the target contamination rate
    # We find the threshold distance where the top 5% furthest points lie.
    threshold_distance = np.percentile(distances, 100 * (1 - params['contamination']))
    
    # If distance > threshold, it's an anomaly (-1), else normal (1)
    predictions = np.where(distances > threshold_distance, -1, 1)
    
    # For decision scores, K-Means doesn't have a native one.
    # We will use the negative distance so that lower (more negative) = more anomalous
    anomaly_scores = -distances
    
    # We must package the Scaler and KMeans into a Pipeline for serialization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=params['n_clusters'], random_state=params['random_state'], n_init='auto'))
    ])
    pipeline.fit(X)
    
    df_results = df.copy()
    df_results['anomaly_label'] = predictions
    df_results['anomaly_score'] = anomaly_scores
    
    anomaly_count = len(df_results[df_results['anomaly_label'] == -1])
    logger.info(f"K-Means Hybrid complete. Isolated {anomaly_count} spatial anomalies.")
    
    return pipeline, df_results, features

if __name__ == "__main__":
    # Allow passing different config files via command line for later experiments
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config/exp1_baseline.json'
    config = load_config(config_file)
    
    # Setup MLflow Tracking
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run(run_name=f"Train_{config['model_params']['algorithm']}"):
        logger.info(f"--- Starting Experiment: {config['experiment_name']} ---")
        
        # 1. Conditional Data Loading
        if config['model_params']['algorithm'] != "Ensemble":
            input_path = config['data_paths']['input']
            logger.info(f"Loading training matrix from {input_path}")
            df = pd.read_csv(input_path)
        else:
            df = None
            
        # 2. Log Hyperparameters
        for key, value in config['model_params'].items():
            mlflow.log_param(key, value)
            
        # 3. Algorithm Routing
        if config['model_params']['algorithm'] == "IsolationForest":
            model, df_predictions, used_features = train_isolation_forest(df, config['model_params'])
        elif config['model_params']['algorithm'] == "OneClassSVM":
            model, df_predictions, used_features = train_one_class_svm(df, config['model_params'])
        elif config['model_params']['algorithm'] == "LOF_PCA_Pipeline":
            model, df_predictions, used_features = train_lof_pca_pipeline(df, config['model_params'])
        elif config['model_params']['algorithm'] == "KMeans_Distance":
            model, df_predictions, used_features = train_kmeans_hybrid(df, config['model_params'])
        elif config['model_params']['algorithm'] == "Ensemble":
            model, df_predictions, used_features = execute_ensemble_tribunal(config)
        else:
            logger.error(f"Algorithm {config['model_params']['algorithm']} not yet implemented.")
            sys.exit(1)
            
        # 4. Save Predictions for Forensic Evaluation
        output_path = config['data_paths']['output_predictions']
        df_predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # 5. Save Model via Pickle (Skip for Ensemble as there is no single model object)
        if model is not None:
            model_path = config['data_paths']['model_path']
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model successfully serialized and saved to {model_path}")
            
            # Log Model to MLflow
            mlflow.sklearn.log_model(model, f"{config['model_params']['algorithm']}_model")
            mlflow.log_metric("feature_count", len(used_features))
            
        # 6. Log Final Metrics
        anomaly_ratio = len(df_predictions[df_predictions['anomaly_label'] == -1]) / len(df_predictions)
        mlflow.log_metric("actual_anomaly_ratio", anomaly_ratio)
        
        logger.info("Experiment run finalized and tracked in MLflow.")