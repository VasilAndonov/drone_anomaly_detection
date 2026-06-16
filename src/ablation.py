import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import IsolationForest
from utils import get_logger, load_config

logger = get_logger('AblationStudy')

def calculate_boundary_gap(model, X):
    """Calculates the distance between the mean Normal score and mean Anomaly score."""
    scores = model.decision_function(X)
    predictions = model.predict(X)
    
    normal_scores = scores[predictions == 1]
    anomaly_scores = scores[predictions == -1]
    
    if len(anomaly_scores) == 0:
        return 0.0
    return np.mean(normal_scores) - np.mean(anomaly_scores)

def execute_ablation(df):
    """Systematically removes features to prove the necessity of the Dual-Sensor pipeline."""
    logger.info("--- Initiating Feature Ablation Study on Champion Model (Isolation Forest) ---")
    
    all_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'drone_id']
    
    # 1. Define the Feature Sub-Spaces
    kinematic_base = ['altitude_m', 'velocity', 'acceleration', 'battery', 'hover', 'cpu']
    cyber_base = ['packet', 'signal', 'error', 'response', 'port', 'encryption']
    
    feature_sets = {
        "Baseline_Full_Temporal_Pipeline": all_cols,
        "Ablation_1_Only_Kinematic": [c for c in all_cols if any(k in c for k in kinematic_base)],
        "Ablation_2_Only_Cyber": [c for c in all_cols if any(cy in c for cy in cyber_base)],
        "Ablation_3_No_Temporal_Memory": [c for c in all_cols if 'rolling' not in c and 'lag' not in c]
    }
    
    for study_name, columns in feature_sets.items():
        if not columns:
            continue
            
        logger.info(f"Training {study_name} ({len(columns)} dimensions)...")
        X_subset = df[columns]
        
        # Train identical models to ensure a fair test
        model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        model.fit(X_subset)
        
        gap = calculate_boundary_gap(model, X_subset)
        
        with mlflow.start_run(run_name=f"Ablation_{study_name}"):
            mlflow.log_param("ablation_state", study_name)
            mlflow.log_metric("feature_count", len(columns))
            mlflow.log_metric("decision_boundary_gap", gap)
            
        logger.info(f"Result -> Boundary Gap: {gap:.4f}")

if __name__ == "__main__":
    mlflow.set_experiment("Ablation_Study")
    
    # We load the engineered temporal features for the ablation tests
    df = pd.read_csv("data/engineered_features.csv")
    execute_ablation(df)
    logger.info("Ablation Study Complete.")