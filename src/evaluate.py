import pandas as pd
import numpy as np
import sys
from utils import get_logger

# Initialize Logger
logger = get_logger('ForensicEvaluation')

def forensic_profiling(df):
    """
    Reverse-engineers the anomaly detection boundary to provide
    Root Cause Explainability and Feature Distribution Profiling.
    """
    logger.info("Starting Forensic Interpretability Profiling...")
    
    # 1. Anomaly Separation
    normal = df[df['anomaly_label'] == 1]
    anomalies = df[df['anomaly_label'] == -1]
    
    logger.info(f"Density Separation: {len(normal)} Normal Events vs {len(anomalies)} Anomalies.")
    
    if len(anomalies) == 0:
        logger.warning("No anomalies found to evaluate.")
        return

    # Isolate numeric columns for mathematical comparison
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    eval_cols = [col for col in numeric_cols if col not in ['anomaly_label', 'anomaly_score', 'timestamp', 'drone_id']]
    
    # Calculate baseline distributions
    normal_means = normal[eval_cols].mean()
    normal_stds = normal[eval_cols].std()
    
    # 2. Root Cause Explainability (The "Worst" Anomaly)
    # The lowest decision score is the deepest, most severe anomaly
    most_severe_idx = anomalies['anomaly_score'].idxmin()
    most_severe_event = df.loc[most_severe_idx]
    
    timestamp = most_severe_event.get('timestamp', 'Unknown Time')
    logger.info(f"--- ROOT CAUSE EXPLAINABILITY ---")
    logger.info(f"Most severe anomaly detected at: {timestamp}")
    
    # Calculate how many standard deviations the event's features are from the normal mean
    event_values = most_severe_event[eval_cols]
    
    # Prevent division by zero if a feature has no variance
    safe_stds = normal_stds.replace(0, 1e-9) 
    z_scores = np.abs((event_values - normal_means) / safe_stds)
    
    top_factors = z_scores.sort_values(ascending=False).head(4)
    
    for feature, z in top_factors.items():
        norm_val = normal_means[feature]
        anom_val = event_values[feature]
        logger.info(f"Trigger: {feature} | Normal Mean: {norm_val:.2f} | Anomaly Value: {anom_val:.2f} | Deviation: {z:.2f} Z-Scores")

    # 3. Feature Distribution Profiling (Global View)
    # Which features shift the most, globally, when an anomaly occurs?
    mean_shifts = (np.abs(normal_means - anomalies[eval_cols].mean()) / np.abs(normal_means.replace(0, 1e-9))) * 100
    top_global_shifts = mean_shifts.sort_values(ascending=False).head(3)
    
    logger.info(f"--- GLOBAL DISTRIBUTION PROFILING ---")
    for feature, shift in top_global_shifts.items():
        logger.info(f"Macro Indicator: {feature} | Cluster Distribution Shift: {shift:.1f}%")

if __name__ == "__main__":
    # Default to Exp 1 predictions, but allow command-line overrides
    pred_file = sys.argv[1] if len(sys.argv) > 1 else 'data/exp1_predictions.csv'
    logger.info(f"Loading predictions matrix from {pred_file}")
    
    try:
        df_preds = pd.read_csv(pred_file)
        forensic_profiling(df_preds)
        logger.info("Forensic evaluation complete.")
    except FileNotFoundError:
        logger.error(f"Prediction file not found: {pred_file}. Have you run train.py yet?")