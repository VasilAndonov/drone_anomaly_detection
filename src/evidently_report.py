import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from utils import get_logger, load_config
import os

logger = get_logger('EvidentlyAI')

def generate_drift_report(df, output_path="data/data_drift_report.html"):
    """
    Generates an interactive Evidently AI Data Drift report.
    Splits the operational timeline into 'Reference' (Past) and 'Current' (Present).
    """
    logger.info("Initializing Evidently AI Data Drift Analysis...")
    
    # Isolate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # To simulate production drift monitoring, we split the dataset chronologically
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.5)
    
    reference_data = df_sorted.iloc[:split_idx][numeric_cols]
    current_data = df_sorted.iloc[split_idx:][numeric_cols]
    
    logger.info(f"Reference Set: {len(reference_data)} logs. Current Set: {len(current_data)} logs.")
    
    # Generate the Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save the interactive HTML report
    report.save_html(output_path)
    logger.info(f"Evidently AI interactive report successfully generated: {output_path}")

if __name__ == "__main__":
    df = pd.read_csv("data/engineered_features.csv")
    generate_drift_report(df)