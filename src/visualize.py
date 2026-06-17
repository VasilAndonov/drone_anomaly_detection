import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os
from utils import get_logger

logger = get_logger('VisualizationEngine')

def plot_feature_density(df, feature_name, experiment_name):
    """
    Generates a Kernel Density Estimate (KDE) plot showing the 
    distribution shift between Normal operations and Anomalies.
    """
    logger.info(f"Generating KDE Density Plot for {feature_name}...")
    
    plt.figure(figsize = (10, 6))
    sns.kdeplot(data = df[df['anomaly_label'] == 1], x = feature_name, 
                fill = True, color = 'steelblue', label = 'Normal State', alpha = 0.5)
    sns.kdeplot(data = df[df['anomaly_label'] == -1], x = feature_name, 
                fill = True, color = 'crimson', label = 'Anomalous State', alpha = 0.5)
    
    plt.title(f"Distribution Shift: {feature_name} ({experiment_name})", fontsize = 14, fontweight = 'bold')
    plt.xlabel(feature_name.replace('_', ' ').title())
    plt.ylabel("Density (Frequency)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_name}_{feature_name}_density.png", dpi = 300)
    plt.close()

def plot_anomaly_radar_fingerprint(df, experiment_name):
    """
    Generates a Radar (Spider) chart comparing the average normal state 
    against the absolute worst anomaly, creating a visual 'Threat Fingerprint'.
    """
    logger.info("Generating Threat Fingerprint Radar Chart...")
    normal = df[df['anomaly_label'] == 1]
    anomalies = df[df['anomaly_label'] == -1]
    
    if len(anomalies) == 0:
        return
        
    radar_features = ['velocity_x', 'acceleration_z', 'packet_loss_rate', 'control_command_frequency', 'cpu_usage', 'battery_level_pct']
    radar_features = [f for f in radar_features if f in df.columns]
    
    if len(radar_features) < 3:
        return 
    
    normal_means = normal[radar_features].mean().values
    
    if 'anomaly_score' in df.columns:
        worst_idx = df[df['anomaly_score'] == df['anomaly_score'].min()].index[0]
        worst_anomaly = df.loc[worst_idx, radar_features].values
    else:
        worst_anomaly = anomalies[radar_features].mean().values

    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint = False).tolist()
    normal_means = np.concatenate((normal_means, [normal_means[0]]))
    worst_anomaly = np.concatenate((worst_anomaly, [worst_anomaly[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize = (8, 8), subplot_kw = dict(polar = True))
    ax.plot(angles, normal_means, color = 'steelblue', linewidth = 2, label = 'Normal Baseline')
    ax.fill(angles, normal_means, color = 'steelblue', alpha = 0.25)
    ax.plot(angles, worst_anomaly, color = 'crimson', linewidth = 2, linestyle = 'dashed', label = 'Severe Anomaly')
    ax.fill(angles, worst_anomaly, color = 'crimson', alpha = 0.15)
    
    ax.set_yticklabels([]) 
    ax.set_xticks(angles[:-1])
    clean_labels = [f.replace('_', '\n').title() for f in radar_features]
    ax.set_xticklabels(clean_labels, fontsize=10)
    
    plt.title(f"Threat Fingerprint\n{experiment_name}", size = 15, fontweight = 'bold', pad = 20)
    plt.legend(loc='upper right', bbox_to_anchor = (1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_name}_radar_fingerprint.png", dpi = 300, bbox_inches = 'tight')
    plt.close()

def generate_experiment_plots(csv_path, experiment_name):
    """
    Generates all Spatial, Temporal, and Threat plots.
    """
    logger.info(f"Generating visual reports for {experiment_name}...")
    os.makedirs('plots', exist_ok = True)
    
    try:
        df = pd.read_csv(csv_path)
        df = df.copy() # Fixes the Pandas Fragmentation PerformanceWarning
    except FileNotFoundError:
        logger.error(f"Could not find {csv_path}. Skipping.")
        return

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

    normal = df[df['anomaly_label'] == 1]
    anomalies = df[df['anomaly_label'] == -1]

    if len(anomalies) == 0:
        logger.warning(f"No anomalies found in {experiment_name}. Plots may be uninformative.")

    # 1. Spatial Manifold (PCA Scatter)
    logger.info("Computing 2D PCA for Spatial Plot...")
    features = df.select_dtypes(include=[np.number]).columns
    eval_cols = [c for c in features if c not in ['anomaly_label', 'anomaly_score', 'drone_id']]
    
    X = df[eval_cols]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components = 2, random_state = 42)
    X_pca = pca.fit_transform(X_scaled)
    
    df['PCA_1'] = X_pca[:, 0]
    df['PCA_2'] = X_pca[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    plt.scatter(df[df['anomaly_label'] == 1]['PCA_1'], df[df['anomaly_label'] == 1]['PCA_2'], 
                c = 'steelblue', label = 'Normal Operations', alpha = 0.3, s = 15)
    plt.scatter(df[df['anomaly_label'] == -1]['PCA_1'], df[df['anomaly_label'] == -1]['PCA_2'], 
                c = 'crimson', label = f'Anomalies (n = {len(anomalies)})', alpha = 0.9, s = 40, edgecolors = 'black')
    
    plt.title(f"Spatial Distribution: {experiment_name}", fontsize=14, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_name}_spatial_manifold.png", dpi = 300)
    plt.close()
    
    # 2. Temporal Threat Timeline
    logger.info("Rendering Temporal Threat Timeline...")
    plt.figure(figsize=(14, 6))
    
    if 'anomaly_score' in df.columns:
        df['plot_score'] = -df['anomaly_score']
        plt.plot(df['timestamp'], df['plot_score'], color = 'gray', alpha = 0.5, label = 'Decision Score')
        plt.scatter(anomalies['timestamp'], -anomalies['anomaly_score'], 
                    color = 'crimson', label = 'Detected Anomaly', zorder = 5, s = 30)
        plt.ylabel("Threat Score (Higher = More Anomalous)")
    else:
        plt.scatter(normal['timestamp'], normal['anomaly_label'], color = 'steelblue', label = 'Normal')
        plt.scatter(anomalies['timestamp'], anomalies['anomaly_label'], color = 'crimson', label = 'Anomaly')
        plt.ylabel("State")
        
    plt.title(f"Temporal Threat Timeline: {experiment_name}", fontsize = 14, fontweight = 'bold')
    plt.xlabel("Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{experiment_name}_temporal_timeline.png", dpi = 300)
    plt.close()

    # 3 & 4. Advanced Profiling (KDE and Radar)
    plot_feature_density(df, 'packet_loss_rate', experiment_name)
    plot_anomaly_radar_fingerprint(df, experiment_name)
    
    logger.info(f"Successfully generated ALL visual reports for {experiment_name}.")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        csv_path = sys.argv[1]
        exp_name = sys.argv[2]
        generate_experiment_plots(csv_path, exp_name)
    else:
        logger.error("Usage: python src/visualize.py <path_to_csv> <experiment_name>")