import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import IsolationForest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "error_analysis")

os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

def load_environment():
    """
    Simulates loading the 195-dimensional Dual-Sensor test matrix and the Champion Model.
    Replace this with your actual DVC data loader and MLflow model loader if running in production.
    """
    print("Loading Dual-Sensor Matrix and Champion Model...")
    
    np.random.seed(42)
    X_dual = pd.DataFrame({
        'battery_level_pct': np.random.normal(50, 15, 9107),
        'cpu_usage': np.random.normal(40, 10, 9107),
        'packet_loss_rate': np.random.lognormal(-4, 1, 9107),
        'control_command_frequency': np.random.normal(10, 2, 9107),
        'altitude_m_rolling_mean': np.random.normal(80, 20, 9107),
        'velocity_x_rolling_std': np.random.normal(5, 1, 9107)
    })
    
    # Inject synthetic boundary noise to simulate the 5% contamination
    X_dual.iloc[:455, :] += np.random.normal(10, 5, X_dual.iloc[:455, :].shape)
    
    # Load your Optuna-optimized Isolation Forest
    model = IsolationForest(
        n_estimators=200, 
        max_samples=0.5, 
        max_features=0.5, 
        contamination=0.05, 
        random_state=42
    )
    model.fit(X_dual)
    
    return X_dual, model

def analyze_boundary_uncertainty(X, model):
    """Evaluates the geometric margins and defines the Zone of Uncertainty."""
    print("Calculating Geometric Margins and Zone of Uncertainty...")
    
    scores = model.decision_function(X)
    predictions = model.predict(X) 
    
    results_df = X.copy()
    results_df['decision_score'] = scores
    results_df['prediction'] = predictions
    
    # Define the "Zone of Uncertainty"
    margin = 0.02 
    results_df['is_borderline'] = np.abs(results_df['decision_score']) <= margin
    
    uncertain_count = results_df['is_borderline'].sum()
    print(f"Found {uncertain_count} Borderline Events (Potential False Positives/Negatives).")
    
    return results_df, margin

def plot_uncertainty_distribution(results_df, margin):
    """Plots the distribution of scores and highlights the error-prone boundary."""
    print("Generating Uncertainty Distribution Plot...")
    
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df['decision_score'], bins=100, color="steelblue")
    
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0.0)')
    plt.axvspan(-margin, margin, color='red', alpha=0.15, label='Zone of Uncertainty (Errors)')
    
    plt.title("Error Analysis: Model Decision Boundary and Zone of Uncertainty", fontsize=14, fontweight='bold')
    plt.xlabel("Decision Function Score (Negative = Anomaly, Positive = Normal)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "1_uncertainty_distribution.png"), dpi=300)
    plt.close()

def plot_borderline_feature_variance(results_df):
    """Plots a bivariate scatter to show where the model is confused."""
    print("Generating Borderline Feature Variance Scatter...")
    
    plt.figure(figsize=(10, 6))
    
    plot_df = results_df.copy()
    plot_df['Classification'] = plot_df['prediction'].map({
        1: 'Normal (High Confidence)', 
        -1: 'Anomaly (High Confidence)'
    })
    
    # Normal & Anomalies 
    sns.scatterplot(
        data=plot_df[~plot_df['is_borderline']], 
        x='battery_level_pct', y='packet_loss_rate', 
        hue='Classification', 
        palette={'Normal (High Confidence)': 'lightblue', 'Anomaly (High Confidence)': 'darkred'}, 
        alpha=0.3, edgecolor=None
    )
    
    # Errors (Borderline)
    sns.scatterplot(
        data=plot_df[plot_df['is_borderline']], 
        x='battery_level_pct', y='packet_loss_rate', 
        color='yellow', edgecolor='black', s=80, marker='X', label='Borderline (Errors)'
    )
    
    plt.title("Spatial Vulnerability: Where the Champion Model Struggles", fontsize=14, fontweight='bold')
    plt.xlabel("Physical Domain: Battery Level (%)")
    plt.ylabel("Cyber Domain: Packet Loss Rate")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "2_borderline_scatter.png"), dpi=300)
    plt.close()

def execute_shap_error_explainability(X, model, results_df):
    """Uses SHAP TreeExplainer to find which features are responsible for errors."""
    print("Calculating SHAP Values for Error Explainability (This may take a moment)...")
    
    X_errors = X[results_df['is_borderline']]
    
    if X_errors.empty:
        print("No borderline errors found to explain.")
        return
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_errors)
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_errors, show=False)
    plt.title("SHAP Analysis: Features Driving Borderline Errors", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "3_shap_error_summary.png"), dpi=300)
    plt.close()

    # 2. SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_errors, plot_type="bar", show=False, color="crimson")
    plt.title("Mean Absolute SHAP Value (Magnitude of Confusion)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "4_shap_error_bar.png"), dpi=300)
    plt.close()

def extract_confusion_examples(results_df):
    """Extracts the exact telemetry rows to write Section 6.6."""
    print("\n=== DIAGNOSTIC EXTRACTION: BORDERLINE CONFUSIONS ===")
    
    borderline_cases = results_df[results_df['is_borderline'] == True].copy()
    
    if borderline_cases.empty:
        return
        
    borderline_cases['absolute_score'] = borderline_cases['decision_score'].abs()
    most_confused = borderline_cases.sort_values(by='absolute_score')
    
    columns_to_view = [
        'decision_score', 
        'battery_level_pct', 
        'packet_loss_rate', 
        'velocity_x_rolling_std',
        'control_command_frequency'
    ]
    
    print("Top 3 Most Confused Operational States:")
    print(most_confused[columns_to_view].head(3).to_string())
    print("\n===================================================")

if __name__ == "__main__":
    print("===================================================")
    print("   Champion Model Deep Error Analysis Pipeline     ")
    print("===================================================")
    
    X, champion_model = load_environment()
    results, margin_val = analyze_boundary_uncertainty(X, champion_model)
    
    plot_uncertainty_distribution(results, margin_val)
    plot_borderline_feature_variance(results)
    execute_shap_error_explainability(X, champion_model, results)
    
    extract_confusion_examples(results)
    
    print(f"\n Pipeline Complete. All diagnostic plots saved to '{PLOT_DIR}'.")