# Drone Anomaly Detection 

## 1. Problem Formulation & Significance
Drone fleets operate in complex, hostile environments where relying on predefined threat signatures (malware hashes, static threshold alerts) is insufficient. Latent hardware degradation and cyber attacks (e.g., Command & Control hijacking, GPS spoofing) rarely trigger single-metric alarms. 

**The Mission:** To engineer a dual-sensor, unsupervised machine learning architecture capable of independently monitoring physical kinematics and network health, flagging the top 5% most structurally abnormal events for human forensic review.

## 2. Core Hypotheses & Methodology
1. **The Kinematic Hypothesis:** Unsupervised anomaly detection on multi-variate telemetry can isolate physical failures undetectable by static alerts.
2. **The Cyber Hypothesis:** Latent cyber threats manifest as sustained statistical outliers in network logs without requiring predefined signatures.
3. **The Temporal State Hypothesis:** UAV failures are sustained events. Incorporating 5-hour rolling windows and lag variables yields higher-fidelity anomaly boundaries than point-in-time analysis.

## 3. Project Architecture & MLOps
* **`DVC` (Data Version Control):** Tracks multi-gigabyte matrix transformations and model artifacts.
* **`MLflow`:** Centralized logging engine for hyperparameter tracking and model registry.
* **`Optuna`:** Automated Bayesian optimization to maximize the unsupervised decision boundary gap.
* **`Evidently AI`:** Automated data drift monitoring for operational stability.

## 4. Model Card
Following a 6-experiment evaluation phase (including PCA+LOF, One-Class SVMs, and Strict Tribunals), a finalized champion model was selected and optimized.

* **Architecture:** Temporally-Shifted Isolation Forest
* **Dimensionality:** 195 engineered features (Dual-Sensor Fusion)
* **Contamination Rate:** 0.05 (Top 5% Anomaly Isolation)
* **Optimized Hyperparameters:** `n_estimators: 50`, `max_samples: 0.5`, `max_features: 0.5`
* **Performance:** Successfully isolated sustained command floods and rogue port routing by evaluating the orthogonal variance between physical and cyber feature spaces.

### The Ablation Proof
An ablation study confirmed that removing the temporal rolling windows or decoupling the cyber/kinematic sensors resulted in blind spots to complex, multi-vector attacks, proving the necessity of the 195-dimensional matrix.

## 5. Execution Guide
To reproduce the experimental pipeline:
1. Install dependencies
`pip install -r requirements.txt`

2. Pull datasets via DVC
`dvc pull` 

3. Feature Engineering
`python src/features.py`

4. Experiments
`python src/train.py config/exp1_baseline.json` (Iterate 1-6)

5. Evaluations
`python src/evaluate.py data/exp5_predictions.csv`

6. Visualizations
`python src/visualize.py data/exp1_predictions.csv Exp1_IsolationForest` (Iterate 1-6)

7. Ablation Study
`python src/ablation.py`

8. Optimization
`python src/optimize.py`

9. Monitoring with Evidently AI
`python src/evidently_report.py`

10. MLflow Dashboard
`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`