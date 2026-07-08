# Customer Churn Predictive Analytics Engine

An intermediate-level predictive behavioral analytics pipeline and interactive dashboard engineered to identify attrition indicators and calculate real-time customer churn probabilities.

## The Core Uniqueness
This engine shifts focus away from simple binary classification by deploying a HistGradientBoostingClassifier to output granular, calibrated risk probabilities. The pipeline integrates a specialized preprocessing strategy that handles missing values and utilizes Synthetic Minority Over-sampling Technique (SMOTE) to balance the minority attrition class without inducing data leakage.

## Tech Stack & Requirements
* Language: Python
* Machine Learning: Scikit-Learn (HistGradientBoostingClassifier), Imbalanced-Learn (SMOTE)
* Data Processing: Pandas, NumPy
* Artifact Serialization: Joblib
* Frontend Interface: Streamlit

## Pipeline Architecture
1. Telemetry Ingestion: Loading and profiling transactional and engagement customer data.
2. Imbalance Remediation: Applying SMOTE to synthesize minority churn vectors inside the training partition.
3. Model Training: Fitting a gradient boosted tree structure optimized via cross-validation for Area Under the Precision-Recall Curve (PR-AUC).
4. Artifact Serialization: Exporting the trained model weights and scaling parameters using Joblib for production pipeline isolation.

## Hardware Strategy
Optimized for local or low-resource cloud runtimes. The pipeline handles tabular customer data entirely within memory constraints matching a standard 8GB RAM development system, maintaining rapid inference speeds without requiring external GPU hardware.