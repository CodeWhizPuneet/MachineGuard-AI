"""
MachineGuard AI: Configuration and Global Hyperparameters
Central configuration for data splits, feature selection, leakage prevention, and model artifacts.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_FILE = DATA_DIR / "ai4i2020.csv"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for strict reproducibility
RANDOM_SEED = 42

# Data Splitting Strategy:
# 64% Training, 16% Validation, 20% Test (Stratified)
TRAIN_RATIO = 0.64
VAL_RATIO = 0.16
TEST_RATIO = 0.20
# Step 1: Split into (Train + Val = 80%) and (Test = 20%)
# Step 2: Split (Train + Val) into Train (64/80 = 80%) and Val (16/80 = 20%)
TEST_SIZE = 0.20
VAL_SIZE_OF_TRAIN_VAL = 0.20  # 0.20 * 0.80 = 0.16

# Cross-Validation Folds inside Training Set
CV_FOLDS = 5

# Target Column
TARGET_COLUMN = "Machine failure"

# Identification and Target Leakage Columns to Strictly Exclude from Predictors
ID_COLUMNS = ["UDI", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Feature Definitions
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Physical Feature Limits for Validation and Domain Sanity Checks
FEATURE_RANGES = {
    "Air temperature [K]": {"min": 290.0, "max": 310.0, "default": 300.0, "unit": "K"},
    "Process temperature [K]": {"min": 300.0, "max": 320.0, "default": 310.0, "unit": "K"},
    "Rotational speed [rpm]": {"min": 1000, "max": 3000, "default": 1500, "unit": "rpm"},
    "Torque [Nm]": {"min": 1.0, "max": 100.0, "default": 40.0, "unit": "Nm"},
    "Tool wear [min]": {"min": 0, "max": 300, "default": 60, "unit": "min"},
    "Type": {"allowed": ["L", "M", "H"], "default": "M"}
}

# Artifact file names
MODEL_PIPELINE_FILE = ARTIFACTS_DIR / "machineguard_pipeline.joblib"
METADATA_FILE = ARTIFACTS_DIR / "model_metadata.json"
MODEL_COMPARISON_FILE = ARTIFACTS_DIR / "model_comparison.json"
THRESHOLD_ANALYSIS_FILE = ARTIFACTS_DIR / "threshold_analysis.json"
FEATURE_IMPORTANCE_FILE = ARTIFACTS_DIR / "feature_importance.json"
FINAL_METRICS_FILE = ARTIFACTS_DIR / "final_test_metrics.json"

# Operational Risk Categories based on Probability and Threshold
RISK_BANDS = {
    "LOW_RISK": {"label": "Lower Risk (Normal Operation)", "color": "#10b981"},
    "ELEVATED_RISK": {"label": "Elevated Risk (Monitor Operating Parameters)", "color": "#f59e0b"},
    "HIGH_RISK": {"label": "Critical Risk (Immediate Maintenance Required)", "color": "#ef4444"}
}
