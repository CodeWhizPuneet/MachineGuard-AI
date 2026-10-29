"""
MachineGuard AI: Data Ingestion and Stratified Splitting
Loads AI4I 2020 dataset, performs EDA summaries, and creates train/val/test splits with leakage prevention.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_FILE,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE_OF_TRAIN_VAL,
    TARGET_COLUMN,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from src.validation import clean_column_names, validate_raw_dataset, ValidationError

def load_raw_dataset(csv_path: str = str(DATA_FILE)) -> pd.DataFrame:
    """Load dataset from disk and ensure clean column names."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Please run 'python scripts/download_data.py' first."
        )

    df = clean_column_names(df)
    is_valid, errors = validate_raw_dataset(df)
    if not is_valid:
        raise ValidationError(f"Dataset validation failed: {errors}")

    return df

def get_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive EDA statistics:
    - Dimensions, missing values, duplicates
    - Target distribution (class imbalance)
    - Numeric descriptive statistics
    - Machine Type failure rates
    - Failure subtype counts (for historical insight only, NOT for training)
    """
    total_rows = len(df)
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    failure_rate = float(df[TARGET_COLUMN].mean())

    missing_counts = df.isnull().sum().to_dict()
    duplicate_count = int(df.duplicated().sum())

    # Numeric summary
    numeric_cols = NUMERICAL_FEATURES
    num_summary = df[numeric_cols].describe().to_dict()

    # Type failure rate
    type_stats = {}
    for mtype, group in df.groupby("Type"):
        type_stats[mtype] = {
            "total": int(len(group)),
            "failures": int(group[TARGET_COLUMN].sum()),
            "failure_rate": float(group[TARGET_COLUMN].mean())
        }

    # Subtype distribution (for informational purposes)
    subtype_counts = {}
    for sub in LEAKAGE_COLUMNS:
        if sub in df.columns:
            subtype_counts[sub] = int(df[sub].sum())

    # Correlation matrix of numeric features + target
    corr_cols = NUMERICAL_FEATURES + [TARGET_COLUMN]
    corr_matrix = df[corr_cols].corr().round(4).to_dict()

    return {
        "total_observations": total_rows,
        "total_features": df.shape[1],
        "missing_values": missing_counts,
        "duplicate_rows": duplicate_count,
        "target_distribution": {
            "negative_0": int(target_counts.get(0, 0)),
            "positive_1": int(target_counts.get(1, 0)),
            "failure_rate_percent": round(failure_rate * 100, 2)
        },
        "numeric_summary": num_summary,
        "type_failure_rates": type_stats,
        "failure_subtypes": subtype_counts,
        "correlation_matrix": corr_matrix,
        "leakage_rationale": {
            "excluded_identifiers": ID_COLUMNS,
            "excluded_subtypes": LEAKAGE_COLUMNS,
            "reason": (
                "UDI/Product ID are arbitrary index keys causing memorization. "
                "TWF, HDF, PWF, OSF, RNF are post-event failure modes that directly "
                "reveal the target and constitute critical data leakage."
            )
        }
    }

def prepare_splits(
    df: pd.DataFrame,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create strictly isolated, stratified Train (64%), Validation (16%), and Test (20%) sets.
    Features X are stripped of ID columns, leakage columns, and target column.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = clean_column_names(df)

    # Exclude IDs, Leakage Subtypes, and Target from predictor features
    drop_cols = ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET_COLUMN]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Verify expected predictors
    expected = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    for exp in expected:
        if exp not in feature_cols:
            raise ValidationError(f"Expected feature '{exp}' missing from features.")

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # Step 1: Split into Train+Val (80%) and Test (20%) with stratification
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=random_seed
    )

    # Step 2: Split Train+Val into Train (80% of 80% = 64%) and Val (20% of 80% = 16%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VAL_SIZE_OF_TRAIN_VAL,
        stratify=y_train_val,
        random_state=random_seed
    )

    # Sanity checks on target distribution
    for name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        pos = int(split_y.sum())
        total = len(split_y)
        pct = (pos / total) * 100
        if pos == 0:
            raise ValueError(f"Split {name} has 0 positive failure instances! Stratification error.")

    return X_train, X_val, X_test, y_train, y_val, y_test
