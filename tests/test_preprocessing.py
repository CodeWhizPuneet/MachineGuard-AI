"""
Automated Unit Tests: Preprocessing and Stratified Data Splits
Tests pipeline encoding, scaling, and strict 64/16/20 split stratification.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import create_preprocessor, get_transformed_feature_names
from src.data_loader import prepare_splits
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

@pytest.fixture
def mock_dataset():
    """Generates synthetic dataset for split and preprocessing testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "UDI": list(range(1, n + 1)),
        "Product ID": [f"M_{i}" for i in range(n)],
        "Type": np.random.choice(["L", "M", "H"], size=n),
        "Air temperature [K]": np.random.uniform(295.0, 305.0, size=n),
        "Process temperature [K]": np.random.uniform(305.0, 315.0, size=n),
        "Rotational speed [rpm]": np.random.randint(1200, 2500, size=n),
        "Torque [Nm]": np.random.uniform(20.0, 70.0, size=n),
        "Tool wear [min]": np.random.randint(0, 220, size=n),
        "Machine failure": np.random.choice([0, 1], size=n, p=[0.90, 0.10]),
        "TWF": [0] * n,
        "HDF": [0] * n,
        "PWF": [0] * n,
        "OSF": [0] * n,
        "RNF": [0] * n,
    })
    return df

def test_preprocessing_pipeline_fit_transform(mock_dataset):
    X = mock_dataset[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    preprocessor = create_preprocessor(scale_numeric=True)
    transformed = preprocessor.fit_transform(X)

    # 3 one-hot columns for Type (L, M, H) + 5 numerical columns = 8 columns
    feature_names = get_transformed_feature_names(preprocessor)
    assert transformed.shape[1] == 8
    assert len(feature_names) == 8
    assert "Type_L" in feature_names or "Type_x0_L" in feature_names or any("L" in f for f in feature_names)

def test_stratified_split_ratios_and_isolation(mock_dataset):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(mock_dataset, random_seed=42)

    total = len(mock_dataset)
    # Check proportions: Train ~ 64%, Val ~ 16%, Test ~ 20%
    assert len(X_train) == int(total * 0.64)
    assert len(X_val) == int(total * 0.16)
    assert len(X_test) == int(total * 0.20)

    # Check both positive and negative classes exist in each split
    assert y_train.sum() > 0
    assert (len(y_train) - y_train.sum()) > 0
    assert y_val.sum() > 0
    assert y_test.sum() > 0

    # Ensure no index overlap (strict data isolation)
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(val_indices)) == 0
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(val_indices.intersection(test_indices)) == 0
