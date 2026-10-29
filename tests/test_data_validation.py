"""
Automated Unit Tests: Data Validation and Leakage Prevention
Verifies schema constraints, target isolation, and exclusion of leakage features.
"""

import pytest
import pandas as pd
import numpy as np
from src.config import ID_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN
from src.validation import (
    validate_raw_dataset,
    validate_inference_record,
    validate_batch_dataframe,
    clean_column_names,
    ValidationError
)

@pytest.fixture
def sample_valid_raw_df():
    """Generates a valid raw dataset mock matching AI4I 2020 schema."""
    n = 20
    return pd.DataFrame({
        "UDI": list(range(1, n + 1)),
        "Product ID": [f"L471{i}" for i in range(n)],
        "Type": ["L" if i % 2 == 0 else "M" for i in range(n)],
        "Air temperature [K]": np.linspace(298.0, 304.0, n),
        "Process temperature [K]": np.linspace(308.0, 314.0, n),
        "Rotational speed [rpm]": np.random.randint(1400, 1600, n),
        "Torque [Nm]": np.linspace(35.0, 55.0, n),
        "Tool wear [min]": np.random.randint(0, 200, n),
        "Machine failure": [1 if i % 10 == 0 else 0 for i in range(n)],
        "TWF": [0] * n,
        "HDF": [0] * n,
        "PWF": [0] * n,
        "OSF": [0] * n,
        "RNF": [0] * n,
    })

def test_clean_column_names():
    df = pd.DataFrame({" \ufeffUDI ": [1], " Type ": ["M"]})
    cleaned = clean_column_names(df)
    assert "UDI" in cleaned.columns
    assert "Type" in cleaned.columns

def test_validate_raw_dataset_success(sample_valid_raw_df):
    is_valid, errors = validate_raw_dataset(sample_valid_raw_df)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_raw_dataset_missing_columns():
    df_broken = pd.DataFrame({"Type": ["L"], "Machine failure": [0]})
    is_valid, errors = validate_raw_dataset(df_broken)
    assert is_valid is False
    assert len(errors) > 0

def test_validate_inference_record_valid():
    valid_rec = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 50
    }
    is_valid, errors = validate_inference_record(valid_rec)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_inference_record_out_of_range():
    invalid_rec = {
        "Type": "X",  # Invalid type
        "Air temperature [K]": 500.0,  # Physically impossible for machine environment
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 50
    }
    is_valid, errors = validate_inference_record(invalid_rec)
    assert is_valid is False
    assert "Type" in errors
    assert "Air temperature [K]" in errors

def test_validate_batch_dataframe_excludes_leakage():
    batch_df = pd.DataFrame({
        "UDI": [101, 102],
        "Product ID": ["M101", "M102"],
        "Type": ["M", "L"],
        "Air temperature [K]": [298.1, 298.2],
        "Process temperature [K]": [308.6, 308.7],
        "Rotational speed [rpm]": [1551, 1408],
        "Torque [Nm]": [42.8, 46.3],
        "Tool wear [min]": [0, 3],
        "Machine failure": [0, 1],  # Ground truth if uploaded
        "TWF": [0, 1]              # Leakage column
    })
    valid_df, invalid_df, warnings = validate_batch_dataframe(batch_df)
    assert len(valid_df) == 2
    assert "Machine failure" not in valid_df.columns
    assert "TWF" not in valid_df.columns
    assert "UDI" not in valid_df.columns
    assert len(warnings) > 0  # Warning regarding excluded leakage columns
