"""
MachineGuard AI: Input Validation and Schema Enforcement
Ensures dataset integrity, prevents target leakage, and validates input features.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    FEATURE_RANGES,
)

class ValidationError(Exception):
    """Custom exception for schema or data validation failures."""
    pass

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and BOM artifacts from dataframe column headers."""
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df

def validate_raw_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the official raw training dataset.
    Must contain all predictor features, the target, and identifiers.
    """
    df = clean_column_names(df)
    errors = []

    required_cols = (
        ID_COLUMNS
        + CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
        + [TARGET_COLUMN]
        + LEAKAGE_COLUMNS
    )

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns in dataset: {missing}")

    if TARGET_COLUMN in df.columns:
        unique_targets = set(df[TARGET_COLUMN].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            errors.append(f"Target '{TARGET_COLUMN}' contains non-binary values: {unique_targets}")

    # Check for empty dataframe
    if len(df) == 0:
        errors.append("Dataset is empty.")

    return len(errors) == 0, errors

def validate_inference_record(record: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """
    Validate a single machine input record for inference.
    Returns (is_valid, error_dict).
    """
    errors = {}

    # Check Categorical
    for cat in CATEGORICAL_FEATURES:
        val = record.get(cat)
        if val is None or str(val).strip() == "":
            errors[cat] = f"Field '{cat}' is required."
        elif str(val).upper() not in FEATURE_RANGES[cat]["allowed"]:
            allowed = FEATURE_RANGES[cat]["allowed"]
            errors[cat] = f"Invalid '{cat}': '{val}'. Must be one of {allowed}."

    # Check Numerical
    for num in NUMERICAL_FEATURES:
        val = record.get(num)
        if val is None or val == "":
            errors[num] = f"Field '{num}' is required."
            continue
        try:
            num_val = float(val)
            r = FEATURE_RANGES[num]
            if np.isnan(num_val):
                errors[num] = f"Field '{num}' cannot be NaN."
            elif num_val < r["min"] or num_val > r["max"]:
                errors[num] = (
                    f"'{num}' value {num_val} is outside plausible physical range "
                    f"[{r['min']}, {r['max']}] {r.get('unit', '')}."
                )
        except (ValueError, TypeError):
            errors[num] = f"Field '{num}' must be a valid number."

    return len(errors) == 0, errors

def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Validate a batch uploaded CSV for prediction.
    - Strips any leakage or target columns (never feeds them to model)
    - Separates valid rows from invalid rows
    - Preserves row identifier or index for tracing
    Returns:
        valid_df (features only),
        invalid_df (with failure reasons),
        warnings
    """
    df = clean_column_names(df)
    warnings = []

    # Check if target or leakage columns were present and notify user of strict exclusion
    excluded_found = [c for c in (LEAKAGE_COLUMNS + [TARGET_COLUMN]) if c in df.columns]
    if excluded_found:
        warnings.append(
            f"Note: Excluded target/leakage columns {excluded_found} from model inputs to prevent data leakage."
        )

    # Check required model features
    required_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValidationError(
            f"Uploaded CSV is missing mandatory features: {missing_features}. "
            f"Expected: {required_features}"
        )

    # Preserve row index / identifier
    if "UDI" in df.columns:
        row_ids = df["UDI"]
    elif "Product ID" in df.columns:
        row_ids = df["Product ID"]
    else:
        row_ids = df.index

    valid_indices = []
    invalid_rows = []

    for idx, row in df.iterrows():
        rec = row.to_dict()
        is_val, errs = validate_inference_record(rec)
        if is_val:
            valid_indices.append(idx)
        else:
            invalid_rows.append({
                "source_row": row_ids.loc[idx],
                "error_reason": "; ".join([f"{k}: {v}" for k, v in errs.items()])
            })

    valid_df = df.loc[valid_indices, required_features].copy()
    invalid_df = pd.DataFrame(invalid_rows)

    return valid_df, invalid_df, warnings
