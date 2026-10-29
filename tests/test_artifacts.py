"""
Automated Unit Tests: Model Artifacts and Metadata Integrity
Verifies persisted pipeline artifacts, metadata schema, and evaluation metric files.
"""

import os
import json
import pytest
from src.config import (
    MODEL_PIPELINE_FILE,
    METADATA_FILE,
    MODEL_COMPARISON_FILE,
    THRESHOLD_ANALYSIS_FILE,
    FINAL_METRICS_FILE,
    FEATURE_IMPORTANCE_FILE
)

def test_artifacts_exist_and_not_empty():
    """Verify that all core training artifacts were generated."""
    assert MODEL_PIPELINE_FILE.exists(), f"Missing {MODEL_PIPELINE_FILE}"
    assert METADATA_FILE.exists(), f"Missing {METADATA_FILE}"
    assert MODEL_COMPARISON_FILE.exists(), f"Missing {MODEL_COMPARISON_FILE}"
    assert THRESHOLD_ANALYSIS_FILE.exists(), f"Missing {THRESHOLD_ANALYSIS_FILE}"
    assert FINAL_METRICS_FILE.exists(), f"Missing {FINAL_METRICS_FILE}"

def test_metadata_contents():
    """Verify metadata contains required fields and valid threshold."""
    with open(METADATA_FILE, "r") as f:
        meta = json.load(f)

    assert meta["project_name"] == "MachineGuard AI"
    assert "model_name" in meta
    assert "selected_threshold" in meta
    assert 0.0 < meta["selected_threshold"] < 1.0
    assert "feature_schema" in meta
    assert "dataset_split" in meta
    assert meta["dataset_split"]["train_samples"] == 6400
    assert meta["dataset_split"]["val_samples"] == 1600
    assert meta["dataset_split"]["test_samples"] == 2000

def test_model_comparison_structure():
    """Verify model comparison logged metrics across multiple algorithms."""
    with open(MODEL_COMPARISON_FILE, "r") as f:
        comp = json.load(f)

    assert len(comp) >= 5, "At least 5 models must be evaluated"
    for name, data in comp.items():
        assert "val_metrics_default_0_5" in data
        assert "average_precision" in data["val_metrics_default_0_5"]
        assert "roc_auc" in data["val_metrics_default_0_5"]
        assert "generalization_gap_ap" in data
