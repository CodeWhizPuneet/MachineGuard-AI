"""
Automated Unit Tests: Single and Batch Inference
Tests prediction calculation, risk threshold classification, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from src.predict import ModelInferenceEngine
from src.validation import ValidationError

@pytest.fixture
def inference_engine():
    """Returns inference engine instance."""
    engine = ModelInferenceEngine()
    return engine

def test_single_prediction_structure(inference_engine):
    sample = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 30
    }
    result = inference_engine.predict_single(sample)
    assert "predicted_probability" in result
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert "predicted_class" in result
    assert result["predicted_class"] in [0, 1]
    assert "risk_band" in result
    assert "explanation" in result
    assert "contributions" in result["explanation"]

def test_single_prediction_invalid_input(inference_engine):
    invalid = {
        "Type": "INVALID",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 30
    }
    with pytest.raises(ValidationError):
        inference_engine.predict_single(invalid)

def test_batch_prediction_flow(inference_engine):
    batch = pd.DataFrame([
        {
            "UDI": 1,
            "Product ID": "L001",
            "Type": "L",
            "Air temperature [K]": 298.5,
            "Process temperature [K]": 308.5,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10
        },
        {
            "UDI": 2,
            "Product ID": "M002",
            "Type": "M",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 311.0,
            "Rotational speed [rpm]": 1300,
            "Torque [Nm]": 65.0,  # High torque
            "Tool wear [min]": 210 # High tool wear
        }
    ])
    results_df, invalid_df, summary = inference_engine.predict_batch(batch)
    assert len(results_df) == 2
    assert len(invalid_df) == 0
    assert "Failure_Probability" in results_df.columns
    assert "Predicted_Class" in results_df.columns
    assert "Risk_Category" in results_df.columns
    # Row 2 with high torque and high tool wear should have higher failure probability than Row 1
    prob1 = results_df.loc[0, "Failure_Probability"]
    prob2 = results_df.loc[1, "Failure_Probability"]
    assert prob2 > prob1

def test_threshold_sensitivity(inference_engine):
    sample = {
        "Type": "L",
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 311.0,
        "Rotational speed [rpm]": 1400,
        "Torque [Nm]": 50.0,
        "Tool wear [min]": 150
    }
    # With threshold 0.01, should almost certainly flag failure
    res_low = inference_engine.predict_single(sample, threshold_override=0.01)
    # With threshold 0.99, should not flag failure
    res_high = inference_engine.predict_single(sample, threshold_override=0.99)
    assert res_low["decision_threshold"] == 0.01
    assert res_high["decision_threshold"] == 0.99
    assert res_low["predicted_class"] >= res_high["predicted_class"]
