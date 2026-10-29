"""
MachineGuard AI: Inference Engine for Single and Batch Predictions
Loads trained artifacts, executes validation, evaluates calibrated risk, and generates results.
"""

from typing import Dict, Any, Tuple, Optional
import json
import numpy as np
import pandas as pd
import joblib

from src.config import (
    MODEL_PIPELINE_FILE,
    METADATA_FILE,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RISK_BANDS
)
from src.validation import validate_inference_record, validate_batch_dataframe, ValidationError
from src.explainability import explain_single_prediction

class ModelInferenceEngine:
    """Encapsulates model loading and inference logic."""

    def __init__(self):
        self.pipeline = None
        self.metadata = {}
        self.threshold = 0.5
        self.load_model()

    def load_model(self) -> bool:
        """Load trained pipeline and metadata from artifacts directory."""
        if not MODEL_PIPELINE_FILE.exists():
            print(f"[WARNING] Model file {MODEL_PIPELINE_FILE} not found. Please train model first.")
            return False

        try:
            self.pipeline = joblib.load(MODEL_PIPELINE_FILE)
            if METADATA_FILE.exists():
                with open(METADATA_FILE, "r") as f:
                    self.metadata = json.load(f)
                self.threshold = float(self.metadata.get("selected_threshold", 0.5))
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model pipeline: {e}")
            return False

    def categorize_risk(self, probability: float, threshold: Optional[float] = None) -> Dict[str, str]:
        """Categorize failure risk into actionable operational bands."""
        t = threshold if threshold is not None else self.threshold
        if probability >= t:
            return {
                "category": "HIGH_RISK",
                "label": "Elevated / Critical Risk (Maintenance Recommended)",
                "action": "Schedule immediate physical inspection; check spindle, torque, and tooling.",
                "color": "#ef4444"
            }
        elif probability >= t * 0.7:
            return {
                "category": "ELEVATED_RISK",
                "label": "Moderate / Elevated Risk (Increased Monitoring)",
                "action": "Increase monitoring cadence; review temperature and lubrication.",
                "color": "#f59e0b"
            }
        else:
            return {
                "category": "LOW_RISK",
                "label": "Lower Risk (Normal Operation)",
                "action": "Nominal operational parameters observed. Routine monitoring.",
                "color": "#10b981"
            }

    def predict_single(
        self,
        record: Dict[str, Any],
        threshold_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Predict failure probability and explanation for a single machine record.
        """
        if self.pipeline is None:
            if not self.load_model():
                raise RuntimeError("Model pipeline is not loaded. Train the model first.")

        # Validate record
        is_valid, errors = validate_inference_record(record)
        if not is_valid:
            raise ValidationError(f"Invalid input record: {errors}")

        # Construct single-row DataFrame
        df = pd.DataFrame([{
            "Type": str(record["Type"]).upper(),
            "Air temperature [K]": float(record["Air temperature [K]"]),
            "Process temperature [K]": float(record["Process temperature [K]"]),
            "Rotational speed [rpm]": float(record["Rotational speed [rpm]"]),
            "Torque [Nm]": float(record["Torque [Nm]"]),
            "Tool wear [min]": float(record["Tool wear [min]"])
        }])

        proba = float(self.pipeline.predict_proba(df)[0, 1])
        t = threshold_override if threshold_override is not None else self.threshold
        risk_info = self.categorize_risk(proba, t)

        # Generate local explanation
        explanation = explain_single_prediction(
            self.pipeline, df, proba, t
        )

        return {
            "predicted_probability": round(proba, 4),
            "predicted_probability_percent": round(proba * 100, 2),
            "decision_threshold": round(t, 4),
            "predicted_class": int(proba >= t),
            "prediction_label": "FAILURE RISK DETECTED" if proba >= t else "NORMAL OPERATION",
            "risk_band": risk_info["label"],
            "recommended_action": risk_info["action"],
            "risk_color": risk_info["color"],
            "explanation": explanation,
            "inputs_verified": record
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold_override: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Predict failure probabilities for an uploaded batch CSV.
        Returns:
            results_df: DataFrame with original row IDs, probabilities, predictions, and risk categories
            invalid_df: DataFrame of skipped/invalid rows with errors
            summary: summary statistics of predictions
        """
        if self.pipeline is None:
            if not self.load_model():
                raise RuntimeError("Model pipeline is not loaded.")

        t = threshold_override if threshold_override is not None else self.threshold

        valid_df, invalid_df, warnings = validate_batch_dataframe(df)

        if len(valid_df) == 0:
            return pd.DataFrame(), invalid_df, {"total_rows": len(df), "valid_rows": 0, "invalid_rows": len(invalid_df)}

        # Predict probabilities
        probabilities = self.pipeline.predict_proba(valid_df)[:, 1]
        classes = (probabilities >= t).astype(int)

        results_df = valid_df.copy()
        # Add identifiers back if present in source
        if "UDI" in df.columns:
            results_df.insert(0, "UDI", df.loc[valid_df.index, "UDI"])
        if "Product ID" in df.columns:
            col_pos = 1 if "UDI" in results_df.columns else 0
            results_df.insert(col_pos, "Product ID", df.loc[valid_df.index, "Product ID"])

        results_df["Failure_Probability"] = np.round(probabilities, 4)
        results_df["Failure_Probability_%"] = np.round(probabilities * 100, 2)
        results_df["Predicted_Class"] = classes
        results_df["Risk_Category"] = [self.categorize_risk(p, t)["label"] for p in probabilities]

        summary = {
            "total_input_rows": len(df),
            "valid_evaluated_rows": len(valid_df),
            "invalid_skipped_rows": len(invalid_df),
            "decision_threshold_used": t,
            "predicted_failures_count": int(classes.sum()),
            "predicted_failure_rate_%": round(float(classes.mean() * 100), 2),
            "mean_failure_probability_%": round(float(probabilities.mean() * 100), 2),
            "warnings": warnings
        }

        return results_df, invalid_df, summary

# Singleton engine instance for reuse
_ENGINE = None

def get_inference_engine() -> ModelInferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ModelInferenceEngine()
    return _ENGINE
