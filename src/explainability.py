"""
MachineGuard AI: Explainable AI Module
Provides global feature importance and local instance-level prediction explanations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.config import (
    FEATURE_IMPORTANCE_FILE,
    METADATA_FILE,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    FEATURE_RANGES
)

def load_global_feature_importance() -> Dict[str, Any]:
    """Load precomputed permutation and tree feature importances from artifacts."""
    if FEATURE_IMPORTANCE_FILE.exists():
        import json
        with open(FEATURE_IMPORTANCE_FILE, "r") as f:
            return json.load(f)
    return {}

def explain_single_prediction(
    pipeline: Any,
    input_df: pd.DataFrame,
    predicted_probability: float,
    threshold: float
) -> Dict[str, Any]:
    """
    Generate local explainability for an individual machine record:
    - Calculates baseline population reference median
    - Calculates directional sensitivity impact of each feature
    - Explains in plain English whether the parameter elevated or mitigated failure probability
    - Distinguishes statistical correlation from mechanical root-cause causation.
    """
    contributions = []

    # Plausible baseline machine operating conditions
    baseline_values = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0,
        "Type": "M"
    }

    # Evaluate each numerical feature against reference and evaluate sensitivity
    current_row = input_df.iloc[0].to_dict()

    for col in NUMERICAL_FEATURES:
        val = float(current_row[col])
        ref = float(baseline_values[col])
        diff = val - ref
        norm_diff = diff / (FEATURE_RANGES[col]["max"] - FEATURE_RANGES[col]["min"])

        # Create counterfactual perturbation (replace feature with baseline)
        counterfactual_df = input_df.copy()
        counterfactual_df.loc[0, col] = ref
        cf_prob = float(pipeline.predict_proba(counterfactual_df)[0, 1])

        impact_delta = predicted_probability - cf_prob  # positive means this feature increased failure prob
        impact_pct = impact_delta * 100.0

        if abs(impact_delta) > 0.01:
            direction = "INCREASED RISK" if impact_delta > 0 else "REDUCED RISK"
            narrative = (
                f"{col} ({val} {FEATURE_RANGES[col]['unit']}) {direction.lower()} "
                f"predicted failure probability by {abs(impact_pct):.1f}% relative to reference baseline ({ref})."
            )
        else:
            direction = "NEUTRAL"
            narrative = f"{col} ({val} {FEATURE_RANGES[col]['unit']}) had negligible impact on risk (<1%)."

        contributions.append({
            "feature": col,
            "actual_value": val,
            "unit": FEATURE_RANGES[col]["unit"],
            "reference_value": ref,
            "impact_delta_probability": round(float(impact_delta), 4),
            "impact_percentage": round(float(impact_pct), 2),
            "direction": direction,
            "narrative": narrative
        })

    # Machine Type effect
    current_type = str(current_row.get("Type", "M"))
    cf_type = input_df.copy()
    cf_type.loc[0, "Type"] = "M" if current_type != "M" else "L"
    cf_prob_type = float(pipeline.predict_proba(cf_type)[0, 1])
    type_delta = predicted_probability - cf_prob_type
    contributions.append({
        "feature": "Type",
        "actual_value": current_type,
        "unit": "variant",
        "reference_value": "M",
        "impact_delta_probability": round(float(type_delta), 4),
        "impact_percentage": round(float(type_delta * 100.0), 2),
        "direction": "INCREASED RISK" if type_delta > 0.01 else ("REDUCED RISK" if type_delta < -0.01 else "NEUTRAL"),
        "narrative": f"Machine variant {current_type} vs standard baseline variant M."
    })

    # Sort contributions by absolute impact
    contributions.sort(key=lambda x: abs(x["impact_delta_probability"]), reverse=True)

    summary_statement = (
        f"Prediction indicates a {predicted_probability*100:.1f}% failure probability "
        f"(Decision Threshold: {threshold:.3f}). "
        f"Top contributing risk factors: {', '.join([c['feature'] for c in contributions[:2] if c['direction'] == 'INCREASED RISK']) or 'None significant'}."
    )

    limitations = (
        "Engineering Caution: Feature importance measures statistical association "
        "and sensitivity in the model's learned decision surface. It does not constitute "
        "physical proof of causal mechanical failure. On-site physical inspection is required."
    )

    return {
        "predicted_probability": round(float(predicted_probability), 4),
        "threshold": round(float(threshold), 4),
        "predicted_class": int(predicted_probability >= threshold),
        "contributions": contributions,
        "summary": summary_statement,
        "limitations": limitations
    }
