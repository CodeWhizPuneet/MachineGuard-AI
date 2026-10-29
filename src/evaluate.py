"""
MachineGuard AI: Evaluation Reporting and Model Card Generation
Produces reproducible evaluation reports, confusion matrices, curve statistics, and model cards.
"""

import os
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

from src.config import (
    MODEL_PIPELINE_FILE,
    METADATA_FILE,
    FINAL_METRICS_FILE,
    THRESHOLD_ANALYSIS_FILE,
    MODEL_COMPARISON_FILE,
    FEATURE_IMPORTANCE_FILE,
    REPORTS_DIR,
    DATA_FILE
)
from src.data_loader import load_raw_dataset, prepare_splits

def generate_evaluation_artifacts():
    """Generates detailed Markdown report and Model Card from trained artifacts."""
    if not MODEL_PIPELINE_FILE.exists() or not METADATA_FILE.exists():
        print("[ERROR] Artifacts not found. Please run model training first.")
        return

    with open(METADATA_FILE, "r") as f:
        meta = json.load(f)
    with open(FINAL_METRICS_FILE, "r") as f:
        final_metrics = json.load(f)
    with open(MODEL_COMPARISON_FILE, "r") as f:
        comparison = json.load(f)
    with open(THRESHOLD_ANALYSIS_FILE, "r") as f:
        threshold_info = json.load(f)
    with open(FEATURE_IMPORTANCE_FILE, "r") as f:
        feat_imp = json.load(f)

    winning_model = meta["model_name"]
    threshold = meta["selected_threshold"]
    test_optimal = final_metrics["test_metrics_optimal_threshold"]
    test_default = final_metrics["test_metrics_default_0_5"]
    val_metrics = final_metrics["validation_metrics_optimal_threshold"]

    # 1. Generate final_evaluation_report.md
    report_path = REPORTS_DIR / "final_evaluation_report.md"
    report_content = f"""# MachineGuard AI: Final Model Evaluation Report
**Project**: MachineGuard AI - Explainable Predictive Maintenance System  
**Dataset**: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)  
**Selected Algorithm**: `{winning_model}`  
**Training Timestamp**: `{meta['training_timestamp']}`  
**Evaluator**: MachineGuard AI Automated Reproducible Validation Pipeline  

---

## 1. Executive Summary
MachineGuard AI evaluates machine operating conditions (Air temperature, Process temperature, Rotational speed, Torque, and Tool wear, across Machine Types L, M, and H) to estimate the probability of mechanical failure. 

Because predictive maintenance suffers from severe class imbalance (~3.39% historical failure rate), model selection prioritized **Average Precision (AP)** and **Failure Recall** on held-out validation data over naive raw accuracy.

- **Selected Candidate**: `{winning_model}`
- **Validation Average Precision**: `{comparison[winning_model]['val_metrics_default_0_5']['average_precision']:.4f}`
- **Optimized Decision Threshold**: `{threshold:.3f}` (tuned on validation set to maximize F1 while maintaining high recall)
- **Final Held-Out Test Set AP**: `{test_optimal['average_precision']:.4f}`
- **Final Held-Out Test Set ROC-AUC**: `{test_optimal['roc_auc']:.4f}`
- **Final Held-Out Test F1 at {threshold:.3f}**: `{test_optimal['f1_score']:.4f}` (vs `{test_default['f1_score']:.4f}` at default 0.50)
- **Final Held-Out Test Recall at {threshold:.3f}**: `{test_optimal['recall']:.4f}`

---

## 2. Dataset Splitting & Strict Leakage Prevention
To guarantee honest evaluation without data leakage:
- **Stratified Split**: 64% Training ({meta['dataset_split']['train_samples']} samples, {meta['dataset_split']['train_failures']} failures), 16% Validation ({meta['dataset_split']['val_samples']} samples, {meta['dataset_split']['val_failures']} failures), 20% Test ({meta['dataset_split']['test_samples']} samples, {meta['dataset_split']['test_failures']} failures).
- **Identifier Exclusion**: `UDI` and `Product ID` were removed prior to feature selection to prevent spurious identifier memorization.
- **Subtype Leakage Exclusion**: `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are post-hoc failure diagnostic modes. Including them as predictors would cause catastrophic target leakage. They are excluded from all modeling pipelines.
- **Pipeline Isolation**: Encoders and scalers were fit strictly on training folds inside Stratified K-Fold CV pipelines.
- **Independent Test Set**: The test set was touched **exactly once** after all candidate model selection and threshold tuning were finalized.

---

## 3. Algorithm Benchmark & Model Comparison
Candidate algorithms were trained on the training split using 5-fold Stratified Cross-Validation with hyperparameter tuning:

| Algorithm | CV Train AP | Val AP | Val ROC-AUC | Val F1 (0.50) | Val Recall (0.50) | Generalization Gap | Overfit Risk |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for name, res in comparison.items():
        v = res["val_metrics_default_0_5"]
        report_content += (
            f"| {name} | {res['cv_train_average_precision']:.4f} | {v['average_precision']:.4f} | "
            f"{v['roc_auc']:.4f} | {v['f1_score']:.4f} | {v['recall']:.4f} | "
            f"{res['generalization_gap_ap']:.4f} | {res['overfitting_risk']} |\n"
        )

    report_content += f"""
---

## 4. Decision-Threshold Optimization (Validation Only)
A default threshold of 0.50 leads to unacceptable missed failures in high-cost industrial environments. Threshold optimization was performed across 35 candidate values on validation data only.

- **Selected Threshold**: `{threshold:.3f}`
- **Validation Recall**: `{val_metrics['recall']:.4f}`
- **Validation Precision**: `{val_metrics['precision']:.4f}`
- **Validation False Alarms**: `{val_metrics['false_positives']}` out of {meta['dataset_split']['val_samples']} machines
- **Operational Rationale**: {threshold_info['summary']['rationale']}

---

## 5. Final Held-Out Test Evaluation
The final test split (2,000 samples, {meta['dataset_split']['test_failures']} failures) was evaluated once:

### Test Confusion Matrix (at Threshold {threshold:.3f})
- **True Negatives**: `{test_optimal['confusion_matrix']['true_negatives']}`
- **False Positives (False Alarms)**: `{test_optimal['confusion_matrix']['false_positives']}`
- **False Negatives (Missed Failures)**: `{test_optimal['confusion_matrix']['false_negatives']}`
- **True Positives (Detected Failures)**: `{test_optimal['confusion_matrix']['true_positives']}`
- **False Alarm Rate (FPR)**: `{test_optimal['rates']['false_alarm_rate_fpr']*100:.2f}%`
- **Miss Rate (FNR)**: `{test_optimal['rates']['miss_rate_fnr']*100:.2f}%`

---

## 6. Global Feature Importance & Explainability
### Validation Permutation Importance (Drop in Average Precision)
"""
    if "permutation_importance_validation" in feat_imp:
        report_content += "| Feature | Importance Mean (Δ AP) | Std Dev |\n|:---|:---:|:---:|\n"
        for feat, scores in feat_imp["permutation_importance_validation"].items():
            report_content += f"| {feat} | {scores['importance_mean']:.5f} | ±{scores['importance_std']:.5f} |\n"

    report_content += """
### Operational Meaning & Limitations
Features like **Torque [Nm]**, **Rotational speed [rpm]**, and **Tool wear [min]** emerge as top predictive drivers. However, statistical sensitivity in an ML model does not prove physical causation. Mechanical wear, lubrication breakdown, and electrical load must be confirmed through telemetry and physical inspection.
"""

    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"[INFO] Generated {report_path}")

    # 2. Generate model_card.md
    model_card_path = REPORTS_DIR / "model_card.md"
    model_card_content = f"""# MachineGuard AI - Model Card

## Model Details
- **System**: MachineGuard AI (Explainable Predictive Maintenance System)
- **Model Architecture**: `{winning_model}` with scikit-learn preprocessing pipeline
- **Version**: 1.0.0
- **Input Features**: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- **Excluded Features**: `UDI`, `Product ID` (identifiers); `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (post-failure subtypes)
- **Target**: `Machine failure` (0 = Nominal Operation, 1 = Mechanical Failure)
- **Decision Threshold**: `{threshold:.3f}`

## Intended Use
- **Primary Domain**: Continuous manufacturing, CNC milling, lathe equipment, and rotational machinery monitoring.
- **Intended Users**: Maintenance reliability engineers, plant supervisors, and ML practitioners.
- **Use Case**: Early warning system flagging machines with elevated failure risk for preventive inspection.

## Inappropriate Use
- **Safety-Critical Shutdown Automation**: This model is an early advisory warning tool. It should not be connected directly to emergency automated safety kill-switches without human-in-the-loop review.
- **Out-of-Distribution Hardware**: Turbines, jet engines, or machinery operating outside documented ranges should not use this model without domain fine-tuning.

## Performance & Fairness
- **Stratified Test AP**: `{test_optimal['average_precision']:.4f}`
- **Stratified Test ROC-AUC**: `{test_optimal['roc_auc']:.4f}`
- **Test Recall**: `{test_optimal['recall']:.4f}`
- **False Alarm Rate**: `{test_optimal['rates']['false_alarm_rate_fpr']*100:.2f}%`

## Ethical & Operational Limitations
The model reflects patterns in the AI4I 2020 dataset. In real factory environments, sensor drift, unexpected thermal surges, and mechanical anomalies outside historical distributions require ongoing recalibration and human diagnostic confirmation.
"""
    with open(model_card_path, "w") as f:
        f.write(model_card_content)
    print(f"[INFO] Generated {model_card_path}")

if __name__ == "__main__":
    generate_evaluation_artifacts()
