# MachineGuard AI: Final Model Evaluation Report
**Project**: MachineGuard AI - Explainable Predictive Maintenance System  
**Dataset**: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)  
**Selected Algorithm**: `Gradient Boosting`  
**Training Timestamp**: `2026-10-29T00:09:25Z`  
**Evaluator**: MachineGuard AI Automated Reproducible Validation Pipeline  

---

## 1. Executive Summary
MachineGuard AI evaluates machine operating conditions (Air temperature, Process temperature, Rotational speed, Torque, and Tool wear, across Machine Types L, M, and H) to estimate the probability of mechanical failure. 

Because predictive maintenance suffers from severe class imbalance (~3.39% historical failure rate), model selection prioritized **Average Precision (AP)** and **Failure Recall** on held-out validation data over naive raw accuracy.

- **Selected Candidate**: `Gradient Boosting`
- **Validation Average Precision**: `0.7687`
- **Optimized Decision Threshold**: `0.475` (tuned on validation set to maximize F1 while maintaining high recall)
- **Final Held-Out Test Set AP**: `0.8004`
- **Final Held-Out Test Set ROC-AUC**: `0.9635`
- **Final Held-Out Test F1 at 0.475**: `0.8000` (vs `0.8000` at default 0.50)
- **Final Held-Out Test Recall at 0.475**: `0.7353`

---

## 2. Dataset Splitting & Strict Leakage Prevention
To guarantee honest evaluation without data leakage:
- **Stratified Split**: 64% Training (6400 samples, 217 failures), 16% Validation (1600 samples, 54 failures), 20% Test (2000 samples, 68 failures).
- **Identifier Exclusion**: `UDI` and `Product ID` were removed prior to feature selection to prevent spurious identifier memorization.
- **Subtype Leakage Exclusion**: `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are post-hoc failure diagnostic modes. Including them as predictors would cause catastrophic target leakage. They are excluded from all modeling pipelines.
- **Pipeline Isolation**: Encoders and scalers were fit strictly on training folds inside Stratified K-Fold CV pipelines.
- **Independent Test Set**: The test set was touched **exactly once** after all candidate model selection and threshold tuning were finalized.

---

## 3. Algorithm Benchmark & Model Comparison
Candidate algorithms were trained on the training split using 5-fold Stratified Cross-Validation with hyperparameter tuning:

| Algorithm | CV Train AP | Val AP | Val ROC-AUC | Val F1 (0.50) | Val Recall (0.50) | Generalization Gap | Overfit Risk |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Dummy Classifier (Baseline) | 0.0000 | 0.0348 | 0.5103 | 0.0541 | 0.0556 | -0.0011 | Low |
| Logistic Regression | 0.4544 | 0.4551 | 0.8926 | 0.2985 | 0.1852 | 0.0038 | Low |
| Random Forest | 0.6624 | 0.5932 | 0.9594 | 0.5606 | 0.6852 | 0.4022 | High |
| Extra Trees | 0.6862 | 0.7397 | 0.9540 | 0.5333 | 0.3704 | 0.2602 | Moderate |
| Gradient Boosting | 0.7798 | 0.7687 | 0.9753 | 0.7368 | 0.6481 | 0.2117 | Moderate |
| Hist Gradient Boosting | 0.7699 | 0.6892 | 0.9741 | 0.6458 | 0.5741 | 0.3108 | High |

---

## 4. Decision-Threshold Optimization (Validation Only)
A default threshold of 0.50 leads to unacceptable missed failures in high-cost industrial environments. Threshold optimization was performed across 35 candidate values on validation data only.

- **Selected Threshold**: `0.475`
- **Validation Recall**: `0.6667`
- **Validation Precision**: `0.8372`
- **Validation False Alarms**: `7` out of 1600 machines
- **Operational Rationale**: Threshold 0.475 selected by maximizing F1-score (0.7423) on held-out validation data. Achieves recall=0.6667 with 7 false alarms out of 1600 machines.

---

## 5. Final Held-Out Test Evaluation
The final test split (2,000 samples, 68 failures) was evaluated once:

### Test Confusion Matrix (at Threshold 0.475)
- **True Negatives**: `1925`
- **False Positives (False Alarms)**: `7`
- **False Negatives (Missed Failures)**: `18`
- **True Positives (Detected Failures)**: `50`
- **False Alarm Rate (FPR)**: `0.36%`
- **Miss Rate (FNR)**: `26.47%`

---

## 6. Global Feature Importance & Explainability
### Validation Permutation Importance (Drop in Average Precision)
| Feature | Importance Mean (Δ AP) | Std Dev |
|:---|:---:|:---:|
| Torque [Nm] | 0.63180 | ±0.00450 |
| Air temperature [K] | 0.29644 | ±0.01332 |
| Tool wear [min] | 0.28562 | ±0.03566 |
| Process temperature [K] | 0.25247 | ±0.02875 |
| Rotational speed [rpm] | 0.19393 | ±0.00608 |
| Type | 0.02856 | ±0.01064 |

### Operational Meaning & Limitations
Features like **Torque [Nm]**, **Rotational speed [rpm]**, and **Tool wear [min]** emerge as top predictive drivers. However, statistical sensitivity in an ML model does not prove physical causation. Mechanical wear, lubrication breakdown, and electrical load must be confirmed through telemetry and physical inspection.
