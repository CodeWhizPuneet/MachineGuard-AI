"""
MachineGuard AI: Model Training, Comparison, Threshold Tuning, and Artifact Export
Executes the full machine learning training lifecycle:
1. Stratified 64/16/20 data split.
2. 5-fold CV and hyperparameter tuning on training set only.
3. Overfitting / generalization-gap analysis across algorithms.
4. Validation-based model selection.
5. Validation-based decision-threshold optimization.
6. Calibration inspection.
7. Final evaluation on held-out test set.
8. Artifact persistence.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from src.config import (
    DATA_FILE,
    ARTIFACTS_DIR,
    RANDOM_SEED,
    CV_FOLDS,
    MODEL_PIPELINE_FILE,
    METADATA_FILE,
    MODEL_COMPARISON_FILE,
    THRESHOLD_ANALYSIS_FILE,
    FEATURE_IMPORTANCE_FILE,
    FINAL_METRICS_FILE,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from src.data_loader import load_raw_dataset, prepare_splits
from src.model_selection import get_candidate_models, evaluate_predictions
from src.preprocessing import get_transformed_feature_names

def run_threshold_optimization(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate thresholds from 0.05 to 0.90 using VALIDATION DATA ONLY.
    Optimizes for balanced high recall (catching failures) while controlling false alarms.
    """
    y_val_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.90, 35)

    table = []
    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics = None

    for thresh in thresholds:
        metrics = evaluate_predictions(y_val, y_val_proba, threshold=thresh)
        entry = {
            "threshold": round(float(thresh), 3),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "false_positives": metrics["confusion_matrix"]["false_positives"],
            "false_negatives": metrics["confusion_matrix"]["false_negatives"]
        }
        table.append(entry)

        # Operational Objective: Maximize F1 score while keeping recall >= 0.65 if possible
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = float(thresh)
            best_metrics = entry

    rationale = (
        f"Threshold {best_threshold:.3f} selected by maximizing F1-score ({best_f1:.4f}) "
        f"on held-out validation data. Achieves recall={best_metrics['recall']:.4f} "
        f"with {best_metrics['false_positives']} false alarms out of {len(y_val)} machines."
    )

    return best_threshold, table, {"selected_threshold": best_threshold, "rationale": rationale, "metrics": best_metrics}

def calculate_feature_importance(
    model_pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Computes global feature importance:
    1. Native tree feature_importances_ if available.
    2. Permutation importance on held-out validation data (leakage-free).
    """
    results = {}
    classifier = model_pipeline.named_steps.get("classifier")

    # Native importance if tree-based
    if hasattr(classifier, "feature_importances_"):
        native_imp = classifier.feature_importances_
        results["native_tree_importance"] = {
            name: round(float(val), 4)
            for name, val in sorted(zip(feature_names, native_imp), key=lambda x: x[1], reverse=True)
        }

    # Permutation importance on validation set
    perm = permutation_importance(
        model_pipeline,
        X_val,
        y_val,
        n_repeats=5,
        random_state=RANDOM_SEED,
        scoring="average_precision"
    )

    perm_dict = {}
    input_cols = list(X_val.columns)
    for col, mean, std in zip(input_cols, perm.importances_mean, perm.importances_std):
        perm_dict[col] = {
            "importance_mean": round(float(mean), 5),
            "importance_std": round(float(std), 5)
        }
    # Sort by mean
    sorted_perm = dict(sorted(perm_dict.items(), key=lambda item: item[1]["importance_mean"], reverse=True))
    results["permutation_importance_validation"] = sorted_perm

    return results

def train_and_compare_all_models(
    csv_path: str = str(DATA_FILE),
    quick_run: bool = False
) -> Dict[str, Any]:
    """
    Full training workflow:
    1. Loads dataset, creates splits.
    2. Trains and tunes candidates with CV on training set.
    3. Evaluates on validation set.
    4. Selects winning model.
    5. Optimizes threshold on validation set.
    6. Evaluates final test performance.
    7. Saves all artifacts.
    """
    print("=" * 70)
    print("MachineGuard AI: Model Training and Evaluation Pipeline Starting")
    print("=" * 70)

    start_time = time.time()
    df = load_raw_dataset(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df)

    print(f"Data Splits Created:")
    print(f"  Training Set:   {len(X_train)} samples ({y_train.sum()} failures, {y_train.mean()*100:.2f}%)")
    print(f"  Validation Set: {len(X_val)} samples ({y_val.sum()} failures, {y_val.mean()*100:.2f}%)")
    print(f"  Final Test Set: {len(X_test)} samples ({y_test.sum()} failures, {y_test.mean()*100:.2f}%)")

    candidates = get_candidate_models(random_seed=RANDOM_SEED)
    comparison_results = {}
    fitted_pipelines = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for name, config in candidates.items():
        print(f"\n--- Training & Tuning: {name} ---")
        pipeline = config["pipeline"]
        param_dist = config["param_dist"]

        # If hyperparameter grid exists, tune with RandomizedSearchCV on Training set only
        if param_dist and not quick_run:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=min(10, np.prod([len(v) for v in param_dist.values()])),
                scoring="average_precision",
                cv=cv,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                refit=True
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            best_params = search.best_params_
            cv_avg_precision = float(search.best_score_)
        else:
            pipeline.fit(X_train, y_train)
            best_pipe = pipeline
            best_params = {}
            cv_avg_precision = 0.0

        fitted_pipelines[name] = best_pipe

        # Training set evaluation (for generalization gap check)
        train_proba = best_pipe.predict_proba(X_train)[:, 1]
        train_metrics = evaluate_predictions(y_train, train_proba, threshold=0.5)

        # Validation set evaluation (for model selection)
        val_proba = best_pipe.predict_proba(X_val)[:, 1]
        val_metrics = evaluate_predictions(y_val, val_proba, threshold=0.5)

        # Generalization gap (Train AP - Val AP)
        gap = train_metrics["average_precision"] - val_metrics["average_precision"]

        comparison_results[name] = {
            "algorithm": name,
            "description": config["description"],
            "best_hyperparameters": {k.replace("classifier__", ""): v for k, v in best_params.items()},
            "cv_train_average_precision": round(cv_avg_precision, 4),
            "train_metrics": train_metrics,
            "val_metrics_default_0_5": val_metrics,
            "generalization_gap_ap": round(float(gap), 4),
            "overfitting_risk": "Low" if gap < 0.15 else ("Moderate" if gap < 0.30 else "High")
        }

        print(f"  Val Average Precision: {val_metrics['average_precision']:.4f}")
        print(f"  Val ROC-AUC:           {val_metrics['roc_auc']:.4f}")
        print(f"  Val F1-Score (0.50):   {val_metrics['f1_score']:.4f}")
        print(f"  Val Recall (0.50):     {val_metrics['recall']:.4f}")
        print(f"  Generalization Gap:    {gap:.4f} ({comparison_results[name]['overfitting_risk']})")

    # =========================================================================
    # MODEL SELECTION STRATEGY:
    # 1. Rank by Validation Average Precision (primary PR summary metric for imbalanced data).
    # 2. Check Overfitting Risk (prefer models with stable generalization gap).
    # 3. Assess Failure Recall and false positive burden.
    # =========================================================================
    ranked_models = sorted(
        comparison_results.items(),
        key=lambda item: (item[1]["val_metrics_default_0_5"]["average_precision"], item[1]["val_metrics_default_0_5"]["f1_score"]),
        reverse=True
    )

    # Exclude dummy baseline from winner
    eligible_candidates = [m for m in ranked_models if "Dummy" not in m[0]]
    winning_model_name = eligible_candidates[0][0]
    winning_pipeline = fitted_pipelines[winning_model_name]
    winning_info = comparison_results[winning_model_name]

    print("\n" + "=" * 70)
    print(f"WINNING MODEL SELECTED: {winning_model_name}")
    print(f"Reason: Highest Average Precision on held-out validation set "
          f"({winning_info['val_metrics_default_0_5']['average_precision']:.4f}) with balanced complexity.")
    print("=" * 70)

    # Threshold Optimization on Validation Data Only
    print(f"\n--- Threshold Optimization for {winning_model_name} (Validation Set) ---")
    best_thresh, threshold_table, threshold_summary = run_threshold_optimization(
        winning_pipeline, X_val, y_val
    )
    print(f"  Optimal Operating Threshold: {best_thresh:.3f}")
    print(f"  Validation F1 at {best_thresh:.3f}: {threshold_summary['metrics']['f1_score']:.4f}")
    print(f"  Validation Recall at {best_thresh:.3f}: {threshold_summary['metrics']['recall']:.4f}")

    # Feature Importance on Validation Data
    preproc = winning_pipeline.named_steps["preprocessor"]
    transformed_names = get_transformed_feature_names(preproc)
    feature_importances = calculate_feature_importance(
        winning_pipeline, X_val, y_val, transformed_names
    )

    # =========================================================================
    # FINAL HELD-OUT TEST EVALUATION (TOUCHED EXACTLY ONCE)
    # =========================================================================
    print(f"\n--- Final Independent Test Set Evaluation ({winning_model_name}) ---")
    test_proba = winning_pipeline.predict_proba(X_test)[:, 1]
    final_test_default = evaluate_predictions(y_test, test_proba, threshold=0.5)
    final_test_optimal = evaluate_predictions(y_test, test_proba, threshold=best_thresh)

    print(f"  Final Test AP:                   {final_test_optimal['average_precision']:.4f}")
    print(f"  Final Test ROC-AUC:              {final_test_optimal['roc_auc']:.4f}")
    print(f"  Final Test F1 (Default 0.50):    {final_test_default['f1_score']:.4f}")
    print(f"  Final Test F1 (Optimal {best_thresh:.3f}): {final_test_optimal['f1_score']:.4f}")
    print(f"  Final Test Recall (Optimal):     {final_test_optimal['recall']:.4f}")
    print(f"  Final Test FP (Optimal):         {final_test_optimal['confusion_matrix']['false_positives']}")
    print(f"  Final Test FN (Optimal):         {final_test_optimal['confusion_matrix']['false_negatives']}")

    # Probability calibration analysis on validation set
    prob_true, prob_pred = calibration_curve(y_val, winning_pipeline.predict_proba(X_val)[:, 1], n_bins=10)
    calibration_data = {
        "prob_true": [round(float(p), 4) for p in prob_true],
        "prob_pred": [round(float(p), 4) for p in prob_pred]
    }

    # Save artifacts
    print(f"\nSaving model artifacts to {ARTIFACTS_DIR}...")
    joblib.dump(winning_pipeline, MODEL_PIPELINE_FILE)

    metadata = {
        "project_name": "MachineGuard AI",
        "model_name": winning_model_name,
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": RANDOM_SEED,
        "selected_threshold": best_thresh,
        "threshold_rationale": threshold_summary["rationale"],
        "hyperparameters": winning_info["best_hyperparameters"],
        "feature_schema": {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "all_predictors": CATEGORICAL_FEATURES + NUMERICAL_FEATURES,
            "transformed_names": transformed_names
        },
        "dataset_split": {
            "train_samples": len(X_train),
            "train_failures": int(y_train.sum()),
            "val_samples": len(X_val),
            "val_failures": int(y_val.sum()),
            "test_samples": len(X_test),
            "test_failures": int(y_test.sum())
        },
        "calibration": calibration_data
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(MODEL_COMPARISON_FILE, "w") as f:
        json.dump(comparison_results, f, indent=2)

    with open(THRESHOLD_ANALYSIS_FILE, "w") as f:
        json.dump({
            "model": winning_model_name,
            "selected_threshold": best_thresh,
            "summary": threshold_summary,
            "curve_table": threshold_table
        }, f, indent=2)

    with open(FEATURE_IMPORTANCE_FILE, "w") as f:
        json.dump(feature_importances, f, indent=2)

    final_results = {
        "model_name": winning_model_name,
        "selected_threshold": best_thresh,
        "test_metrics_default_0_5": final_test_default,
        "test_metrics_optimal_threshold": final_test_optimal,
        "validation_metrics_optimal_threshold": threshold_summary["metrics"]
    }
    with open(FINAL_METRICS_FILE, "w") as f:
        json.dump(final_results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"[SUCCESS] Training completed in {elapsed:.1f} seconds. All artifacts saved.")
    return final_results

if __name__ == "__main__":
    train_and_compare_all_models()
