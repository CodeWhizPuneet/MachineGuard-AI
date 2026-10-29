"""
MachineGuard AI: Multi-Algorithm Comparison and Candidate Model Factory
Defines candidate algorithms, hyperparameter tuning spaces, cross-validation, and metrics evaluation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from src.config import RANDOM_SEED, CV_FOLDS
from src.preprocessing import create_preprocessor

def get_candidate_models(random_seed: int = RANDOM_SEED) -> Dict[str, Dict[str, Any]]:
    """
    Returns candidate models with their pipelines and hyperparameter search grids.
    Each model is encapsulated in a Pipeline with the preprocessor to prevent data leakage.
    """
    models = {}

    # 1. Baseline Dummy Classifier
    dummy_pipe = Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric=False)),
        ("classifier", DummyClassifier(strategy="stratified", random_state=random_seed))
    ])
    models["Dummy Classifier (Baseline)"] = {
        "pipeline": dummy_pipe,
        "param_dist": {},
        "scale_numeric": False,
        "description": "Stratified random guessing baseline reflecting dataset class prior."
    }

    # 2. Regularized Logistic Regression with class-weight handling
    lr_pipe = Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_seed))
    ])
    models["Logistic Regression"] = {
        "pipeline": lr_pipe,
        "param_dist": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver": ["lbfgs", "liblinear"]
        },
        "scale_numeric": True,
        "description": "Linear classifier with L2 regularization and class weight adjustments."
    }

    # 3. Random Forest with complexity limits to control overfitting
    rf_pipe = Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric=False)),
        ("classifier", RandomForestClassifier(random_state=random_seed, n_jobs=-1))
    ])
    models["Random Forest"] = {
        "pipeline": rf_pipe,
        "param_dist": {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [6, 10, 15, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__class_weight": [None, "balanced", "balanced_subsample"]
        },
        "scale_numeric": False,
        "description": "Ensemble of decision trees with bagging, depth limits, and class weighting."
    }

    # 4. Extra Trees Classifier
    et_pipe = Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric=False)),
        ("classifier", ExtraTreesClassifier(random_state=random_seed, n_jobs=-1))
    ])
    models["Extra Trees"] = {
        "pipeline": et_pipe,
        "param_dist": {
            "classifier__n_estimators": [100, 150],
            "classifier__max_depth": [8, 12, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__class_weight": [None, "balanced"]
        },
        "scale_numeric": False,
        "description": "Extremely Randomized Trees with random thresholds for variance reduction."
    }

    # 5. Gradient Boosting Classifier
    gb_pipe = Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric=True)),
        ("classifier", GradientBoostingClassifier(random_state=random_seed))
    ])
    models["Gradient Boosting"] = {
        "pipeline": gb_pipe,
        "param_dist": {
            "classifier__n_estimators": [100, 150],
            "classifier__learning_rate": [0.05, 0.1, 0.15],
            "classifier__max_depth": [3, 4, 5],
            "classifier__subsample": [0.8, 1.0]
        },
        "scale_numeric": True,
        "description": "Sequential boosting model optimizing deviance loss."
    }

    # 6. Histogram-based Gradient Boosting (fast & handles large feature spaces cleanly)
    hgb_pipe = Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric=False)),
        ("classifier", HistGradientBoostingClassifier(random_state=random_seed))
    ])
    models["Hist Gradient Boosting"] = {
        "pipeline": hgb_pipe,
        "param_dist": {
            "classifier__max_iter": [100, 150],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth": [4, 6, 8],
            "classifier__class_weight": [None, "balanced"]
        },
        "scale_numeric": False,
        "description": "Histogram-binned gradient boosting with native speed and complexity controls."
    }

    # 7. Check if XGBoost / LightGBM is available
    try:
        from xgboost import XGBClassifier
        xgb_pipe = Pipeline([
            ("preprocessor", create_preprocessor(scale_numeric=False)),
            ("classifier", XGBClassifier(
                random_state=random_seed,
                eval_metric="logloss",
                n_jobs=-1
            ))
        ])
        models["XGBoost"] = {
            "pipeline": xgb_pipe,
            "param_dist": {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__scale_pos_weight": [1, 5, 10]
            },
            "scale_numeric": False,
            "description": "Extreme Gradient Boosting with regularized objective and scale_pos_weight."
        }
    except ImportError:
        pass  # XGBoost optional; scikit-learn models provide full coverage

    return models

def evaluate_predictions(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate comprehensive binary classification metrics for imbalanced classes.
    """
    y_true_arr = np.array(y_true).astype(int)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_pred_proba))
    except Exception:
        roc_auc = 0.5

    try:
        avg_precision = float(average_precision_score(y_true_arr, y_pred_proba))
    except Exception:
        avg_precision = 0.0

    return {
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "average_precision": round(avg_precision, 4),
        "roc_auc": round(roc_auc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "rates": {
            "false_alarm_rate_fpr": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
            "miss_rate_fnr": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4)
        }
    }
