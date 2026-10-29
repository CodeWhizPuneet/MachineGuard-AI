"""
MachineGuard AI: Preprocessing Pipeline Construction
Constructs scikit-learn ColumnTransformer pipelines with strict leakage prevention.
"""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def create_preprocessor(
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing:
    - OneHotEncoder for 'Type' (handles 'L', 'M', 'H')
    - StandardScaler for numerical operating variables (optional or default True)
    Fit strictly on training data / training folds.
    """
    transformers = [
        (
            "cat",
            OneHotEncoder(categories=[["L", "M", "H"]], drop=None, handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES
        )
    ]

    if scale_numeric:
        transformers.append(
            ("num", StandardScaler(), NUMERICAL_FEATURES)
        )
    else:
        transformers.append(
            ("num", "passthrough", NUMERICAL_FEATURES)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor

def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract human-readable feature names after one-hot encoding and scaling."""
    names = []
    # Categorical features
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        names.extend(list(cat_names))
    except Exception:
        names.extend(["Type_L", "Type_M", "Type_H"])

    # Numerical features
    names.extend(NUMERICAL_FEATURES)
    return names
