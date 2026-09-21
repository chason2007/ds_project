"""
Dataset module for All-India Real Estate Engine.
Handles categorical encoding, metadata preservation, and train/test splitting.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, List

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    META_COLS,
    CATEGORICAL_COLS,
    BINARY_COLS,
    NUMERICAL_COLS,
)
from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_model_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Transforms the engineered dataset into machine learning train/test matrices:
      - Isolates metadata columns for reporting and Power BI slicing.
      - One-hot encodes nominal categoricals with drop_first=True.
      - Enforces strict target leakage prevention: excludes target, price/sqft, and luxury flags.
      - Sanitizes feature column names for XGBoost and Scikit-Learn compatibility.
      - Performs deterministic 80/20 train/test split.

    Returns:
        X_train, X_test, y_train, y_test, meta_train, meta_test, feature_names
    """
    logger.info("Building machine learning dataset and feature matrices...")

    # Preserve metadata slice
    meta_present = [c for c in META_COLS if c in df.columns]
    meta_df = df[meta_present].copy()

    # Define candidate model features
    candidate_features = [
        "SQUARE_FT",
        "BHK_NO",
        "LATITUDE",
        "LONGITUDE",
        "Log_Sqft",
        "BHK_Sqft_Ratio",
        "UNDER_CONSTRUCTION",
        "RERA",
        "READY_TO_MOVE",
        "RESALE",
        "Is_RK",
        "Is_Dealer_Or_Builder",
        "Is_UC_RERA",
        "Is_Resale_Ready",
        "POSTED_BY",
        "BHK_OR_RK",
        "City_Grouped",
    ]
    model_df = df[[c for c in candidate_features if c in df.columns]].copy()

    # One-hot encode categorical features
    cat_cols_to_encode = [c for c in ["POSTED_BY", "BHK_OR_RK", "City_Grouped"] if c in model_df.columns]
    X_encoded = pd.get_dummies(model_df, columns=cat_cols_to_encode, drop_first=True)

    # Sanitize feature column names for XGBoost (remove [, ], <, >, etc.)
    clean_cols = [
        col.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_").replace(" ", "_")
        for col in X_encoded.columns
    ]
    X_encoded.columns = clean_cols
    feature_names = list(X_encoded.columns)

    y = df[TARGET].copy()

    # Deterministic train/test split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X_encoded,
        y,
        meta_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Convert all feature data types to float32 for model consistency
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    logger.info(
        f"Dataset split complete: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows, "
        f"Features={len(feature_names)}."
    )
    return X_train, X_test, y_train, y_test, meta_train, meta_test, feature_names


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    feat = engineer_features(cleaned)
    X_train, X_test, y_train, y_test, m_train, m_test, f_names = build_model_dataset(feat)
    print("Dataset Build Test Passed.")
    print("Feature count:", len(f_names))
    print("Sample features:", f_names[:10])
