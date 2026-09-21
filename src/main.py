"""
Main pipeline execution script.
Runs data loading, preprocessing, feature engineering, model training,
and evaluation for the Indian housing price dataset.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    OUTPUTS_DIR,
    CHARTS_DIR,
    POWERBI_DIR,
    MODELS_DIR,
)
from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.eda import run_eda
from src.dataset import build_model_dataset
from src.train import train_and_evaluate
from src.explain import explain_model
from src.valuation_index import (
    build_valuation_index,
    generate_test_predictions,
    save_master_cleaned_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """
    Executes the end-to-end pipeline:
    1. Loads the raw dataset
    2. Cleans data and parses locations
    3. Engineers features
    4. Generates EDA figures
    5. Prepares train/test split
    6. Trains and evaluates models (Linear Regression, Random Forest, XGBoost)
    7. Computes feature importance and SHAP values
    8. Exports valuation index and test predictions
    """
    logger.info("Starting housing price prediction pipeline...")

    # Data Ingestion
    logger.info("Loading dataset from %s...", RAW_DATA_PATH)
    raw_df = load_raw_data(data_path=RAW_DATA_PATH, auto_download=True)
    logger.info("Loaded %d raw records.", len(raw_df))

    # Preprocessing
    logger.info("Cleaning data and parsing locations...")
    cleaned_df = clean_data(raw_df)
    logger.info("Retained %d records after cleaning.", len(cleaned_df))

    # Feature Engineering
    logger.info("Engineering features...")
    engineered_df = engineer_features(cleaned_df)
    save_master_cleaned_dataset(engineered_df)

    # Exploratory Data Analysis
    logger.info("Generating EDA figures...")
    run_eda(engineered_df)

    # Dataset Preparation
    logger.info("Building feature matrix and 80/20 train/test split...")
    X_train, X_test, y_train, y_test, meta_train, meta_test, feature_names = build_model_dataset(engineered_df)

    # Model Training & Evaluation
    logger.info("Training and benchmarking models...")
    best_model, q_lower, q_upper, metrics_df = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    # Explainability
    logger.info("Computing feature importances and SHAP values...")
    explain_model(best_model, X_train, X_test, feature_names)

    # Valuation Index & Test Predictions
    logger.info("Computing city valuation index and generating test predictions...")
    build_valuation_index(engineered_df)
    generate_test_predictions(best_model, q_lower, q_upper, X_test, y_test, meta_test)

    # Summary Output
    print("\n--- Model Benchmark Results ---")
    print(metrics_df.to_string(index=False))

    best_row = metrics_df.sort_values(by="Test_R2", ascending=False).iloc[0]
    print(f"\nBest Model: {best_row['Model']}")
    print(f"  Test R2:   {best_row['Test_R2']:.4f}")
    print(f"  Test RMSE: ₹ {best_row['RMSE']:.2f} Lakhs")
    print(f"  Test MAE:  ₹ {best_row['MAE']:.2f} Lakhs\n")


if __name__ == "__main__":
    run_pipeline()
