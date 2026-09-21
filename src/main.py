"""
Main Pipeline Orchestration module for All-India Real Estate Engine.
Executes the end-to-end data ingestion, cleaning, feature engineering, visual diagnostics,
modeling, explainability, index computation, and artifact serialization.
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
    Orchestrates the entire automated Indian real estate valuation pipeline:
    1. Ingestion & Validation (29,451 listings across Indian metros)
    2. Data Preprocessing & Address Extraction
    3. Indian Real Estate Feature Engineering
    4. EDA Diagnostic Charts (Price distributions, Metro volume, Correlation, Panels)
    5. Dataset Preparation & Train/Test Split
    6. Multi-Model Benchmarking (OLS, Random Forest, XGBoost), 5-Fold CV, Quantiles
    7. Model Explainability: Feature Importance & SHAP Values
    8. Valuation Index (City x BHK) & Power BI Exports
    9. Model Serialization & Summary Reporting
    """
    print("=" * 85)
    print(" ALL-INDIA REAL ESTATE VALUATION INDEX & PRICE PREDICTION ENGINE ")
    print("=" * 85)

    # Stage 1: Data Ingestion
    logger.info(">>> Stage 1: Loading All-India Housing dataset...")
    raw_df = load_raw_data(data_path=RAW_DATA_PATH, auto_download=True)
    logger.info(f"Loaded: {raw_df.shape[0]} properties across Indian urban centers.")

    # Stage 2: Preprocessing
    logger.info(">>> Stage 2: Preprocessing and locality parsing...")
    cleaned_df = clean_data(raw_df)
    logger.info(f"Cleaned dataset: {len(cleaned_df)} properties retained.")

    # Stage 3: Feature Engineering
    logger.info(">>> Stage 3: Engineering Indian real estate features...")
    engineered_df = engineer_features(cleaned_df)

    # Save cleaned_engineered_dataset.csv for Power BI
    save_master_cleaned_dataset(engineered_df)

    # Stage 4: Exploratory Data Analysis & Diagnostic Charts
    logger.info(">>> Stage 4: Generating visual diagnostic charts...")
    run_eda(engineered_df)

    # Stage 5: Dataset Preparation
    logger.info(">>> Stage 5: Building machine learning feature matrix & 80/20 split...")
    X_train, X_test, y_train, y_test, meta_train, meta_test, feature_names = build_model_dataset(engineered_df)

    # Stage 6: Model Training & Evaluation
    logger.info(">>> Stage 6: Benchmarking Linear Regression, Random Forest, and XGBoost...")
    best_model, q_lower, q_upper, metrics_df = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    # Stage 7: Explainability
    logger.info(">>> Stage 7: Computing feature importances and SHAP values...")
    explain_model(best_model, X_train, X_test, feature_names)

    # Stage 8: Valuation Index & Test Predictions
    logger.info(">>> Stage 8: Computing City & BHK Valuation Index and test predictions...")
    build_valuation_index(engineered_df)
    generate_test_predictions(best_model, q_lower, q_upper, X_test, y_test, meta_test)

    # Final Pipeline Summary
    print("\n" + "=" * 85)
    print(" PIPELINE EXECUTION SUMMARY & BENCHMARK RESULTS (INDIAN REAL ESTATE) ")
    print("=" * 85)

    print("\nMODEL PERFORMANCE BENCHMARK (5-Fold CV & Test Set):")
    header = f"{'Model':<20} | {'Train R2':<10} | {'Test R2':<10} | {'5-Fold CV R2':<18} | {'RMSE (Lakhs)':<14} | {'MAE (Lakhs)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for _, row in metrics_df.iterrows():
        cv_str = f"{row['CV_R2_Mean']:.4f} +/- {row['CV_R2_Std']:.4f}"
        print(
            f"{row['Model']:<20} | "
            f"{row['Train_R2']:<10.4f} | "
            f"{row['Test_R2']:<10.4f} | "
            f"{cv_str:<18} | "
            f"INR {row['RMSE']:<10.2f} | "
            f"INR {row['MAE']:<8.2f}"
        )
    print("-" * len(header))

    champion_row = metrics_df.sort_values(by="Test_R2", ascending=False).iloc[0]
    print(f"\nCHAMPION MODEL: {champion_row['Model']}")
    print(f" - Test R2 Score : {champion_row['Test_R2']:.4f}")
    print(f" - Test RMSE     : INR {champion_row['RMSE']:.2f} Lakhs")
    print(f" - Test MAE      : INR {champion_row['MAE']:.2f} Lakhs")

    print("\n" + "=" * 85)
    print(" GENERATED ARTIFACTS IN outputs/ ")
    print("=" * 85)
    all_files = sorted(OUTPUTS_DIR.rglob("*"))
    file_records = []
    for f in all_files:
        if f.is_file():
            rel_path = f.relative_to(OUTPUTS_DIR)
            size_kb = f.stat().st_size / 1024.0
            category = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
            file_records.append((str(rel_path), f"{size_kb:,.1f} KB", category))

    out_header = f"{'Output File Path':<50} | {'Size':<12} | {'Category':<15}"
    print(out_header)
    print("-" * len(out_header))
    for path_str, size_str, cat in file_records:
        print(f"{path_str:<50} | {size_str:<12} | {cat:<15}")
    print("-" * len(out_header))
    print(f"Total artifacts verified: {len(file_records)} files.\n")


if __name__ == "__main__":
    run_pipeline()
