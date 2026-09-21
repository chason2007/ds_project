"""
Valuation Index and Test Predictions module for All-India Real Estate Engine.
Computes City & BHK valuation indices and exports out-of-sample prediction bounds.
"""

import sys
import logging
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import POWERBI_DIR, CHARTS_DIR, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_valuation_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs an Indian Real Estate Valuation Index:
      - Groups by City and BHK configuration (1 to 5 BHK).
      - Computes Mean Price (₹ Lakhs), Median Price, Avg ₹/SqFt, and Volume.
      - Normalizes a Benchmark Index (Base 100.0 = National Median ₹/SqFt).
      - Exports outputs/powerbi_data/valuation_index.csv.
      - Generates comparative chart outputs/charts/valuation_index_top6.png for top 6 metros.

    Returns:
        pd.DataFrame: Relational valuation index table.
    """
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Computing Indian Real Estate Valuation Index...")

    # Filter for standard BHKs and cities with valid volume
    valid_df = df[df["BHK_NO"].between(1, 5)].copy()

    # National median rate per square foot as benchmark baseline
    national_median_rate = float(valid_df["Price_Per_SqFt"].median())
    if national_median_rate <= 0:
        national_median_rate = 5000.0

    idx_df = (
        valid_df.groupby(["City", "BHK_NO"])
        .agg(
            AvgPrice_Lakhs=(TARGET, "mean"),
            MedianPrice_Lakhs=(TARGET, "median"),
            AvgPricePerSqFt=("Price_Per_SqFt", "mean"),
            MedianPricePerSqFt=("Price_Per_SqFt", "median"),
            PropertyCount=(TARGET, "count"),
        )
        .reset_index()
    )

    # Filter out combinations with very small sample sizes (less than 5 properties)
    idx_df = idx_df[idx_df["PropertyCount"] >= 5].copy()

    # Valuation Index: 100.0 = National Median Rate
    idx_df["ValuationIndex"] = (
        (idx_df["AvgPricePerSqFt"] / national_median_rate) * 100.0
    ).round(2)

    idx_df["AvgPrice_Lakhs"] = idx_df["AvgPrice_Lakhs"].round(2)
    idx_df["MedianPrice_Lakhs"] = idx_df["MedianPrice_Lakhs"].round(2)
    idx_df["AvgPricePerSqFt"] = idx_df["AvgPricePerSqFt"].round(2)
    idx_df["MedianPricePerSqFt"] = idx_df["MedianPricePerSqFt"].round(2)

    # Sort logically
    idx_df = idx_df.sort_values(by=["City", "BHK_NO"]).reset_index(drop=True)

    index_path = POWERBI_DIR / "valuation_index.csv"
    idx_df.to_csv(index_path, index=False)
    logger.info(f"Saved valuation index table ({len(idx_df)} rows) to {index_path}")

    # Generate Top 6 Metros Multi-Series Chart
    top6_cities = ["Mumbai", "Bangalore", "Pune", "Noida", "Kolkata", "Chennai"]
    chart_data = idx_df[idx_df["City"].isin(top6_cities)].copy()

    if not chart_data.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        palette = sns.color_palette("tab10", n_colors=len(top6_cities))

        for city, color in zip(top6_cities, palette):
            sub = chart_data[chart_data["City"] == city].sort_values("BHK_NO")
            if not sub.empty:
                ax.plot(
                    sub["BHK_NO"],
                    sub["ValuationIndex"],
                    marker="o",
                    linewidth=2.2,
                    markersize=6,
                    label=city,
                    color=color,
                )

        ax.axhline(100.0, color="gray", linestyle="--", alpha=0.7, label="National Benchmark (100.0)")
        ax.set_title("Real Estate Valuation Index by BHK across Top 6 Indian Metros", fontsize=13, fontweight="bold")
        ax.set_xlabel("Property Configuration (BHK)", fontsize=11)
        ax.set_ylabel("Valuation Index (Base 100 = National Median ₹/SqFt)", fontsize=11)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(title="Metro City", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        chart_path = CHARTS_DIR / "valuation_index_top6.png"
        plt.savefig(chart_path, dpi=200)
        plt.close()
        logger.info(f"Saved: {chart_path}")

    return idx_df


def generate_test_predictions(
    model: Any,
    q_lower: Any,
    q_upper: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes predictions, 80% interval bounds, and residuals for test properties:
      - ActualPrice, PredictedPrice, LowerBound_80, UpperBound_80 (all in ₹ Lakhs).
      - AbsoluteError, PercentageError, WithinInterval_80.
      - Exports outputs/powerbi_data/predictions.csv.

    Returns:
        pd.DataFrame: Enriched test predictions table.
    """
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating test predictions and 80% confidence interval bounds...")

    preds = np.maximum(0.0, model.predict(X_test))
    low_preds = np.maximum(0.0, q_lower.predict(X_test))
    high_preds = np.maximum(low_preds, q_upper.predict(X_test))

    res_df = meta_test.copy().reset_index(drop=True)
    res_df["ActualPrice"] = np.round(y_test.values, 2)
    res_df["PredictedPrice"] = np.round(preds, 2)
    res_df["LowerBound_80"] = np.round(low_preds, 2)
    res_df["UpperBound_80"] = np.round(high_preds, 2)

    res_df["AbsoluteError"] = np.round(np.abs(res_df["ActualPrice"] - res_df["PredictedPrice"]), 2)
    res_df["PercentageError"] = np.round((res_df["AbsoluteError"] / res_df["ActualPrice"]) * 100.0, 2)
    res_df["WithinInterval_80"] = (
        (res_df["ActualPrice"] >= res_df["LowerBound_80"])
        & (res_df["ActualPrice"] <= res_df["UpperBound_80"])
    ).astype(int)

    # Reorder columns
    cols_order = [
        "Property_Id",
        "City",
        "Locality",
        "BHK_NO",
        "BHK_OR_RK",
        "SQUARE_FT",
        "POSTED_BY",
        "RERA",
        "READY_TO_MOVE",
        "RESALE",
        "ActualPrice",
        "PredictedPrice",
        "LowerBound_80",
        "UpperBound_80",
        "AbsoluteError",
        "PercentageError",
        "WithinInterval_80",
    ]
    final_cols = [c for c in cols_order if c in res_df.columns] + [
        c for c in res_df.columns if c not in cols_order
    ]
    res_df = res_df[final_cols]

    preds_path = POWERBI_DIR / "predictions.csv"
    res_df.to_csv(preds_path, index=False)
    logger.info(f"Saved predictions table ({len(res_df)} rows) to {preds_path}")

    return res_df


def save_master_cleaned_dataset(df: pd.DataFrame) -> None:
    """
    Saves the master cleaned and engineered dataset for Power BI exploration.
    """
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = POWERBI_DIR / "cleaned_engineered_dataset.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved master property inventory ({len(df)} rows) to {out_path}")


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data
    from src.feature_engineering import engineer_features
    from src.dataset import build_model_dataset
    from src.config import MODELS_DIR
    import joblib

    raw = load_raw_data()
    cleaned = clean_data(raw)
    feat = engineer_features(cleaned)

    # 1. Valuation index
    idx = build_valuation_index(feat)
    print("Valuation Index rows:", len(idx))

    # 2. Master dataset
    save_master_cleaned_dataset(feat)

    # 3. Test predictions
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    q_low = joblib.load(MODELS_DIR / "quantile_lower.joblib")
    q_up = joblib.load(MODELS_DIR / "quantile_upper.joblib")

    X_tr, X_te, y_tr, y_te, m_tr, m_te, _ = build_model_dataset(feat)
    preds = generate_test_predictions(model, q_low, q_up, X_te, y_te, m_te)
    print("Predictions sample:\n", preds[["City", "BHK_NO", "SQUARE_FT", "ActualPrice", "PredictedPrice", "LowerBound_80", "UpperBound_80"]].head())

