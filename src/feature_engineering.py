"""
Feature Engineering module for All-India Real Estate Valuation and Price Prediction Engine.
Constructs real estate metrics, non-linear representations, and interaction indicators.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import TARGET
from src.data_loader import load_raw_data
from src.preprocessing import clean_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs domain-specific real estate features:
      - Log_Sqft: Logarithmic transformation of property area.
      - BHK_Sqft_Ratio: Average square footage per bedroom room.
      - Is_RK: Binary flag indicating Studio Room-Kitchen unit.
      - Is_Dealer_Or_Builder: Professional broker or developer listing flag.
      - Is_UC_RERA: Under construction with certified RERA compliance.
      - Is_Resale_Ready: Ready-to-move secondary resale property.
      - Is_Luxury: Flag for properties priced at or above ₹ 1.5 Crore (for analytics only).

    Strict Target Leakage Rule:
      - 'Price_Per_SqFt' and 'Is_Luxury' are computed for downstream indexing, EDA,
        and Power BI tables, but are strictly excluded from predictive model features.

    Parameters:
        df: Preprocessed DataFrame.

    Returns:
        pd.DataFrame: Augmented DataFrame with engineered attributes.
    """
    df = df.copy()

    # 1. Logarithmic square footage
    df["Log_Sqft"] = np.log1p(df["SQUARE_FT"])

    # 2. Space intensity: Sqft per BHK room
    df["BHK_Sqft_Ratio"] = df["SQUARE_FT"] / np.maximum(df["BHK_NO"], 1)

    # 3. Unit configuration flag: RK (Room Kitchen studio) vs standard BHK
    df["Is_RK"] = (df["BHK_OR_RK"].astype(str).str.upper() == "RK").astype(int)

    # 4. Listing source indicator: Broker/Dealer vs direct Owner
    df["Is_Dealer_Or_Builder"] = df["POSTED_BY"].isin(["Dealer", "Builder"]).astype(int)

    # 5. Composite trust & stage flags
    df["Is_UC_RERA"] = ((df["UNDER_CONSTRUCTION"] == 1) & (df["RERA"] == 1)).astype(int)
    df["Is_Resale_Ready"] = ((df["RESALE"] == 1) & (df["READY_TO_MOVE"] == 1)).astype(int)

    # 6. Analytics-only luxury classification (₹ 150 Lakhs = ₹ 1.5 Crore)
    df["Is_Luxury"] = (df[TARGET] >= 150.0).astype(int)

    logger.info(f"Feature engineering completed. Resulting dataset shape: {df.shape}")
    return df


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    feat = engineer_features(cleaned)
    print("Feature Engineering Test Passed.")
    print("Engineered columns sample:", feat.columns.tolist()[-7:])
    print("Sample rows:\n", feat[["City", "BHK_NO", "SQUARE_FT", "Log_Sqft", "BHK_Sqft_Ratio", "Is_UC_RERA", "Price_Lakhs"]].head())
