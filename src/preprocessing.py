"""
Preprocessing module for All-India Real Estate Valuation and Price Prediction Engine.
Performs data cleaning, address parsing, city grouping, and outlier filtering.
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

from src.config import (
    TARGET,
    RAW_TARGET,
    MIN_SQUARE_FT,
    MAX_SQUARE_FT,
    MIN_PRICE_LAKHS,
    MAX_PRICE_LAKHS,
    MAX_BHK,
    TOP_CITIES,
)
from src.data_loader import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Indian housing dataset:
      - Standardizes column headers.
      - Extracts City and Locality from ADDRESS.
      - Groups geographic regions into top metros and 'Other_City'.
      - Filters extreme data-entry outliers for realistic estimation.
      - Assigns sequential Property_Id.
      - Computes rate per square foot in ₹/sqft.

    Parameters:
        df: Raw DataFrame containing 29,451 rows.

    Returns:
        pd.DataFrame: Cleaned DataFrame with zero missing values and realistic bounds.
    """
    df = df.copy()

    # Standardize column names
    col_rename = {
        "BHK_NO.": "BHK_NO",
        RAW_TARGET: TARGET,
    }
    df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

    # Address parsing: format is usually "Locality, City"
    def parse_city(addr: str) -> str:
        parts = [p.strip() for p in str(addr).split(",") if p.strip()]
        if not parts:
            return "Unknown"
        city = parts[-1]
        # City name normalizations
        if city.lower() in ["bengaluru", "bangalore"]:
            return "Bangalore"
        if city.lower() in ["navi mumbai", "mumbai"]:
            return "Mumbai"
        if city.lower() in ["new delhi", "delhi"]:
            return "Delhi"
        return city

    def parse_locality(addr: str) -> str:
        parts = [p.strip() for p in str(addr).split(",") if p.strip()]
        return parts[0] if parts else "Unknown"

    df["City"] = df["ADDRESS"].apply(parse_city)
    df["Locality"] = df["ADDRESS"].apply(parse_locality)

    # City categorical grouping (top 20-25 cities vs Other_City)
    df["City_Grouped"] = df["City"].apply(lambda c: c if c in TOP_CITIES else "Other_City")

    # Add unique identifier
    df["Property_Id"] = np.arange(1, len(df) + 1)

    initial_count = len(df)

    # Outlier filtering for realistic residential modeling
    valid_mask = (
        (df["SQUARE_FT"] >= MIN_SQUARE_FT)
        & (df["SQUARE_FT"] <= MAX_SQUARE_FT)
        & (df[TARGET] >= MIN_PRICE_LAKHS)
        & (df[TARGET] <= MAX_PRICE_LAKHS)
        & (df["BHK_NO"] >= 1)
        & (df["BHK_NO"] <= MAX_BHK)
    )
    df_clean = df[valid_mask].copy().reset_index(drop=True)
    filtered_count = initial_count - len(df_clean)
    logger.info(
        f"Filtered {filtered_count} unrealistic outliers ({initial_count} -> {len(df_clean)} properties retained)."
    )

    # Compute rate per square foot in INR
    df_clean["Price_Per_SqFt"] = (df_clean[TARGET] * 100000.0) / df_clean["SQUARE_FT"]

    # Impute or verify missing values
    missing_sum = df_clean.isnull().sum().sum()
    if missing_sum > 0:
        logger.warning(f"Found {missing_sum} missing values in dataset. Imputing...")
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ["float64", "int64"]:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    logger.info(f"Preprocessing completed successfully. Clean dataset shape: {df_clean.shape}")
    return df_clean


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    print("Preprocessing Test Passed.")
    print("Cleaned shape:", cleaned.shape)
    print("Sample:\n", cleaned[["Property_Id", "City", "Locality", "BHK_NO", "SQUARE_FT", "Price_Lakhs", "Price_Per_SqFt"]].head())
