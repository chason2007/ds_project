"""
Data Loader module for All-India Real Estate Valuation and Price Prediction Engine.
Responsible for downloading (if missing), loading, and validating the Indian housing dataset.
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    DATA_DESCRIPTION_PATH,
    URL_TRAIN,
    RAW_TARGET,
    TARGET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, target_path: Path) -> None:
    """
    Downloads a remote file if not already present locally.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from {url} to {target_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_f:
        out_f.write(resp.read())
    logger.info(f"Successfully downloaded {target_path.name} ({target_path.stat().st_size} bytes).")


def load_raw_data(
    data_path: Path = RAW_DATA_PATH,
    auto_download: bool = True,
) -> pd.DataFrame:
    """
    Loads the raw All-India House Price train dataset from local CSV, downloading it if missing.

    Parameters:
        data_path: Local path to train.csv.
        auto_download: Whether to automatically download missing dataset files.

    Returns:
        pd.DataFrame: Raw dataset containing 29,451 property records.
    """
    needs_download = False
    if not data_path.exists():
        needs_download = True
    else:
        # Check if the existing file is the Indian dataset
        try:
            sample_df = pd.read_csv(data_path, nrows=5)
            if RAW_TARGET not in sample_df.columns and TARGET not in sample_df.columns:
                logger.warning(f"File at {data_path} is not the Indian housing dataset. Re-downloading...")
                needs_download = True
        except Exception:
            needs_download = True

    if needs_download:
        if auto_download:
            logger.info("Downloading All-India Housing dataset...")
            download_file(URL_TRAIN, data_path)
        else:
            raise FileNotFoundError(f"Raw data file not found at {data_path} and auto_download is disabled.")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded raw Indian housing dataset from {data_path} with shape: {df.shape}")

    # Validate target presence
    if RAW_TARGET not in df.columns and TARGET not in df.columns:
        raise ValueError(f"Neither '{RAW_TARGET}' nor '{TARGET}' found in dataset columns.")

    expected_min_rows = 20000
    if len(df) < expected_min_rows:
        logger.warning(f"Expected at least {expected_min_rows} rows in dataset, but found {len(df)}.")

    return df


if __name__ == "__main__":
    raw_df = load_raw_data()
    print("Indian Data Loader Test Passed.")
    print("Columns:", list(raw_df.columns))
    print("Row count:", len(raw_df))
