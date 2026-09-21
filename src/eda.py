"""
Exploratory Data Analysis (EDA) module for Indian Real Estate dataset.
Generates diagnostic charts in outputs/charts/.
"""

import sys
import logging
from pathlib import Path

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

from src.config import CHARTS_DIR, TARGET
from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set consistent chart styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def run_eda(df: pd.DataFrame) -> None:
    """
    Produces 4 diagnostic visualizations and saves them to outputs/charts/:
      1. target_distribution.png: Distribution of prices (raw Lakhs and log scale).
      2. city_price_summary.png: Median price & listing volume across top Indian metros.
      3. correlation_heatmap.png: Correlation matrix of housing attributes.
      4. market_overview_panels.png: 4-panel real estate market dynamics.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating EDA diagnostic visualizations...")

    # 1. Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Clip extreme display tail for clean visualization
    display_prices = df[df[TARGET] <= 500][TARGET]
    sns.histplot(display_prices, kde=True, ax=axes[0], color="#1f77b4", bins=40)
    axes[0].set_title("Property Price Distribution (<= ₹ 500 Lakhs)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Price (₹ in Lakhs)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)

    log_prices = np.log1p(df[TARGET])
    sns.histplot(log_prices, kde=True, ax=axes[1], color="#2ca02c", bins=40)
    axes[1].set_title("Log-Transformed Price Distribution [log(1 + Price)]", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Log(Price in Lakhs)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)

    plt.tight_layout()
    target_dist_path = CHARTS_DIR / "target_distribution.png"
    plt.savefig(target_dist_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {target_dist_path}")

    # 2. City Price Summary & Volume
    top_cities_df = (
        df.groupby("City_Grouped")
        .agg(Median_Price=(TARGET, "median"), Volume=(TARGET, "count"))
        .query("City_Grouped != 'Other_City'")
        .sort_values(by="Median_Price", ascending=True)
        .tail(15)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    top_cities_df["Median_Price"].plot(kind="barh", ax=axes[0], color="#ff7f0e", edgecolor="none")
    axes[0].set_title("Median Property Price by Top Indian Cities", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Median Price (₹ in Lakhs)", fontsize=11)
    axes[0].set_ylabel("City", fontsize=11)

    top_cities_df.sort_values(by="Volume", ascending=True)["Volume"].plot(
        kind="barh", ax=axes[1], color="#9467bd", edgecolor="none"
    )
    axes[1].set_title("Market Transaction Volume by City", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of Property Listings", fontsize=11)
    axes[1].set_ylabel("")

    plt.tight_layout()
    city_path = CHARTS_DIR / "city_price_summary.png"
    plt.savefig(city_path, dpi=200)
    # Also save as missingness_summary.png for backwards catalog compatibility
    plt.savefig(CHARTS_DIR / "missingness_summary.png", dpi=200)
    plt.close()
    logger.info(f"Saved: {city_path}")

    # 3. Correlation Heatmap
    corr_cols = [
        TARGET,
        "SQUARE_FT",
        "BHK_NO",
        "Price_Per_SqFt",
        "RERA",
        "UNDER_CONSTRUCTION",
        "READY_TO_MOVE",
        "RESALE",
        "Log_Sqft",
        "BHK_Sqft_Ratio",
    ]
    avail_corr_cols = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[avail_corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap (Indian Real Estate)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_path = CHARTS_DIR / "correlation_heatmap.png"
    plt.savefig(corr_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {corr_path}")

    # 4. Market Overview 4-Panel Grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel A: Price by BHK (1 to 5 BHK)
    bhk_subset = df[df["BHK_NO"].between(1, 5)]
    sns.boxplot(
        x="BHK_NO",
        y=TARGET,
        data=bhk_subset[bhk_subset[TARGET] <= 400],
        ax=axes[0, 0],
        hue="BHK_NO",
        palette="Blues_r",
        legend=False,
    )
    axes[0, 0].set_title("Price Distribution by BHK Configuration", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Number of Bedrooms (BHK)", fontsize=10)
    axes[0, 0].set_ylabel("Price (₹ in Lakhs)", fontsize=10)

    # Panel B: RERA Premium
    rera_summary = df.groupby("RERA")[TARGET].mean()
    axes[0, 1].bar(
        ["Non-RERA (0)", "RERA Approved (1)"],
        rera_summary.values,
        color=["#e74c3c", "#2ecc71"],
        width=0.45,
    )
    axes[0, 1].set_title("Average Price: RERA Approved vs Unapproved", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Mean Price (₹ in Lakhs)", fontsize=10)
    for i, v in enumerate(rera_summary.values):
        axes[0, 1].text(i, v + 2, f"₹{v:.1f}L", ha="center", fontweight="bold")

    # Panel C: Square Feet vs Price Scatter
    sample_scatter = df.sample(n=min(2500, len(df)), random_state=42)
    sample_scatter = sample_scatter[sample_scatter[TARGET] <= 500]
    axes[1, 0].scatter(
        sample_scatter["SQUARE_FT"],
        sample_scatter[TARGET],
        alpha=0.35,
        s=18,
        color="#8e44ad",
        edgecolors="none",
    )
    axes[1, 0].set_title("Property Area (Sq.Ft.) vs Price (₹ Lakhs)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Area in Square Feet", fontsize=10)
    axes[1, 0].set_ylabel("Price (₹ in Lakhs)", fontsize=10)
    axes[1, 0].set_xlim(0, 4500)

    # Panel D: Price by Posted_By
    posted_summary = df.groupby("POSTED_BY")[TARGET].median()
    posted_summary.plot(kind="bar", ax=axes[1, 1], color="#34495e", edgecolor="none")
    axes[1, 1].set_title("Median Price by Listing Entity (Posted By)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Listing Entity", fontsize=10)
    axes[1, 1].set_ylabel("Median Price (₹ in Lakhs)", fontsize=10)
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    overview_path = CHARTS_DIR / "market_overview_panels.png"
    plt.savefig(overview_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {overview_path}")


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    feat = engineer_features(cleaned)
    run_eda(feat)
    print("EDA Visualizations Successfully Generated.")
