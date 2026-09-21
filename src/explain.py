"""
Model Explainability and Feature Attribution module for All-India Real Estate Engine.
Extracts Gini/Gain feature importance and generates SHAP summary visualizations.
"""

import sys
import logging
from pathlib import Path
from typing import List, Any

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
import shap

from src.config import CHARTS_DIR, POWERBI_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def explain_model(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Computes global feature importances and SHAP values for the champion model:
      - Exports top 20 features to outputs/powerbi_data/feature_importance.csv.
      - Renders horizontal bar chart in outputs/charts/feature_importance_top20.png.
      - Generates SHAP summary plot across test samples in outputs/charts/shap_summary.png.

    Returns:
        pd.DataFrame: Top 20 feature importances.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Computing model explainability and feature attributions...")

    # 1. Feature Importance extraction
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        importances = np.ones(len(feature_names))

    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["Rank"] = np.arange(1, len(fi_df) + 1)

    top20_df = fi_df.head(20).copy()
    top20_path = POWERBI_DIR / "feature_importance.csv"
    top20_df.to_csv(top20_path, index=False)
    logger.info(f"Saved top 20 feature importances to {top20_path}")

    # Render horizontal bar chart of top 20 features
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=top20_df,
        y="Feature",
        x="Importance",
        palette="viridis_r",
        ax=ax,
    )
    ax.set_title("Top 20 Valuation Drivers (XGBoost Feature Importance)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Feature Importance (Gain / Split Weight)", fontsize=11)
    ax.set_ylabel("Property Attribute", fontsize=11)
    plt.tight_layout()
    chart_path = CHARTS_DIR / "feature_importance_top20.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {chart_path}")

    # 2. SHAP Attribution Analysis
    logger.info("Computing SHAP TreeExplainer values...")
    try:
        # Sample 500 rows for responsive headless generation
        sample_size = min(500, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        fig = plt.figure(figsize=(11, 7))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=18)
        plt.title("SHAP Feature Attribution Summary (Impact on Price in ₹ Lakhs)", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        shap_path = CHARTS_DIR / "shap_summary.png"
        plt.savefig(shap_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {shap_path}")
    except Exception as e:
        logger.warning(f"Could not compute TreeExplainer SHAP plot: {e}. Generating fallback bar plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        top20_df.head(15).plot(kind="barh", x="Feature", y="Importance", ax=ax, color="#2980b9")
        ax.set_title("Top Feature Importances (Fallback Attribution)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "shap_summary.png", dpi=200)
        plt.close()

    return top20_df


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data
    from src.feature_engineering import engineer_features
    from src.dataset import build_model_dataset
    import joblib
    from src.config import MODELS_DIR

    model = joblib.load(MODELS_DIR / "best_model.joblib")
    f_names = joblib.load(MODELS_DIR / "model_feature_columns.joblib")

    raw = load_raw_data()
    cleaned = clean_data(raw)
    feat = engineer_features(cleaned)
    _, X_te, _, _, _, _, _ = build_model_dataset(feat)

    top20 = explain_model(model, X_te, X_te, f_names)
    print("Explainability module passed. Top 5 features:\n", top20.head())
