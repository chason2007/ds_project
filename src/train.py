"""
Model Training module for All-India Real Estate Valuation Engine.
Trains Linear Regression, Random Forest, and XGBoost with 5-fold CV,
evaluates metrics, trains quantile prediction interval models, and saves artifacts.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

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
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.config import (
    RANDOM_STATE,
    MODELS_DIR,
    CHARTS_DIR,
    POWERBI_DIR,
)
from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.dataset import build_model_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list,
) -> Tuple[Any, Any, Any, pd.DataFrame]:
    """
    Trains multiple models, evaluates CV and test set performance, selects the best model,
    and fits quantile prediction interval models (10th & 90th percentiles).

    Returns:
        best_model, quantile_lower, quantile_upper, metrics_df
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)

    models: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    metrics_records = []
    trained_models = {}
    test_predictions = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        logger.info(f"Training {name}...")

        # 5-Fold Cross-Validation on training split
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Train on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # In-sample & Out-of-sample evaluations
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        test_predictions[name] = test_preds

        train_r2 = float(r2_score(y_train, train_preds))
        test_r2 = float(r2_score(y_test, test_preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
        mae = float(mean_absolute_error(y_test, test_preds))

        metrics_records.append({
            "Model": name,
            "Train_R2": round(train_r2, 4),
            "Test_R2": round(test_r2, 4),
            "CV_R2_Mean": round(cv_mean, 4),
            "CV_R2_Std": round(cv_std, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
        })

        logger.info(
            f"{name} -> Test R2: {test_r2:.4f}, CV R2: {cv_mean:.4f} +/- {cv_std:.4f}, "
            f"RMSE: ₹{rmse:.2f} Lakhs, MAE: ₹{mae:.2f} Lakhs"
        )

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = POWERBI_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved model metrics to {metrics_path}")

    # Identify best performing model based on Test R2
    best_model_name = metrics_df.sort_values(by="Test_R2", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    logger.info(f"Best model selected: {best_model_name}")

    # Save best model & feature column contract
    best_model_path = MODELS_DIR / "best_model.joblib"
    features_path = MODELS_DIR / "model_feature_columns.joblib"
    joblib.dump(best_model, best_model_path)
    joblib.dump(feature_names, features_path)
    logger.info(f"Saved best model to {best_model_path} and features to {features_path}")

    # Train Quantile Regressors for asymmetric 80% Prediction Interval (10th & 90th percentiles)
    logger.info("Training Quantile Interval Regressors (alpha=0.10 and alpha=0.90)...")
    q_lower = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.10,
        max_iter=100,
        max_depth=6,
        random_state=RANDOM_STATE,
    )
    q_lower.fit(X_train, y_train)

    q_upper = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.90,
        max_iter=100,
        max_depth=6,
        random_state=RANDOM_STATE,
    )
    q_upper.fit(X_train, y_train)

    # Save quantile models
    joblib.dump(q_lower, MODELS_DIR / "quantile_lower.joblib")
    joblib.dump(q_upper, MODELS_DIR / "quantile_upper.joblib")
    logger.info("Saved quantile models to outputs/models/")

    # Empirical coverage check on test set
    lower_preds = q_lower.predict(X_test)
    upper_preds = q_upper.predict(X_test)
    coverage = float(np.mean((y_test >= lower_preds) & (y_test <= upper_preds)) * 100.0)
    logger.info(f"Empirical 80% Prediction Interval Test Coverage: {coverage:.2f}%")

    # Generate Performance Diagnostic Charts
    _generate_training_plots(metrics_df, y_test, test_predictions[best_model_name], best_model_name)

    return best_model, q_lower, q_upper, metrics_df


def _generate_training_plots(
    metrics_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """
    Renders and saves 3 model evaluation diagnostic plots.
    """
    # 1. Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    metrics_plot = metrics_df.set_index("Model")

    metrics_plot[["Test_R2", "CV_R2_Mean"]].plot(kind="bar", ax=axes[0], colormap="viridis")
    axes[0].set_title("Model Comparison: Test R² vs 5-Fold CV R²", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", rotation=0)

    metrics_plot[["RMSE", "MAE"]].plot(kind="bar", ax=axes[1], colormap="magma")
    axes[1].set_title("Model Error Comparison (RMSE & MAE in ₹ Lakhs)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Error (₹ in Lakhs)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    comp_path = CHARTS_DIR / "model_comparison.png"
    plt.savefig(comp_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {comp_path}")

    # 2. Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    # Display subset up to ₹ 600 Lakhs for clean visual inspection
    mask = (y_test <= 600) & (y_pred <= 600)
    ax.scatter(y_test[mask], y_pred[mask], alpha=0.35, color="#1f77b4", edgecolors="none", s=20)
    max_val = 600
    ax.plot([0, max_val], [0, max_val], color="crimson", linestyle="--", linewidth=1.5, label="Perfect Agreement (y = x)")
    ax.set_title(f"Actual vs. Predicted Property Price ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Price (₹ in Lakhs)", fontsize=11)
    ax.set_ylabel("Predicted Price (₹ in Lakhs)", fontsize=11)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(loc="upper left")
    plt.tight_layout()
    act_path = CHARTS_DIR / "actual_vs_predicted.png"
    plt.savefig(act_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {act_path}")

    # 3. Residuals Diagnostic Plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    res_mask = y_pred <= 600
    axes[0].scatter(y_pred[res_mask], residuals[res_mask], alpha=0.35, color="#e67e22", edgecolors="none", s=20)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Residuals vs. Fitted Values", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Price (₹ in Lakhs)", fontsize=11)
    axes[0].set_ylabel("Residual (Actual - Predicted, ₹ Lakhs)", fontsize=11)
    axes[0].set_xlim(0, 600)

    res_sub = residuals[(residuals >= -150) & (residuals <= 150)]
    sns.histplot(res_sub, kde=True, ax=axes[1], color="#8e44ad", bins=40)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title("Residuals Distribution (Zero-Centered)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Residual (₹ in Lakhs)", fontsize=11)

    plt.tight_layout()
    res_path = CHARTS_DIR / "residuals_plot.png"
    plt.savefig(res_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {res_path}")


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    feat = engineer_features(cleaned)
    X_tr, X_te, y_tr, y_te, m_tr, m_te, f_names = build_model_dataset(feat)
    best_m, q_low, q_up, metrics = train_and_evaluate(X_tr, X_te, y_tr, y_te, f_names)
    print("Training Pipeline Test Passed.")
    print("Metrics Table:\n", metrics)
