"""
Streamlit Web GUI for All-India Real Estate Valuation Index & Price Prediction Engine.
Provides an interactive Indian property price estimator, city & BHK valuation index explorer,
model benchmark scorecard, SHAP explainability hub, and Power BI data export downloads.

Launch with:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import importlib

# Safely import and reload configuration to prevent Streamlit hot-reload stale cache
import src.config
importlib.reload(src.config)

from src.config import (
    CHARTS_DIR,
    POWERBI_DIR,
    TARGET,
)

try:
    from src.config import TOP_CITIES
except ImportError:
    TOP_CITIES = [
        "Bangalore", "Mumbai", "Pune", "Noida", "Kolkata", "Chennai",
        "Ghaziabad", "Jaipur", "Chandigarh", "Faridabad", "Mohali",
        "Gurgaon", "Vadodara", "Surat", "Nagpur", "Lucknow", "Indore",
        "Bhubaneswar", "Hyderabad", "Kochi", "Lalitpur", "Maharashtra",
    ]

import src.predict_service
importlib.reload(src.predict_service)
from src.predict_service import IndianValuationService
from src.main import run_pipeline

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Real Estate Valuation Engine",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .val-highlight {
        font-size: 2.3rem;
        font-weight: 800;
        color: #047857;
    }
    .interval-badge {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e40af;
        background-color: #dbeafe;
        padding: 5px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .stat-label {
        font-size: 0.88rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data Loaders (Direct file readers to prevent stale cache issues)
# -----------------------------------------------------------------------------
def get_model_metrics() -> pd.DataFrame:
    metrics_file = POWERBI_DIR / "model_metrics.csv"
    if metrics_file.exists():
        try:
            return pd.read_csv(metrics_file)
        except Exception:
            pass
    return pd.DataFrame()


def get_valuation_index() -> pd.DataFrame:
    val_file = POWERBI_DIR / "valuation_index.csv"
    if val_file.exists():
        try:
            df = pd.read_csv(val_file)
            if "City" in df.columns:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def get_feature_importance() -> pd.DataFrame:
    fi_file = POWERBI_DIR / "feature_importance.csv"
    if fi_file.exists():
        try:
            return pd.read_csv(fi_file)
        except Exception:
            pass
    return pd.DataFrame()


def get_predictions() -> pd.DataFrame:
    pred_file = POWERBI_DIR / "predictions.csv"
    if pred_file.exists():
        try:
            return pd.read_csv(pred_file)
        except Exception:
            pass
    return pd.DataFrame()


def get_master_dataset_sample() -> pd.DataFrame:
    master_file = POWERBI_DIR / "cleaned_engineered_dataset.csv"
    if master_file.exists():
        try:
            return pd.read_csv(master_file, nrows=100)
        except Exception:
            pass
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🇮🇳 All-India Real Estate Valuation Index & Price Prediction Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated pricing intelligence, 80% confidence interval risk modeling, '
    'and BHK market indices across 29,450+ residential listings in major Indian cities.</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Property Input Parameters
# -----------------------------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1582407947304-fd86f028f716?w=600&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.header("🏡 Property Estimator Parameters")
st.sidebar.caption("Configure property details to calculate real-time valuation:")

# City Selection
default_cities = sorted(TOP_CITIES)
selected_city = st.sidebar.selectbox("Metro / City", default_cities, index=default_cities.index("Bangalore") if "Bangalore" in default_cities else 0)

# BHK Selection
bhk_no = st.sidebar.radio("Bedrooms (BHK)", [1, 2, 3, 4, 5], index=1, horizontal=True)

# Layout Style
bhk_or_rk = st.sidebar.selectbox("Property Layout", ["BHK (Apartment / Flat)", "RK (Studio Room-Kitchen)"], index=0)
layout_type = "RK" if "RK" in bhk_or_rk else "BHK"

# Square Footage Slider
square_ft = st.sidebar.slider("Covered / Super Built-up Area (Sq. Ft.)", min_value=300, max_value=8000, value=1250, step=25)

# Posted By
posted_by = st.sidebar.selectbox("Posted By (Seller Entity)", ["Owner (Direct)", "Dealer (Broker / Agent)", "Builder (Developer)"], index=1)
seller_clean = "Owner" if "Owner" in posted_by else ("Builder" if "Builder" in posted_by else "Dealer")

# Key Market Status Flags
st.sidebar.markdown("**Regulatory & Possession Status**")
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    is_rera = st.checkbox("RERA Approved", value=True, help="Certified under Real Estate Regulatory Authority")
    is_ready = st.checkbox("Ready To Move", value=True, help="Immediate possession vs under-construction")
with col_s2:
    is_resale = st.checkbox("Resale Market", value=True, help="Secondary market sale vs fresh developer booking")
    is_uc = st.checkbox("Under Construction", value=not is_ready, help="Property currently being built")

st.sidebar.markdown("---")
st.sidebar.caption("System Status: **Indian Real Estate Model Active**")

# -----------------------------------------------------------------------------
# Real-Time Valuation Service Execution
# -----------------------------------------------------------------------------
service = IndianValuationService.get_instance()
try:
    valuation_result = service.predict(
        city=selected_city,
        bhk_no=int(bhk_no),
        square_ft=float(square_ft),
        posted_by=seller_clean,
        bhk_or_rk=layout_type,
        rera=1 if is_rera else 0,
        ready_to_move=1 if is_ready else 0,
        resale=1 if is_resale else 0,
        under_construction=1 if is_uc else 0,
    )
except Exception as e:
    valuation_result = None
    st.error(f"Inference Engine notice: {e}")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "🏡 Live Valuation Estimator",
    "📈 City Valuation Index Explorer",
    "📊 Model Benchmarking",
    "🧠 Valuation Drivers & SHAP",
    "📥 Power BI Data Hub & Downloads",
    "🔄 Pipeline Management",
])

# =============================================================================
# TAB 1: Live Valuation Estimator
# =============================================================================
with tabs[0]:
    st.subheader("Property Valuation Appraisal Card")
    st.caption(f"Estimated market fair value for a **{bhk_no} BHK ({square_ft:,.0f} Sq.Ft.)** in **{selected_city}**")

    if valuation_result:
        val_col1, val_col2, val_col3 = st.columns([1.5, 1.2, 1.2])

        with val_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="stat-label">Estimated Fair Market Valuation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="val-highlight">{valuation_result["formatted_price"]}</div>', unsafe_allow_html=True)
            st.caption(f"Base price: ₹ {valuation_result['predicted_price_lakhs']:.2f} Lakhs")
            st.markdown("</div>", unsafe_allow_html=True)

        with val_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="stat-label">80% Prediction Risk Interval</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="interval-badge" style="margin-top: 10px; margin-bottom: 8px;">'
                f'{valuation_result["formatted_lower"]}  ⟷  {valuation_result["formatted_upper"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Range: ₹ {valuation_result['lower_bound_lakhs']:.2f}L — ₹ {valuation_result['upper_bound_lakhs']:.2f}L")
            st.markdown("</div>", unsafe_allow_html=True)

        with val_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="stat-label">Implied Rate per Square Foot</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size: 2.1rem; font-weight: 700; color: #2563eb; margin-top: 4px;">'
                f'₹ {valuation_result["price_per_sqft"]:,.0f} <span style="font-size: 1rem; color: #64748b;">/ sqft</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Computed over {square_ft:,.0f} sq.ft. area")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Property Characteristics Breakdown Table
        st.markdown("#### Appraisal Configuration Summary")
        spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)
        spec_col1.metric("City / Metro", selected_city)
        spec_col2.metric("Configuration", f"{bhk_no} BHK ({layout_type})")
        spec_col3.metric("RERA Status", "✅ Approved" if is_rera else "⚠️ Unapproved")
        spec_col4.metric("Seller Listing", seller_clean)

        st.markdown("<br>", unsafe_allow_html=True)

        # Diagnostic Visualizations from Training
        st.markdown("#### Model Quality & Prediction Verification")
        v_col1, v_col2 = st.columns(2)
        act_chart = CHARTS_DIR / "actual_vs_predicted.png"
        res_chart = CHARTS_DIR / "residuals_plot.png"

        if act_chart.exists():
            v_col1.image(str(act_chart), caption="Actual vs. Predicted Prices (Test Set)", use_container_width=True)
        if res_chart.exists():
            v_col2.image(str(res_chart), caption="Model Residual Diagnostics (Error Distribution)", use_container_width=True)

# =============================================================================
# TAB 2: City Valuation Index Explorer
# =============================================================================
with tabs[1]:
    st.subheader("📈 Indian Real Estate Valuation Index Explorer")
    st.caption("Normalized valuation index (Base 100.0 = National Median ₹/SqFt) across Indian urban micro-markets and BHK tiers.")

    val_idx_df = get_valuation_index()

    if not val_idx_df.empty and "City" in val_idx_df.columns:
        # Multi-metro comparison visual
        top6_chart = CHARTS_DIR / "valuation_index_top6.png"
        if top6_chart.exists():
            st.image(str(top6_chart), caption="Valuation Index by BHK Tier: Top 6 Indian Metros", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Filter & Explore Market Index Data")

        c1, c2 = st.columns(2)
        available_cities = sorted(val_idx_df["City"].unique().tolist())
        filter_city = c1.multiselect("Filter Cities", available_cities, default=[c for c in ["Bangalore", "Mumbai", "Pune", "Noida", "Kolkata", "Chennai"] if c in available_cities])
        filter_bhk = c2.multiselect("Filter BHK Tiers", [1, 2, 3, 4, 5], default=[1, 2, 3, 4])

        filtered_idx = val_idx_df.copy()
        if filter_city:
            filtered_idx = filtered_idx[filtered_idx["City"].isin(filter_city)]
        if filter_bhk:
            filtered_idx = filtered_idx[filtered_idx["BHK_NO"].isin(filter_bhk)]

        st.dataframe(
            filtered_idx,
            use_container_width=True,
            column_config={
                "ValuationIndex": st.column_config.NumberColumn("Valuation Index", format="%.2f"),
                "AvgPrice_Lakhs": st.column_config.NumberColumn("Avg Price (₹ Lakhs)", format="₹ %.2f L"),
                "MedianPrice_Lakhs": st.column_config.NumberColumn("Median Price (₹ Lakhs)", format="₹ %.2f L"),
                "AvgPricePerSqFt": st.column_config.NumberColumn("Avg Rate (₹/SqFt)", format="₹ %d"),
                "PropertyCount": st.column_config.NumberColumn("Sample Volume", format="%d"),
            },
        )
    else:
        st.warning("Valuation Index data not found. Please run the pipeline to generate `valuation_index.csv`.")

# =============================================================================
# TAB 3: Model Benchmarking
# =============================================================================
with tabs[2]:
    st.subheader("📊 Multi-Model Performance Benchmark")
    st.caption("Rigorous evaluation comparing Linear Regression, Random Forest, and XGBoost using 5-Fold Cross Validation.")

    metrics_df = get_model_metrics()

    if not metrics_df.empty:
        st.dataframe(
            metrics_df,
            use_container_width=True,
            column_config={
                "Train_R2": st.column_config.NumberColumn("Train R²", format="%.4f"),
                "Test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
                "CV_R2_Mean": st.column_config.NumberColumn("5-Fold CV R² (Mean)", format="%.4f"),
                "CV_R2_Std": st.column_config.NumberColumn("CV R² Std Dev (±)", format="%.4f"),
                "RMSE": st.column_config.NumberColumn("RMSE (₹ in Lakhs)", format="₹ %.2f L"),
                "MAE": st.column_config.NumberColumn("MAE (₹ in Lakhs)", format="₹ %.2f L"),
            },
        )

        st.markdown("<br>", unsafe_allow_html=True)
        comp_img = CHARTS_DIR / "model_comparison.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Model Benchmark Comparison: R² Accuracy & Lakhs Error Metrics", use_container_width=True)
    else:
        st.warning("Model metrics not found. Run the pipeline to populate model benchmarking tables.")

# =============================================================================
# TAB 4: Valuation Drivers & SHAP Attributions
# =============================================================================
with tabs[3]:
    st.subheader("🧠 Key Valuation Drivers & Explainable AI (SHAP)")
    st.caption("Feature attribution ranking property characteristics driving residential real estate prices in India.")

    fi_col1, fi_col2 = st.columns(2)
    fi_chart = CHARTS_DIR / "feature_importance_top20.png"
    shap_chart = CHARTS_DIR / "shap_summary.png"

    if fi_chart.exists():
        fi_col1.image(str(fi_chart), caption="Top 20 Valuation Drivers (XGBoost Relative Importance)", use_container_width=True)
    if shap_chart.exists():
        fi_col2.image(str(shap_chart), caption="SHAP TreeExplainer Summary Plot", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Feature Importance Leaderboard")
    fi_df = get_feature_importance()
    if not fi_df.empty:
        st.dataframe(
            fi_df,
            use_container_width=True,
            column_config={
                "Importance": st.column_config.ProgressColumn("Relative Importance", min_value=0.0, max_value=float(fi_df["Importance"].max()), format="%.4f"),
                "Rank": st.column_config.NumberColumn("Rank", format="#%d"),
            },
        )

# =============================================================================
# TAB 5: Power BI Data Hub & Downloads
# =============================================================================
with tabs[4]:
    st.subheader("📥 Power BI Ingestion Hub & Curated Datasets")
    st.caption("Download cleaned, non-empty, relational CSV tables ready for instant import into Microsoft Power BI or Tableau.")

    pbi_files = [
        ("predictions.csv", "Out-of-sample test property predictions, 80% interval bounds, errors, and metadata slicers."),
        ("valuation_index.csv", "City and BHK level price metrics, average ₹/sqft, volume, and normalized valuation index."),
        ("feature_importance.csv", "Ranked property price sensitivity drivers extracted from the champion model."),
        ("model_metrics.csv", "Benchmark scorecard comparing Linear Regression, Random Forest, and XGBoost."),
        ("cleaned_engineered_dataset.csv", "Master property inventory fact table with 29,200+ clean listings."),
    ]

    for fname, desc in pbi_files:
        fpath = POWERBI_DIR / fname
        if fpath.exists():
            f_size_kb = fpath.stat().st_size / 1024.0
            col_d1, col_d2, col_d3 = st.columns([2, 5, 2])
            col_d1.markdown(f"**`{fname}`**")
            col_d2.caption(f"{desc} ({f_size_kb:,.1f} KB)")
            with open(fpath, "rb") as f:
                col_d3.download_button(
                    label=f"⬇️ Download CSV",
                    data=f.read(),
                    file_name=fname,
                    mime="text/csv",
                    key=f"dl_{fname}",
                )
            st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

    st.markdown("#### Preview Test Predictions Table (`predictions.csv`)")
    pred_sample = get_predictions()
    if not pred_sample.empty:
        st.dataframe(pred_sample.head(25), use_container_width=True)

# =============================================================================
# TAB 6: Pipeline Management
# =============================================================================
with tabs[5]:
    st.subheader("🔄 Automated Pipeline Retraining Hub")
    st.caption("Execute full end-to-end retraining: ingestion, cleaning, feature engineering, modeling, 5-fold CV, and artifact exports.")

    if st.button("🚀 Run Full Pipeline Now", type="primary"):
        with st.spinner("Executing end-to-end training pipeline on 29,450+ Indian properties..."):
            try:
                run_pipeline()
                st.success("✅ End-to-end pipeline finished successfully! Refreshing dashboard data...")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline execution encountered an error: {e}")

    st.markdown("---")
    st.markdown("#### Exploratory Diagnostic Visualizations")
    eda_col1, eda_col2 = st.columns(2)

    dist_chart = CHARTS_DIR / "target_distribution.png"
    city_chart = CHARTS_DIR / "city_price_summary.png"
    corr_chart = CHARTS_DIR / "correlation_heatmap.png"
    panels_chart = CHARTS_DIR / "market_overview_panels.png"

    if dist_chart.exists():
        eda_col1.image(str(dist_chart), caption="Price Distribution in ₹ Lakhs (Raw vs Log)", use_container_width=True)
    if city_chart.exists():
        eda_col2.image(str(city_chart), caption="Median Prices & Volume across Top Indian Cities", use_container_width=True)
    if corr_chart.exists():
        eda_col1.image(str(corr_chart), caption="Feature Correlation Heatmap", use_container_width=True)
    if panels_chart.exists():
        eda_col2.image(str(panels_chart), caption="4-Panel Indian Market Overview", use_container_width=True)
