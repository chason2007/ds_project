"""
Streamlit Web GUI for All-India Real Estate Valuation Index & Price Prediction Engine.
Provides an interactive Indian property price estimator, city & BHK valuation index explorer,
mortgage and investment calculator, model benchmark scorecard, SHAP explainability hub,
and Power BI data export downloads.

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
# Page Configuration & Modern PropTech Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BharatVal | Real Estate Valuation Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Typography & Font Smoothing */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0369a1 100%);
        padding: 28px 36px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #93c5fd;
        font-weight: 400;
        max-width: 900px;
        line-height: 1.5;
    }
    .hero-tags {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Appraisal Metric Cards */
    .appraisal-card {
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .appraisal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    .card-val {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        border-color: #bbf7d0;
    }
    .card-interval {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        border-color: #bfdbfe;
    }
    .card-rate {
        background: linear-gradient(135deg, #ffffff 0%, #faf5ff 100%);
        border-color: #e9d5ff;
    }

    .card-tag {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .price-display {
        font-size: 2.6rem;
        font-weight: 800;
        color: #065f46;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .range-display {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e40af;
        margin-top: 10px;
        margin-bottom: 6px;
    }
    .rate-display {
        font-size: 2.4rem;
        font-weight: 800;
        color: #6b21a8;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .card-subtext {
        font-size: 0.86rem;
        color: #64748b;
    }

    /* Spec Pill Badges */
    .spec-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .spec-pill-green {
        background-color: #ecfdf5;
        color: #047857;
        border-color: #a7f3d0;
    }
    .spec-pill-blue {
        background-color: #eff6ff;
        color: #1d4ed8;
        border-color: #bfdbfe;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 20px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-caption {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 16px;
    }

    /* Financial ROI Cards */
    .finance-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .finance-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data Loaders (Direct file readers to eliminate cache issues)
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
# Session State Initialization for Quick Presets
# -----------------------------------------------------------------------------
if "city_input" not in st.session_state:
    st.session_state.city_input = "Bangalore"
if "bhk_input" not in st.session_state:
    st.session_state.bhk_input = 2
if "sqft_input" not in st.session_state:
    st.session_state.sqft_input = 1250
if "posted_input" not in st.session_state:
    st.session_state.posted_input = "Dealer (Broker / Agent)"
if "layout_input" not in st.session_state:
    st.session_state.layout_input = "BHK (Apartment / Flat)"
if "rera_input" not in st.session_state:
    st.session_state.rera_input = True
if "ready_input" not in st.session_state:
    st.session_state.ready_input = True
if "resale_input" not in st.session_state:
    st.session_state.resale_input = True


# -----------------------------------------------------------------------------
# Hero Banner Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">
            <span>BharatVal Real Estate Intelligence</span>
        </div>
        <div class="hero-subtitle">
            Enterprise Automated Real Estate Valuation Index, Predictive Pricing & Investment Analytics
            trained on 29,450+ residential properties across premier Indian urban centers.
        </div>
        <div class="hero-tags">
            <span class="hero-pill">Real-Time XGBoost Engine</span>
            <span class="hero-pill">80% Quantile Risk Bounds</span>
            <span class="hero-pill">22+ Metro Markets</span>
            <span class="hero-pill">RERA & Resale Modeling</span>
            <span class="hero-pill">Power BI Relational Hub</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Property Input Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Property Specifications")
    st.caption("Adjust parameters to recalculate market value in real-time:")

    default_cities = sorted(TOP_CITIES)
    selected_city = st.selectbox("Metro / City", default_cities, key="city_input")

    bhk_no = st.radio("Bedrooms (BHK)", [1, 2, 3, 4, 5], key="bhk_input", horizontal=True)

    bhk_or_rk = st.selectbox("Property Layout", ["BHK (Apartment / Flat)", "RK (Studio Room-Kitchen)"], key="layout_input")
    layout_type = "RK" if "RK" in bhk_or_rk else "BHK"

    square_ft = st.slider("Covered / Super Built-up Area (Sq. Ft.)", min_value=300, max_value=8000, step=25, key="sqft_input")

    posted_by = st.selectbox("Posted By (Seller Entity)", ["Owner (Direct)", "Dealer (Broker / Agent)", "Builder (Developer)"], key="posted_input")
    seller_clean = "Owner" if "Owner" in posted_by else ("Builder" if "Builder" in posted_by else "Dealer")

    st.markdown("---")
    st.markdown("**Regulatory & Market Stage**")
    is_rera = st.checkbox("RERA Certified Project", key="rera_input", help="Certified under Real Estate Regulatory Authority")
    is_ready = st.checkbox("Ready To Move", key="ready_input", help="Immediate possession vs future handover date")
    is_resale = st.checkbox("Secondary Resale Market", key="resale_input", help="Resale flat vs fresh developer booking")
    is_uc = not is_ready

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
            <strong>Engine:</strong> XGBoost Champion (R²: 0.76)<br>
            <strong>Training Grain:</strong> 29,451 Pan-India Listings<br>
            <strong>Currency:</strong> Indian Rupee (₹ Lakhs & Crores)
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    st.error(f"Inference Engine alert: {e}")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "Valuation Appraisal",
    "Metro Valuation Index",
    "Mortgage & Investment ROI",
    "Model Benchmarking",
    "Valuation Drivers (SHAP)",
    "Power BI Data Hub",
])

# =============================================================================
# TAB 1: Live Valuation Appraisal
# =============================================================================
with tabs[0]:
    # Quick Preset Bar
    st.markdown("<div class='section-title'>Market Presets</div>", unsafe_allow_html=True)
    st.caption("Click any preset to instantaneously populate specs and re-evaluate pricing:")

    pr_col1, pr_col2, pr_col3, pr_col4, pr_col5 = st.columns(5)
    if pr_col1.button("Mumbai - 2 BHK (950 sqft)", use_container_width=True):
        st.session_state.city_input = "Mumbai"
        st.session_state.bhk_input = 2
        st.session_state.sqft_input = 950
        st.session_state.posted_input = "Dealer (Broker / Agent)"
        st.session_state.rera_input = True
        st.rerun()

    if pr_col2.button("Bangalore - 3 BHK (1,650 sqft)", use_container_width=True):
        st.session_state.city_input = "Bangalore"
        st.session_state.bhk_input = 3
        st.session_state.sqft_input = 1650
        st.session_state.posted_input = "Dealer (Broker / Agent)"
        st.session_state.rera_input = True
        st.rerun()

    if pr_col3.button("Gurgaon - 4 BHK (2,800 sqft)", use_container_width=True):
        st.session_state.city_input = "Gurgaon"
        st.session_state.bhk_input = 4
        st.session_state.sqft_input = 2800
        st.session_state.posted_input = "Builder (Developer)"
        st.session_state.rera_input = True
        st.rerun()

    if pr_col4.button("Pune - 2 BHK (1,050 sqft)", use_container_width=True):
        st.session_state.city_input = "Pune"
        st.session_state.bhk_input = 2
        st.session_state.sqft_input = 1050
        st.session_state.posted_input = "Owner (Direct)"
        st.session_state.rera_input = True
        st.rerun()

    if pr_col5.button("Kolkata - 3 BHK (1,400 sqft)", use_container_width=True):
        st.session_state.city_input = "Kolkata"
        st.session_state.bhk_input = 3
        st.session_state.sqft_input = 1400
        st.session_state.posted_input = "Dealer (Broker / Agent)"
        st.session_state.rera_input = True
        st.rerun()

    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

    if valuation_result:
        # Hero Valuation Appraisal Cards
        val_col1, val_col2, val_col3 = st.columns([1.5, 1.3, 1.2])

        with val_col1:
            st.markdown(
                f"""
                <div class="appraisal-card card-val">
                    <div class="card-tag" style="color: #047857;">Estimated Fair Market Value</div>
                    <div class="price-display">{valuation_result["formatted_price"]}</div>
                    <div class="card-subtext">Base value: <strong>₹ {valuation_result["predicted_price_lakhs"]:.2f} Lakhs</strong></div>
                    <div style="margin-top: 10px;">
                        <span class="spec-pill spec-pill-green">Champion XGBoost Predictor</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with val_col2:
            st.markdown(
                f"""
                <div class="appraisal-card card-interval">
                    <div class="card-tag" style="color: #1e40af;">80% Confidence Risk Bounds</div>
                    <div class="range-display">{valuation_result["formatted_lower"]}  ⟷  {valuation_result["formatted_upper"]}</div>
                    <div class="card-subtext">Quantile Interval (α=0.10 to α=0.90)</div>
                    <div style="margin-top: 10px;">
                        <span class="spec-pill spec-pill-blue">77.84% Test Empirical Coverage</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with val_col3:
            st.markdown(
                f"""
                <div class="appraisal-card card-rate">
                    <div class="card-tag" style="color: #6b21a8;">Implied Spatial Rate</div>
                    <div class="rate-display">₹ {valuation_result["price_per_sqft"]:,.0f}</div>
                    <div class="card-subtext">per square foot over <strong>{square_ft:,.0f} sq.ft.</strong></div>
                    <div style="margin-top: 10px;">
                        <span class="spec-pill">{layout_type} Layout</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Specifications Pill Strip
        st.markdown("<div class='section-title'>Active Appraisal Parameters</div>", unsafe_allow_html=True)
        spec_badges = [
            f"Metro: <strong>{selected_city}</strong>",
            f"Configuration: <strong>{bhk_no} BHK ({layout_type})</strong>",
            f"Area: <strong>{square_ft:,.0f} Sq.Ft.</strong>",
            f"Seller: <strong>{seller_clean}</strong>",
            f"RERA: <strong>{'Approved' if is_rera else 'Unapproved'}</strong>",
            f"Possession: <strong>{'Ready to Move' if is_ready else 'Under Construction'}</strong>",
            f"Market: <strong>{'Resale' if is_resale else 'Direct Booking'}</strong>",
        ]
        st.markdown(
            f"""
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px;">
                {' '.join([f'<span class="spec-pill">{b}</span>' for b in spec_badges])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Multi-Metro Benchmark: What would this property cost in other cities?
        st.markdown("<div class='section-title'>Cross-Metro Comparative Valuation</div>", unsafe_allow_html=True)
        st.caption(f"Fair market value simulation for this exact **{bhk_no} BHK ({square_ft:,.0f} Sq.Ft.)** specification across top Indian cities:")

        comp_cities = ["Mumbai", "Bangalore", "Pune", "Noida", "Gurgaon", "Chennai", "Kolkata", "Hyderabad"]
        comp_records = []
        for c in comp_cities:
            p_val = service.predict(
                city=c,
                bhk_no=int(bhk_no),
                square_ft=float(square_ft),
                posted_by=seller_clean,
                bhk_or_rk=layout_type,
                rera=1 if is_rera else 0,
                ready_to_move=1 if is_ready else 0,
                resale=1 if is_resale else 0,
                under_construction=1 if is_uc else 0,
            )
            comp_records.append({
                "Metro": c,
                "Estimated Price (₹ Lakhs)": p_val["predicted_price_lakhs"],
                "Rate (₹/Sq.Ft.)": p_val["price_per_sqft"],
            })
        comp_df = pd.DataFrame(comp_records).set_index("Metro")
        st.bar_chart(comp_df["Estimated Price (₹ Lakhs)"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Dynamic Price vs Area Sensitivity Simulation
        st.markdown(f"<div class='section-title'>Real-Time Area Sensitivity Curve ({selected_city} — {bhk_no} BHK)</div>", unsafe_allow_html=True)
        st.caption(f"Continuous pricing curve across varying property dimensions in **{selected_city}** with 80% interval bands:")

        sim_sqfts = [600, 900, 1200, 1500, 1800, 2200, 2800, 3500]
        sim_data = []
        for s in sim_sqfts:
            res_sim = service.predict(
                city=selected_city,
                bhk_no=int(bhk_no),
                square_ft=float(s),
                posted_by=seller_clean,
                bhk_or_rk=layout_type,
                rera=1 if is_rera else 0,
                ready_to_move=1 if is_ready else 0,
                resale=1 if is_resale else 0,
                under_construction=1 if is_uc else 0,
            )
            sim_data.append({
                "Area (Sq.Ft.)": s,
                "Predicted Price (₹ Lakhs)": res_sim["predicted_price_lakhs"],
                "Lower 80% Bound": res_sim["lower_bound_lakhs"],
                "Upper 80% Bound": res_sim["upper_bound_lakhs"],
            })
        sim_df = pd.DataFrame(sim_data).set_index("Area (Sq.Ft.)")
        st.line_chart(sim_df)


# =============================================================================
# TAB 2: Metro Valuation Index
# =============================================================================
with tabs[1]:
    st.markdown("<div class='section-title'>Indian Metro Real Estate Valuation Index</div>", unsafe_allow_html=True)
    st.caption("Normalized price-to-rate indices (Base 100.0 = National Median ₹/SqFt) across Indian urban micro-markets.")

    val_idx_df = get_valuation_index()

    if not val_idx_df.empty and "City" in val_idx_df.columns:
        # Multi-metro comparison visual
        top6_chart = CHARTS_DIR / "valuation_index_top6.png"
        if top6_chart.exists():
            st.image(str(top6_chart), caption="BHK Tier Valuation Multipliers: Top 6 Indian Metros", use_container_width=True)

        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Filter & Search Metro Index Directory</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 1.5, 1.5])
        available_cities = sorted(val_idx_df["City"].unique().tolist())
        filter_city = c1.multiselect("Select Cities", available_cities, default=[c for c in ["Bangalore", "Mumbai", "Pune", "Noida", "Kolkata", "Chennai"] if c in available_cities])
        filter_bhk = c2.multiselect("BHK Configuration", [1, 2, 3, 4, 5], default=[1, 2, 3, 4])
        search_kw = c3.text_input("Quick City Search", placeholder="e.g. Pune, Jaipur...")

        filtered_idx = val_idx_df.copy()
        if filter_city:
            filtered_idx = filtered_idx[filtered_idx["City"].isin(filter_city)]
        if filter_bhk:
            filtered_idx = filtered_idx[filtered_idx["BHK_NO"].isin(filter_bhk)]
        if search_kw.strip():
            filtered_idx = filtered_idx[filtered_idx["City"].str.contains(search_kw.strip(), case=False, na=False)]

        st.dataframe(
            filtered_idx,
            use_container_width=True,
            column_config={
                "City": st.column_config.TextColumn("Metro City"),
                "BHK_NO": st.column_config.NumberColumn("BHK", format="%d BHK"),
                "ValuationIndex": st.column_config.ProgressColumn("Valuation Index", min_value=0.0, max_value=250.0, format="%.2f"),
                "AvgPrice_Lakhs": st.column_config.NumberColumn("Average Price", format="₹ %.2f L"),
                "MedianPrice_Lakhs": st.column_config.NumberColumn("Median Price", format="₹ %.2f L"),
                "AvgPricePerSqFt": st.column_config.NumberColumn("Rate per Sq.Ft.", format="₹ %d"),
                "PropertyCount": st.column_config.NumberColumn("Sampled Listings", format="%d"),
            },
        )
    else:
        st.warning("Valuation Index data not populated. Run the pipeline to regenerate `valuation_index.csv`.")


# =============================================================================
# TAB 3: Mortgage & Investment ROI
# =============================================================================
with tabs[2]:
    st.markdown("<div class='section-title'>Mortgage EMI & Real Estate Investment Yield Calculator</div>", unsafe_allow_html=True)
    st.caption("Calculate monthly home loan commitments, down payment requirements, and estimated rental yield based on your active property appraisal:")

    if valuation_result:
        prop_val_lakhs = valuation_result["predicted_price_lakhs"]
        prop_val_inr = prop_val_lakhs * 100000.0

        # Financing parameters
        f_col1, f_col2, f_col3 = st.columns(3)
        down_payment_pct = f_col1.slider("Down Payment Percentage (%)", min_value=10, max_value=50, value=20, step=5)
        interest_rate = f_col2.slider("Annual Interest Rate (%)", min_value=6.5, max_value=12.0, value=8.5, step=0.25)
        tenure_years = f_col3.slider("Loan Tenure (Years)", min_value=5, max_value=30, value=20, step=5)

        # Amortization Formula
        down_payment_amt = prop_val_inr * (down_payment_pct / 100.0)
        loan_amt = prop_val_inr - down_payment_amt
        monthly_rate = (interest_rate / 12.0) / 100.0
        months = tenure_years * 12
        if monthly_rate > 0:
            emi_amt = loan_amt * monthly_rate * ((1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
        else:
            emi_amt = loan_amt / months

        # Estimated Indian Rental Yield (2.8% benchmark)
        annual_rent = prop_val_inr * 0.028
        monthly_rent = annual_rent / 12.0

        st.markdown("<br>", unsafe_allow_html=True)

        # Investment Summary Metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.markdown(
                f"""
                <div class="finance-box" style="border-top: 4px solid #2563eb;">
                    <div class="stat-label">Down Payment Required</div>
                    <div class="finance-number">₹ {down_payment_amt/100000:.2f} L</div>
                    <div class="card-subtext">{down_payment_pct}% of property value</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col2:
            st.markdown(
                f"""
                <div class="finance-box" style="border-top: 4px solid #0284c7;">
                    <div class="stat-label">Home Loan Principal</div>
                    <div class="finance-number">₹ {loan_amt/100000:.2f} L</div>
                    <div class="card-subtext">Financed at {interest_rate}% p.a.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col3:
            st.markdown(
                f"""
                <div class="finance-box" style="border-top: 4px solid #059669;">
                    <div class="stat-label">Estimated Monthly EMI</div>
                    <div class="finance-number" style="color: #059669;">₹ {emi_amt:,.0f}</div>
                    <div class="card-subtext">{tenure_years} years ({months} months)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col4:
            st.markdown(
                f"""
                <div class="finance-box" style="border-top: 4px solid #7c3aed;">
                    <div class="stat-label">Est. Monthly Rental Yield</div>
                    <div class="finance-number" style="color: #7c3aed;">₹ {monthly_rent:,.0f}</div>
                    <div class="card-subtext">~2.8% Annual Yield</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Affordability Note:** A total loan of **₹ {loan_amt/100000:.2f} Lakhs** requires a minimum recommended net household income of approximately **₹ {emi_amt * 2:,.0f} / month** (assuming 50% EMI-to-income ratio).")


# =============================================================================
# TAB 4: Model Benchmarking
# =============================================================================
with tabs[3]:
    st.markdown("<div class='section-title'>Multi-Model Performance Scorecard</div>", unsafe_allow_html=True)
    st.caption("Rigorous cross-validation benchmark comparing Ordinary Least Squares, Random Forest, and XGBoost:")

    metrics_df = get_model_metrics()

    if not metrics_df.empty:
        # Champion summary pills
        champ_row = metrics_df.sort_values(by="Test_R2", ascending=False).iloc[0]
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Champion Algorithm", champ_row["Model"], "Winner")
        kpi2.metric("Test Accuracy (R²)", f"{champ_row['Test_R2']:.4f}", "+30% over OLS")
        kpi3.metric("Cross-Validation (5-Fold)", f"{champ_row['CV_R2_Mean']:.4f} ± {champ_row['CV_R2_Std']:.3f}", "High Stability")
        kpi4.metric("Mean Absolute Error", f"₹ {champ_row['MAE']:.2f} Lakhs", "Avg deviation")

        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            metrics_df,
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn("Modeling Framework"),
                "Train_R2": st.column_config.NumberColumn("Train R²", format="%.4f"),
                "Test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
                "CV_R2_Mean": st.column_config.NumberColumn("5-Fold CV R² (Mean)", format="%.4f"),
                "CV_R2_Std": st.column_config.NumberColumn("CV R² Std Dev (±)", format="%.4f"),
                "RMSE": st.column_config.NumberColumn("RMSE (₹ Lakhs)", format="₹ %.2f L"),
                "MAE": st.column_config.NumberColumn("MAE (₹ Lakhs)", format="₹ %.2f L"),
            },
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        comp_img = CHARTS_DIR / "model_comparison.png"
        act_img = CHARTS_DIR / "actual_vs_predicted.png"

        if comp_img.exists():
            col_m1.image(str(comp_img), caption="Model Evaluation: R² & Error Comparison", use_container_width=True)
        if act_img.exists():
            col_m2.image(str(act_img), caption="Actual vs. Predicted Prices (Test Set)", use_container_width=True)
    else:
        st.warning("Model metrics table not found. Execute the pipeline to regenerate.")


# =============================================================================
# TAB 5: Valuation Drivers & SHAP
# =============================================================================
with tabs[4]:
    st.markdown("<div class='section-title'>Key Valuation Drivers & Explainable AI (SHAP)</div>", unsafe_allow_html=True)
    st.caption("Feature attribution ranking the structural, spatial, and geographic factors that govern property prices in India:")

    fi_col1, fi_col2 = st.columns(2)
    fi_chart = CHARTS_DIR / "feature_importance_top20.png"
    shap_chart = CHARTS_DIR / "shap_summary.png"

    if fi_chart.exists():
        fi_col1.image(str(fi_chart), caption="Top 20 Valuation Drivers (XGBoost Relative Gain)", use_container_width=True)
    if shap_chart.exists():
        fi_col2.image(str(shap_chart), caption="SHAP TreeExplainer Impact Distribution", use_container_width=True)

    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Top 20 Valuation Driver Rankings</div>", unsafe_allow_html=True)
    
    fi_df = get_feature_importance()
    if not fi_df.empty:
        st.dataframe(
            fi_df,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="#%d"),
                "Feature": st.column_config.TextColumn("Property Attribute"),
                "Importance": st.column_config.ProgressColumn("Relative Importance", min_value=0.0, max_value=float(fi_df["Importance"].max()), format="%.4f"),
            },
        )


# =============================================================================
# TAB 6: Power BI Data Hub & Pipeline Trigger
# =============================================================================
with tabs[5]:
    st.markdown("<div class='section-title'>Power BI & Tableau Relational Ingestion Hub</div>", unsafe_allow_html=True)
    st.caption("Curated, non-empty, relational CSV tables ready for direct drag-and-drop ingestion into Power BI:")

    pbi_files = [
        ("predictions.csv", "5,844 out-of-sample test property predictions, 80% interval bounds, errors, and metadata slicers."),
        ("valuation_index.csv", "299 city-BHK price summaries, average ₹/sqft, sample counts, and normalized base-100 valuation index."),
        ("feature_importance.csv", "Top 20 property price drivers ranked by XGBoost feature attribution."),
        ("model_metrics.csv", "Multi-model benchmark scorecard across Linear Regression, Random Forest, and XGBoost."),
        ("cleaned_engineered_dataset.csv", "29,216 cleaned listings master property inventory fact table."),
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
                    label="Download CSV",
                    data=f.read(),
                    file_name=fname,
                    mime="text/csv",
                    key=f"dl_{fname}",
                    use_container_width=True,
                )
            st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Pipeline Retraining Trigger</div>", unsafe_allow_html=True)
    st.caption("Re-execute the automated pipeline across all 29,450+ listings to regenerate models, charts, and CSVs:")

    if st.button("Trigger Full End-to-End Retraining Run", type="primary", use_container_width=True):
        with st.spinner("Retraining pipeline on Indian real estate listings..."):
            try:
                run_pipeline()
                st.success("Retraining complete. All models and datasets updated.")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
