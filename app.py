"""
BHOOMIQ — Real Estate Valuation & Market Intelligence Platform.
Executive-Grade AI Valuation Engine & Interactive Property Analytics Dashboard.
Project Submission & Viva Presentation Edition.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    POWERBI_DIR,
    CHARTS_DIR,
    TOP_CITIES,
)
from src.predict_service import BhoomIQValuationService, CITY_COORDINATES
from src.affordability import (
    match_affordable_cities,
    evaluate_deal,
    get_city_localities_in_budget,
)

# Set page configuration
st.set_page_config(
    page_title="BHOOMIQ | Real Estate Valuation & Market Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom high-end CSS typography, theme accents, and custom components
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Adjust page padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* BHOOMIQ Premium Hero Banner Header */
    .bhoomiq-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        color: #FFFFFF;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
    }
    .bhoomiq-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
        display: inline-block;
    }
    .bhoomiq-tagline {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }
    .bhoomiq-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .bhoomiq-badge {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        color: #CBD5E1;
        font-weight: 500;
    }

    /* Metric card improvements */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(226, 232, 240, 0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
    }

    /* Subheader enhancements */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    
    .sidebar-credit-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.9rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = POWERBI_DIR / "model_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    path = POWERBI_DIR / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_master_dataset() -> pd.DataFrame:
    path = POWERBI_DIR / "cleaned_engineered_dataset.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# =============================================================================
# SIDEBAR: PRESENTATION PRESETS & PROJECT CREDITS
# =============================================================================
st.sidebar.markdown("### 🏙️ BHOOMIQ Engine")
st.sidebar.caption("Real Estate Valuation & Intelligence Platform")

PRESETS = {
    "🎯 Custom Property Search": None,
    "🌆 Bangalore IT Corridor (2 BHK, 1,200 sqft)": {
        "city": "Bangalore",
        "config": "2 BHK",
        "sqft_range": (1100, 1300),
        "exact_mode": False,
        "exact_sqft": 1200,
        "ready": "Ready to Move",
        "rera": "RERA Approved",
        "resale": "Resale Property",
        "posted": "Owner",
    },
    "🌊 Mumbai Sea-Facing Prime (3 BHK, 1,650 sqft)": {
        "city": "Mumbai",
        "config": "3 BHK",
        "sqft_range": (1500, 1800),
        "exact_mode": False,
        "exact_sqft": 1650,
        "ready": "Ready to Move",
        "rera": "RERA Approved",
        "resale": "Resale Property",
        "posted": "Dealer",
    },
    "🏢 Pune Tech Hub Apartment (2 BHK, 950 sqft)": {
        "city": "Pune",
        "config": "2 BHK",
        "sqft_range": (850, 1050),
        "exact_mode": False,
        "exact_sqft": 950,
        "ready": "Under Construction",
        "rera": "RERA Approved",
        "resale": "New Builder Sale",
        "posted": "Builder",
    },
    "🏙️ Gurgaon High-Rise Residence (4 BHK, 2,800 sqft)": {
        "city": "Gurgaon",
        "config": "4 BHK",
        "sqft_range": (2600, 3000),
        "exact_mode": False,
        "exact_sqft": 2800,
        "ready": "Ready to Move",
        "rera": "RERA Approved",
        "resale": "New Builder Sale",
        "posted": "Dealer",
    },
    "🍃 Kolkata Metro Residence (2 BHK, 850 sqft)": {
        "city": "Kolkata",
        "config": "2 BHK",
        "sqft_range": (750, 950),
        "exact_mode": False,
        "exact_sqft": 850,
        "ready": "Ready to Move",
        "rera": "RERA Approved",
        "resale": "Resale Property",
        "posted": "Owner",
    },
    "🏰 Jaipur Heritage Villa (4 BHK, 2,200 sqft)": {
        "city": "Jaipur",
        "config": "4 BHK",
        "sqft_range": (2000, 2400),
        "exact_mode": False,
        "exact_sqft": 2200,
        "ready": "Ready to Move",
        "rera": "RERA Approved",
        "resale": "Resale Property",
        "posted": "Owner",
    },
}

st.sidebar.markdown("#### ⚡ Live Demo Quick Presets")
preset_choice = st.sidebar.selectbox(
    "Select a preconfigured market scenario for live demonstration:",
    list(PRESETS.keys()),
    index=0,
    help="Quickly prefill property specs to demonstrate different urban markets.",
)
active_preset = PRESETS[preset_choice]

st.sidebar.divider()
st.sidebar.markdown("#### 📊 Model Snapshot")
st.sidebar.markdown(
    """
    - **Primary Model**: XGBoost Regressor
    - **Out-of-Sample $R^2$**: `0.7569`
    - **5-Fold CV $R^2$**: `0.7751 ± 0.026`
    - **RMSE**: `₹ 69.13 Lakhs`
    - **MAE**: `₹ 25.10 Lakhs`
    - **Prediction Interval**: Dual Quantile (80% bounds)
    - **Empirical Coverage**: `77.84%`
    - **Cleaned Dataset**: 29,216 listings
    """
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    <div class="sidebar-credit-box">
        <b style="color: #E2E8F0;">🎓 Academic Project Submission</b><br>
        <b>Candidate</b>: GOPIKA S (722824104067)<br>
        <b>Guide</b>: Dr. M. Suriya, M.E., Ph.D.<br>
        <b>Department</b>: Computer Science & Engg.<br>
        <b>Institution</b>: Sri Eshwar College of Engineering<br>
        <b>Affiliation</b>: Anna University, Chennai<br>
        <b>Session</b>: October 2026
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# BHOOMIQ TOP HERO HEADER BANNER
# =============================================================================
st.markdown(
    """
    <div class="bhoomiq-hero">
        <div class="bhoomiq-title">BHOOMIQ</div>
        <div style="font-size: 1.25rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.3rem;">
            Real Estate Valuation & Market Intelligence Platform
        </div>
        <div class="bhoomiq-tagline">
            Powered by Quantile Machine Learning, SHAP Interpretability & Micro-Market Analytics across Indian Metropolitan Cities.
        </div>
        <div class="bhoomiq-badge-row">
            <span class="bhoomiq-badge">📊 29,451 Listing Data Points</span>
            <span class="bhoomiq-badge">⚡ XGBoost ML Engine (R² = 0.775)</span>
            <span class="bhoomiq-badge">🎯 77.8% Dual Quantile Bounds</span>
            <span class="bhoomiq-badge">🏙️ 22 Major Urban Markets</span>
            <span class="bhoomiq-badge">🔍 SHAP Explainability</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 1. PROPERTY SPECIFICATION INPUT CARD
# =============================================================================
st.markdown('<div class="section-header">🔍 Property Search & Specifications</div>', unsafe_allow_html=True)

# Determine default values from active preset if selected
default_city = active_preset["city"] if active_preset else "Bangalore"
default_config = active_preset["config"] if active_preset else "2 BHK"
default_sqft_range = active_preset["sqft_range"] if active_preset else (1000, 1400)
default_exact_mode = active_preset["exact_mode"] if active_preset else False
default_exact_sqft = active_preset["exact_sqft"] if active_preset else 1250
default_ready = active_preset["ready"] if active_preset else "Ready to Move"
default_rera = active_preset["rera"] if active_preset else "RERA Approved"
default_resale = active_preset["resale"] if active_preset else "Resale Property"
default_posted = active_preset["posted"] if active_preset else "Owner"

sorted_cities = sorted(TOP_CITIES)
city_idx = sorted_cities.index(default_city) if default_city in sorted_cities else 0

config_options = ["1 RK (Studio)", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"]
config_idx = config_options.index(default_config) if default_config in config_options else 2

with st.container(border=True):
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        city = st.selectbox(
            "City / Metropolitan Area",
            sorted_cities,
            index=city_idx,
            help="Select target Indian urban center.",
        )

    with col_c2:
        config_choice = st.selectbox(
            "Configuration / Bedrooms",
            config_options,
            index=config_idx,
            help="Select room configuration.",
        )
        if "RK" in config_choice:
            layout = "RK"
            bhk = 1
        else:
            layout = "BHK"
            bhk = int(config_choice.split()[0])

    # Area Range Slider with Exact Size Toggle
    col_ar1, col_ar2 = st.columns([4, 1])
    with col_ar1:
        sqft_range = st.slider(
            "Area Range (Sq. Ft.)",
            min_value=300,
            max_value=5000,
            value=default_sqft_range,
            step=50,
            help="Drag both handles to select an area range.",
        )
    with col_ar2:
        st.write("")
        st.write("")
        exact_mode = st.checkbox("Exact size", value=default_exact_mode, help="Enable to type a single exact area number")

    if exact_mode:
        exact_sqft = st.number_input("Exact Area (Sq. Ft.)", min_value=250, max_value=8000, value=default_exact_sqft, step=25)
        min_sqft, max_sqft = exact_sqft, exact_sqft
    else:
        min_sqft, max_sqft = sqft_range

    is_range = (min_sqft != max_sqft)

    # Detailed Traits Toggle
    with st.expander("⚙️ Additional Property Attributes (RERA, Handover, Listing Source)"):
        col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
        ready_idx = 0 if default_ready == "Ready to Move" else 1
        ready_choice = col_ex1.selectbox("Construction Status", ["Ready to Move", "Under Construction"], index=ready_idx)
        ready = 1 if ready_choice == "Ready to Move" else 0
        under_construction = 1 - ready

        rera_idx = 0 if default_rera == "RERA Approved" else 1
        rera_choice = col_ex2.selectbox("RERA Compliance", ["RERA Approved", "Not Approved"], index=rera_idx)
        rera = 1 if rera_choice == "RERA Approved" else 0

        resale_idx = 0 if default_resale == "Resale Property" else 1
        resale_choice = col_ex3.selectbox("Market Type", ["Resale Property", "New Builder Sale"], index=resale_idx)
        resale = 1 if resale_choice == "Resale Property" else 0

        posted_opts = ["Dealer", "Owner", "Builder"]
        posted_idx = posted_opts.index(default_posted) if default_posted in posted_opts else 1
        posted_by = col_ex4.selectbox("Posted By", posted_opts, index=posted_idx)


# Execute BHOOMIQ Inference Engine
service = BhoomIQValuationService.get_instance()
try:
    pred_min = service.predict(
        city=city,
        bhk_no=int(bhk),
        square_ft=float(min_sqft),
        posted_by=posted_by,
        bhk_or_rk=layout,
        rera=rera,
        ready_to_move=ready,
        resale=resale,
        under_construction=under_construction,
    )

    pred_max = service.predict(
        city=city,
        bhk_no=int(bhk),
        square_ft=float(max_sqft),
        posted_by=posted_by,
        bhk_or_rk=layout,
        rera=rera,
        ready_to_move=ready,
        resale=resale,
        under_construction=under_construction,
    ) if is_range else pred_min

    mid_sqft = (min_sqft + max_sqft) / 2.0
    pred_mid = service.predict(
        city=city,
        bhk_no=int(bhk),
        square_ft=float(mid_sqft),
        posted_by=posted_by,
        bhk_or_rk=layout,
        rera=rera,
        ready_to_move=ready,
        resale=resale,
        under_construction=under_construction,
    ) if is_range else pred_min
except Exception as err:
    pred_min, pred_max, pred_mid = None, None, None
    st.error(f"BHOOMIQ Valuation Inference Error: {err}")


# =============================================================================
# 2. BHOOMIQ LIVE VALUATION & DEAL FAIRNESS ENGINE
# =============================================================================
if pred_min and pred_max:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">💎 BHOOMIQ Valuation & Deal Analytics</div>', unsafe_allow_html=True)

    with st.container(border=True):
        col_val, col_deal = st.columns([3, 2])

        with col_val:
            area_text = f"{min_sqft:,} – {max_sqft:,} sq.ft." if is_range else f"{min_sqft:,} sq.ft."
            st.caption(f"Estimated for: **{city}** • **{config_choice}** • **{area_text}**")

            if is_range:
                val_text = f"{pred_min['formatted_price']} – {pred_max['formatted_price']}"
                min_rate = min(pred_min['price_per_sqft'], pred_max['price_per_sqft'])
                max_rate = max(pred_min['price_per_sqft'], pred_max['price_per_sqft'])
                rate_text = f"₹ {min_rate:,.0f} – ₹ {max_rate:,.0f} / sqft"
                range_text = f"🎯 **80% Quantile Range**: {pred_min['formatted_lower']} – {pred_max['formatted_upper']}"
            else:
                val_text = pred_min["formatted_price"]
                rate_text = f"₹ {pred_min['price_per_sqft']:,.0f} / sqft"
                range_text = f"🎯 **80% Quantile Range**: {pred_min['formatted_lower']} – {pred_max['formatted_upper']}"

            st.metric(
                label="Estimated Fair Market Value (BHOOMIQ AI)",
                value=val_text,
                delta=rate_text,
                delta_color="off",
            )
            st.markdown(range_text)

        with col_deal:
            st.caption("⚡ BHOOMIQ Deal Arbitrage & Fairness Meter")
            default_asking = float(round(pred_mid["predicted_price_lakhs"], 1))
            asking = st.number_input(
                "Active Listing Asking Price (₹ Lakhs)",
                min_value=1.0,
                max_value=5000.0,
                value=default_asking,
                step=1.0,
                key=f"asking_{city}_{bhk}_{min_sqft}",
            )

            deal = evaluate_deal(
                predicted_price_lakhs=pred_mid["predicted_price_lakhs"],
                asking_price_lakhs=asking,
                lower_bound_lakhs=pred_min["lower_bound_lakhs"],
                upper_bound_lakhs=pred_max["upper_bound_lakhs"],
                rera=rera,
                ready_to_move=ready,
                resale=resale,
                square_ft=float(mid_sqft),
            )

            if is_range:
                min_p = pred_min["predicted_price_lakhs"]
                max_p = pred_max["predicted_price_lakhs"]
                if asking < min_p:
                    spread_str = f"₹ {asking - min_p:+.2f}L vs range"
                    verdict_str = "🔥 Bargain Listing"
                elif min_p <= asking <= max_p:
                    spread_str = f"Within {min_p:.1f}L – {max_p:.1f}L band"
                    verdict_str = "✅ Fair Market Value"
                else:
                    spread_str = f"₹ {asking - max_p:+.2f}L vs range"
                    verdict_str = "⚠️ Premium / Overpriced"
            else:
                spread_str = f"{deal['delta_lakhs']:+.2f} L ({deal['pct_diff']:+.1f}%)"
                verdict_str = deal["verdict"]

            st.metric(
                label=f"Verdict: {verdict_str}",
                value=f"₹ {deal['asking_price_lakhs']:.2f} L",
                delta=spread_str,
                delta_color="inverse",
            )
            st.caption(f"Deal Rating: **{deal['deal_score']}/100** • {deal['interval_status']}")

    st.markdown("<br>", unsafe_allow_html=True)


    # =============================================================================
    # 3. INTERACTIVE DEEP-DIVE TABS
    # =============================================================================
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "🏦 Mortgage & EMI Planner",
        "🏙️ City Benchmark Matrix & Geo Map",
        "🎯 Affordability & Micro-Market Matcher",
        "📊 AI Models & SHAP Explainability",
        "💾 Power BI / Tableau Export Hub",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Loan & EMI Planner
    # -------------------------------------------------------------------------
    with subtab1:
        st.subheader("🏦 Financing & Mortgage EMI Planner")
        st.caption(f"Financial breakdown based on BHOOMIQ fair valuation of **{pred_mid['formatted_price']}** in **{city}**:")

        total_inr = pred_mid["predicted_price_lakhs"] * 100000.0
        c_emi1, c_emi2, c_emi3 = st.columns(3)
        down_pct = c_emi1.slider("Down Payment (%)", 10, 50, 20, 5)
        rate_annual = c_emi2.slider("Interest Rate (% p.a.)", 7.0, 11.0, 8.5, 0.25)
        tenure = c_emi3.slider("Tenure (Years)", 5, 30, 20, 5)

        down_amt = total_inr * (down_pct / 100.0)
        loan_amt = total_inr - down_amt
        r_month = (rate_annual / 12.0) / 100.0
        n_months = tenure * 12
        emi = loan_amt * r_month * ((1 + r_month) ** n_months) / (((1 + r_month) ** n_months) - 1) if r_month > 0 else loan_amt / n_months
        total_payment = emi * n_months
        total_interest = total_payment - loan_amt

        with st.container(border=True):
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Monthly EMI", f"₹ {emi:,.0f} / mo")
            e2.metric("Down Payment", f"₹ {down_amt/100000:.2f} Lakhs")
            e3.metric("Loan Principal", f"₹ {loan_amt/100000:.2f} Lakhs")
            e4.metric("Total Interest", f"₹ {total_interest/100000:.2f} Lakhs")

        st.caption(f"Total Repayment over {tenure} years: **₹ {total_payment/100000:.2f} Lakhs** (Principal: {(loan_amt/total_payment)*100:.0f}%, Interest: {(total_interest/total_payment)*100:.0f}%)")
        st.info(f"💡 **Income Eligibility Rule of Thumb**: A minimum recommended gross household monthly income for this loan commitment is approx **₹ {emi * 2:,.0f} / month**.")

    # -------------------------------------------------------------------------
    # TAB 2: Compare Cities Benchmark & Geo Spatial Overview
    # -------------------------------------------------------------------------
    with subtab2:
        st.subheader("🏙️ Price Benchmark & Spatial Overview Across Major Metros")
        st.caption(f"Comparing estimated valuation for **{config_choice} ({area_text})** across Indian metropolitan real estate markets:")

        compare_cities = ["Mumbai", "Bangalore", "Pune", "Noida", "Gurgaon", "Chennai", "Kolkata", "Hyderabad"]
        comp_rows = []
        for c in compare_cities:
            res = service.predict(
                city=c,
                bhk_no=int(bhk),
                square_ft=float(mid_sqft),
                posted_by=posted_by,
                bhk_or_rk=layout,
                rera=rera,
                ready_to_move=ready,
                resale=resale,
                under_construction=under_construction,
            )
            comp_rows.append({
                "City": c,
                "Estimated Price (₹ Lakhs)": res["predicted_price_lakhs"],
                "Formatted Price": res["formatted_price"],
                "Rate / SqFt": f"₹ {res['price_per_sqft']:,.0f}",
            })
        comp_df = pd.DataFrame(comp_rows)

        col_b1, col_b2 = st.columns([3, 2])
        with col_b1:
            st.markdown("#### Valuation by City")
            st.bar_chart(comp_df.set_index("City")["Estimated Price (₹ Lakhs)"])

        with col_b2:
            st.markdown("#### Comparative Rate Matrix")
            st.dataframe(
                comp_df[["City", "Formatted Price", "Rate / SqFt"]],
                use_container_width=True,
                hide_index=True,
            )

        # Interactive Geographic Map of Indian Metros
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🗺️ Geographic Distribution of Indian Real Estate Metros")
        map_points = []
        for c_name, (c_lat, c_lon) in CITY_COORDINATES.items():
            map_points.append({
                "City": c_name,
                "lat": c_lat,
                "lon": c_lon,
            })
        map_df = pd.DataFrame(map_points)
        st.map(map_df, latitude="lat", longitude="lon", zoom=4)

        # Expandable Reference: Valuation Index Plot
        vi_chart_path = CHARTS_DIR / "valuation_index_top6.png"
        if vi_chart_path.exists():
            with st.expander("📈 View Multi-Tier Benchmark Index (Top 6 Metros vs National Baseline)"):
                st.image(str(vi_chart_path), use_container_width=True, caption="Figure: Valuation Index by BHK tier across top 6 metros (Base 100 = National Median ₹/sqft)")

    # -------------------------------------------------------------------------
    # TAB 3: BHOOMIQ Affordability & Budget Matcher
    # -------------------------------------------------------------------------
    with subtab3:
        st.subheader("🎯 BHOOMIQ Affordability & Micro-Market Matcher")
        st.caption("Reverse-search Indian cities and specific micro-markets that maximize purchasing power and floor area for your target budget.")

        c_af1, c_af2 = st.columns([3, 2])
        b_budget = c_af1.slider("Target Property Budget (₹ in Lakhs)", min_value=20, max_value=300, value=75, step=5)
        b_bhk = c_af2.radio("Bedrooms (BHK)", [1, 2, 3, 4], index=1, horizontal=True)

        master_df = load_master_dataset()
        if not master_df.empty:
            aff = match_affordable_cities(master_df, budget_lakhs=b_budget, bhk_no=b_bhk, min_sqft=600)
            if not aff.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"#### Achievable Carpet Area for ₹ {b_budget} Lakhs ({b_bhk} BHK)")
                metros = ["Jaipur", "Chandigarh", "Kolkata", "Noida", "Chennai", "Bangalore", "Pune", "Gurgaon", "Mumbai"]
                m_chart = aff[aff["City"].isin(metros)][["City", "Achievable_SqFt"]].set_index("City")
                if not m_chart.empty:
                    st.bar_chart(m_chart["Achievable_SqFt"])

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### Top Matched Indian Metros")
                top_aff = aff.head(10)[["City", "Affordability_Pct", "Median_Price_Lakhs", "Achievable_SqFt", "Top_Localities"]]
                st.dataframe(
                    top_aff,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "City": "City",
                        "Affordability_Pct": st.column_config.NumberColumn("In-Budget Inventory (%)", format="%.0f%%"),
                        "Median_Price_Lakhs": st.column_config.NumberColumn("Median Price", format="₹ %.1f L"),
                        "Achievable_SqFt": st.column_config.NumberColumn("Achievable Area", format="%d sqft"),
                        "Top_Localities": "Popular Micro-Markets",
                    },
                )

                # Micro-Market Drilldown for Specific City
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📍 Explore Micro-Markets in a Specific City within Budget")
                drill_cities = sorted(master_df["City"].unique())
                default_drill_idx = drill_cities.index("Bangalore") if "Bangalore" in drill_cities else 0
                chosen_drill_city = st.selectbox(
                    "Select City to inspect affordable localities:",
                    drill_cities,
                    index=default_drill_idx,
                )
                loc_df = get_city_localities_in_budget(master_df, city=chosen_drill_city, budget_lakhs=b_budget, bhk_no=b_bhk)
                if not loc_df.empty:
                    st.dataframe(
                        loc_df.head(12),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Locality": "Micro-Market / Locality",
                            "Affordable_Count": st.column_config.NumberColumn("Available Listings"),
                            "Avg_Price_Lakhs": st.column_config.NumberColumn("Average Price", format="₹ %.1f L"),
                            "Median_Price_Lakhs": st.column_config.NumberColumn("Median Price", format="₹ %.1f L"),
                            "Avg_SqFt": st.column_config.NumberColumn("Average Area", format="%d sqft"),
                            "Avg_Rate_SqFt": st.column_config.NumberColumn("Avg Rate", format="₹ %d / sqft"),
                        },
                    )
                else:
                    st.info(f"No listings found in **{chosen_drill_city}** for a budget of ₹ {b_budget} Lakhs ({b_bhk} BHK). Try increasing the budget or reducing BHK configuration.")

    # -------------------------------------------------------------------------
    # TAB 4: AI Model Performance & SHAP Analytics
    # -------------------------------------------------------------------------
    with subtab4:
        st.subheader("📊 BHOOMIQ Machine Learning Benchmark & Diagnostic Visualizations")
        st.caption("Rigorous cross-validation, prediction interval estimation, and SHAP interpretability across 29,216 cleaned listings.")

        metrics_df = load_metrics()
        if not metrics_df.empty:
            with st.container(border=True):
                st.markdown("**Benchmark Evaluation Metrics across Regression Architectures:**")
                st.dataframe(
                    metrics_df[["Model", "Test_R2", "CV_R2_Mean", "CV_R2_Std", "RMSE", "MAE"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model": "Model Architecture",
                        "Test_R2": st.column_config.NumberColumn("Test R² Score", format="%.4f"),
                        "CV_R2_Mean": st.column_config.NumberColumn("5-Fold CV R² (Mean)", format="%.4f"),
                        "CV_R2_Std": st.column_config.NumberColumn("CV R² Std Dev", format="± %.4f"),
                        "RMSE": st.column_config.NumberColumn("RMSE (₹ Lakhs)", format="₹ %.2f L"),
                        "MAE": st.column_config.NumberColumn("MAE (₹ Lakhs)", format="₹ %.2f L"),
                    },
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Diagnostic Visualizations & Interpretability Gallery")

        diag_options = {
            "🏆 Model Benchmark Comparison": ("model_comparison.png", "Compares Test R² vs 5-Fold Cross-Validation R² and error metrics (RMSE/MAE) across Linear Regression, Random Forest, and XGBoost."),
            "🐝 SHAP Feature Attribution (TreeExplainer)": ("shap_summary.png", "SHAP summary beeswarm plot demonstrating global feature impact on property valuation (in ₹ Lakhs). Area (Log_Sqft), micro-market location, and bedroom counts drive the strongest directional shifts."),
            "🎯 Actual vs. Predicted Fit": ("actual_vs_predicted.png", "Scatter plot of actual out-of-sample test prices against predicted valuations along the ideal identity line (y = x)."),
            "📉 Residuals Analysis & Error Distribution": ("residuals_plot.png", "Residuals vs. fitted values and residuals histogram confirming zero-centered, unbiased prediction errors."),
            "🔝 Top 20 Valuation Drivers": ("feature_importance_top20.png", "Top 20 property attributes ranked by relative XGBoost gain importance."),
            "📊 Market Dynamics (4-Panel Grid)": ("market_overview_panels.png", "4-panel EDA exploratory overview showing price distributions across BHK configurations, market categories, and developer types."),
            "📈 Target Price Distribution & Log Transform": ("target_distribution.png", "Raw positive-skewed property price distribution vs log-transformed normal approximation."),
            "🏗️ Pipeline Architecture Block Diagram": ("block_diagram.png", "End-to-end data flow: Data Ingestion -> Preprocessing -> Feature Engineering -> ML Training -> Quantile Regressors -> SHAP Interpretability -> Streamlit / Power BI."),
        }

        selected_diag = st.radio(
            "Select diagnostic visualization to inspect:",
            list(diag_options.keys()),
            horizontal=True,
        )

        chart_filename, chart_desc = diag_options[selected_diag]
        chart_file_path = CHARTS_DIR / chart_filename
        if chart_file_path.exists():
            st.image(str(chart_file_path), use_container_width=True, caption=chart_desc)
            st.info(f"💡 **Viva Defense Note**: {chart_desc}")
        else:
            st.warning(f"Visualization '{chart_filename}' is not found in outputs/charts/. Run 'python run.py' to generate.")

    # -------------------------------------------------------------------------
    # TAB 5: Power BI Data Hub
    # -------------------------------------------------------------------------
    with subtab5:
        st.subheader("💾 Power BI / Tableau Relational Data Exports")
        st.caption("Download structured CSV export tables generated by the BHOOMIQ automated pipeline for external BI tools and viva presentation.")

        datasets = [
            ("predictions.csv", "Model test set predictions, actual prices, residuals, and 80% quantile confidence bounds"),
            ("valuation_index.csv", "Aggregated City x BHK tier valuation index matrix with average and median rates per square foot"),
            ("cleaned_engineered_dataset.csv", "Cleaned master property inventory with engineered features and parsed localities"),
            ("model_metrics.csv", "Comparative evaluation metrics (R², CV mean/std, RMSE, MAE) across regression models"),
            ("feature_importance.csv", "Top 20 valuation drivers ranked by relative feature gain importance"),
        ]

        for fname, desc in datasets:
            fpath = POWERBI_DIR / fname
            if fpath.exists():
                with st.container(border=True):
                    c_h1, c_h2 = st.columns([3, 1])
                    c_h1.markdown(f"**`{fname}`**")
                    c_h1.caption(desc)
                    with open(fpath, "rb") as f:
                        file_bytes = f.read()
                        c_h2.download_button(
                            label=f"📥 Download ({len(file_bytes)/1024:.0f} KB)",
                            data=file_bytes,
                            file_name=fname,
                            mime="text/csv",
                            key=f"dl_{fname}",
                            use_container_width=True,
                        )
                    # Expandable preview of first 3 rows
                    with st.expander(f"Preview table schema ({fname})"):
                        preview_df = pd.read_csv(fpath, nrows=3)
                        st.dataframe(preview_df, use_container_width=True)
