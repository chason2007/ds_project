"""
BHOOMIQ — Real Estate Valuation & Market Intelligence Platform.
Executive-Grade AI Valuation Engine & Interactive Property Analytics Dashboard.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    POWERBI_DIR,
    TOP_CITIES,
)
from src.predict_service import BhoomIQValuationService
from src.affordability import (
    match_affordable_cities,
    evaluate_deal,
)

# Set page configuration
st.set_page_config(
    page_title="BHOOMIQ | Real Estate Valuation & Market Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
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
        max-width: 1150px;
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

    /* Modern Card Styles */
    .val-card {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        border-radius: 14px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.25);
    }
    .val-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
        margin-bottom: 0.3rem;
    }
    .val-card-price {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .val-card-sub {
        font-size: 0.92rem;
        opacity: 0.95;
    }

    /* Verdict styling */
    .verdict-bargain {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10B981;
        color: #10B981;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .verdict-fair {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid #3B82F6;
        color: #3B82F6;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .verdict-premium {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid #F59E0B;
        color: #F59E0B;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }

    /* Subheader enhancements */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_metrics() -> pd.DataFrame:
    path = POWERBI_DIR / "model_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_feature_importance() -> pd.DataFrame:
    path = POWERBI_DIR / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_master_dataset() -> pd.DataFrame:
    path = POWERBI_DIR / "cleaned_engineered_dataset.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


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
            <span class="bhoomiq-badge">⚡ XGBoost ML Model</span>
            <span class="bhoomiq-badge">🎯 77.8% Dual Quantile Bounds</span>
            <span class="bhoomiq-badge">🏙️ 20+ Major Urban Markets</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 1. PROPERTY SPECIFICATION INPUT CARD
# =============================================================================
st.markdown('<div class="section-header">🔍 Property Search & Specifications</div>', unsafe_allow_html=True)

with st.container(border=True):
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        city = st.selectbox(
            "City / Metropolitan Area",
            sorted(TOP_CITIES),
            index=sorted(TOP_CITIES).index("Bangalore") if "Bangalore" in TOP_CITIES else 0,
            help="Select target Indian urban center.",
        )

    with col_c2:
        config_choice = st.selectbox(
            "Configuration / Bedrooms",
            ["1 RK (Studio)", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"],
            index=2,
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
            value=(1000, 1400),
            step=50,
            help="Drag both handles to select an area range.",
        )
    with col_ar2:
        st.write("")
        st.write("")
        exact_mode = st.checkbox("Exact size", value=False, help="Enable to type a single exact area number")

    if exact_mode:
        exact_sqft = st.number_input("Exact Area (Sq. Ft.)", min_value=250, max_value=8000, value=1250, step=25)
        min_sqft, max_sqft = exact_sqft, exact_sqft
    else:
        min_sqft, max_sqft = sqft_range

    is_range = (min_sqft != max_sqft)

    # Detailed Traits Toggle
    with st.expander("⚙️ Additional Property Attributes (RERA, Handover, Listing Source)"):
        col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
        ready_choice = col_ex1.selectbox("Construction Status", ["Ready to Move", "Under Construction"])
        ready = 1 if ready_choice == "Ready to Move" else 0
        under_construction = 1 - ready

        rera_choice = col_ex2.selectbox("RERA Compliance", ["RERA Approved", "Not Approved"])
        rera = 1 if rera_choice == "RERA Approved" else 0

        resale_choice = col_ex3.selectbox("Market Type", ["Resale Property", "New Builder Sale"])
        resale = 1 if resale_choice == "Resale Property" else 0

        posted_by = col_ex4.selectbox("Posted By", ["Dealer", "Owner", "Builder"])


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
                range_text = f"🎯 **80% Quantile Range**: {pred_min['formatted_lower']} – {pred_min['formatted_upper']}"

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
            st.caption(f"Score: **{deal['deal_score']}/100** • {deal['interval_status']}")

    st.markdown("<br>", unsafe_allow_html=True)


    # =============================================================================
    # 3. INTERACTIVE DEEP-DIVE TABS
    # =============================================================================
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "🏦 Mortgage & EMI Planner",
        "🏙️ City Benchmark Matrix",
        "🎯 Affordability & Budget Matcher",
        "📊 AI Models & SHAP Explainability",
        "💾 Power BI Export Hub",
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

        with st.container(border=True):
            e1, e2, e3 = st.columns(3)
            e1.metric("Monthly EMI", f"₹ {emi:,.0f} / mo")
            e2.metric("Down Payment Amount", f"₹ {down_amt/100000:.2f} L")
            e3.metric("Loan Principal Amount", f"₹ {loan_amt/100000:.2f} L")

        st.info(f"💡 **Income Eligibility Tip**: A minimum recommended monthly household income for this loan is approx **₹ {emi * 2:,.0f} / month**.")

    # -------------------------------------------------------------------------
    # TAB 2: Compare Cities Benchmark
    # -------------------------------------------------------------------------
    with subtab2:
        st.subheader("🏙️ Price Benchmark Across Major Indian Metros")
        st.caption(f"Estimated valuation for a **{config_choice} ({area_text})** across Indian metropolitan real estate markets:")

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
            comp_rows.append({"City": c, "Estimated Price (₹ Lakhs)": res["predicted_price_lakhs"]})
        comp_df = pd.DataFrame(comp_rows).set_index("City")
        st.bar_chart(comp_df["Estimated Price (₹ Lakhs)"])

    # -------------------------------------------------------------------------
    # TAB 3: BHOOMIQ Affordability & Budget Matcher
    # -------------------------------------------------------------------------
    with subtab3:
        st.subheader("🎯 BHOOMIQ Affordability & Budget Matcher")
        st.caption("Discover which Indian cities maximize your purchasing power and floor space for a target budget.")

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
                top_aff = aff.head(8)[["City", "Affordability_Pct", "Median_Price_Lakhs", "Achievable_SqFt", "Top_Localities"]]
                st.dataframe(
                    top_aff,
                    use_container_width=True,
                    column_config={
                        "City": "City",
                        "Affordability_Pct": st.column_config.NumberColumn("In-Budget Inventory (%)", format="%.0f%%"),
                        "Median_Price_Lakhs": st.column_config.NumberColumn("Median Price", format="₹ %.1f L"),
                        "Achievable_SqFt": st.column_config.NumberColumn("Achievable Area", format="%d sqft"),
                        "Top_Localities": "Popular Micro-Markets",
                    },
                )

    # -------------------------------------------------------------------------
    # TAB 4: AI Model Performance & SHAP Analytics
    # -------------------------------------------------------------------------
    with subtab4:
        st.subheader("📊 BHOOMIQ Machine Learning Benchmark & SHAP Explainability")

        metrics_df = load_metrics()
        if not metrics_df.empty:
            with st.container(border=True):
                st.markdown("**Cross-Validation & Test Metrics across ML Models (29,216 Listings):**")
                st.dataframe(
                    metrics_df[["Model", "Test_R2", "CV_R2_Mean", "RMSE", "MAE"]],
                    use_container_width=True,
                    column_config={
                        "Model": "Model Architecture",
                        "Test_R2": st.column_config.NumberColumn("Test R² Score", format="%.4f"),
                        "CV_R2_Mean": st.column_config.NumberColumn("5-Fold CV R²", format="%.4f"),
                        "RMSE": st.column_config.NumberColumn("RMSE (₹ Lakhs)", format="₹ %.2f L"),
                        "MAE": st.column_config.NumberColumn("MAE (₹ Lakhs)", format="₹ %.2f L"),
                    },
                )

        fi_df = load_feature_importance()
        if not fi_df.empty:
            with st.expander("🔥 Top Property Drivers (Feature Importance Ranking)"):
                st.bar_chart(fi_df.head(10).set_index("Feature")["Importance"])

    # -------------------------------------------------------------------------
    # TAB 5: Power BI Data Hub
    # -------------------------------------------------------------------------
    with subtab5:
        st.subheader("💾 Power BI / Tableau Relational Data Exports")
        st.caption("Download structured CSV export tables generated by the BHOOMIQ data pipeline.")

        datasets = [
            ("predictions.csv", "Model test predictions and 80% quantile confidence bounds"),
            ("valuation_index.csv", "City x BHK tier valuation index matrix"),
            ("cleaned_engineered_dataset.csv", "Cleaned & feature-engineered master housing dataset"),
        ]
        for fname, desc in datasets:
            fpath = POWERBI_DIR / fname
            if fpath.exists():
                col_f1, col_f2 = st.columns([3, 2])
                col_f1.markdown(f"**{fname}** — *{desc}*")
                with open(fpath, "rb") as f:
                    col_f2.download_button(f"📥 Download {fname}", f.read(), fname, "text/csv", key=f"dl_{fname}")
