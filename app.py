"""
Indian Real Estate Valuation & Market Intelligence.
Streamlined, uncluttered interface:
- Tab 1: Valuation & Deal Check
- Tab 2: City Affordability
- Tab 3: Mortgage EMI
- Tab 4: Model & Data
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
from src.predict_service import IndianValuationService
from src.affordability import (
    match_affordable_cities,
    evaluate_deal,
)

st.set_page_config(
    page_title="Indian Real Estate Valuation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal typography and layout styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1050px;
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


# Sidebar: Clean & Essential Inputs Only
with st.sidebar:
    st.subheader("Property")
    city = st.selectbox("City", sorted(TOP_CITIES), index=sorted(TOP_CITIES).index("Bangalore") if "Bangalore" in TOP_CITIES else 0)

    c_sb1, c_sb2 = st.columns(2)
    layout = c_sb1.selectbox("Type", ["BHK", "RK"])
    if layout == "BHK":
        bhk = c_sb2.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=1)
    else:
        bhk = 1
        c_sb2.selectbox("Bedrooms", ["1 Room"], disabled=True)

    sqft = st.slider("Area (Sq. Ft.)", min_value=300, max_value=5000, value=1250, step=50)

    # Secondary attributes tucked cleanly into an expander
    with st.expander("More options"):
        posted_by = st.selectbox("Posted By", ["Dealer", "Owner", "Builder"])
        rera = st.checkbox("RERA Approved", value=True)
        ready = st.checkbox("Ready To Move", value=True)
        resale = st.checkbox("Resale Market", value=True)
        under_construction = not ready


# Run Prediction
service = IndianValuationService.get_instance()
try:
    pred = service.predict(
        city=city,
        bhk_no=int(bhk),
        square_ft=float(sqft),
        posted_by=posted_by,
        bhk_or_rk=layout,
        rera=1 if rera else 0,
        ready_to_move=1 if ready else 0,
        resale=1 if resale else 0,
        under_construction=1 if under_construction else 0,
    )
except Exception as err:
    pred = None
    st.error(f"Inference error: {err}")

property_name = f"{bhk} BHK" if layout == "BHK" else "1 RK Studio"


# Main Header
st.title("Indian Real Estate Valuation")
st.caption("Fair market estimates, deal check, and city affordability.")

# 4 Clear, Purposeful Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Valuation & Deal",
    "City Affordability",
    "Mortgage EMI",
    "Model & Data",
])


# =============================================================================
# TAB 1: VALUATION & DEAL CHECK
# =============================================================================
with tab1:
    if pred:
        # Main Hero Card: Theme-adaptive native contrast
        with st.container(border=True):
            col_hero, col_deal = st.columns([3, 2])

            with col_hero:
                st.caption(f"{city} • {property_name} • {sqft:,} sq.ft.")
                st.metric(
                    label="Estimated Fair Market Value",
                    value=pred["formatted_price"],
                    delta=f"₹ {pred['price_per_sqft']:,.0f} / sqft",
                    delta_color="off",
                )
                st.caption(f"Expected 80% Range: {pred['formatted_lower']} – {pred['formatted_upper']}")

            with col_deal:
                st.caption("Deal Check (Asking Price)")
                asking = st.number_input(
                    "Enter Asking Price (₹ Lakhs)",
                    min_value=1.0,
                    max_value=5000.0,
                    value=float(round(pred["predicted_price_lakhs"], 1)),
                    step=1.0,
                    label_visibility="collapsed",
                )

                deal = evaluate_deal(
                    predicted_price_lakhs=pred["predicted_price_lakhs"],
                    asking_price_lakhs=asking,
                    lower_bound_lakhs=pred["lower_bound_lakhs"],
                    upper_bound_lakhs=pred["upper_bound_lakhs"],
                    rera=1 if rera else 0,
                    ready_to_move=1 if ready else 0,
                    resale=1 if resale else 0,
                    square_ft=float(sqft),
                )

                st.metric(
                    label=f"Verdict: {deal['verdict']}",
                    value=f"₹ {deal['asking_price_lakhs']:.2f} L",
                    delta=f"{deal['delta_lakhs']:+.2f} L ({deal['pct_diff']:+.1f}%)",
                    delta_color="inverse",
                )
                st.caption(f"Deal Score: {deal['deal_score']}/100 • {deal['interval_status']}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Single Clean Comparison Chart
        st.subheader("Price in Other Major Cities")
        st.caption(f"What this {property_name} ({sqft:,} sq.ft.) would cost across India:")

        compare_cities = ["Mumbai", "Bangalore", "Pune", "Noida", "Gurgaon", "Chennai", "Kolkata", "Hyderabad"]
        comp_rows = []
        for c in compare_cities:
            res = service.predict(
                city=c,
                bhk_no=int(bhk),
                square_ft=float(sqft),
                posted_by=posted_by,
                bhk_or_rk=layout,
                rera=1 if rera else 0,
                ready_to_move=1 if ready else 0,
                resale=1 if resale else 0,
                under_construction=1 if under_construction else 0,
            )
            comp_rows.append({"City": c, "Price (₹ Lakhs)": res["predicted_price_lakhs"]})
        comp_df = pd.DataFrame(comp_rows).set_index("City")
        st.bar_chart(comp_df["Price (₹ Lakhs)"])


# =============================================================================
# TAB 2: CITY AFFORDABILITY (REVERSE SEARCH)
# =============================================================================
with tab2:
    st.subheader("Where can I afford to buy?")
    st.caption("See which cities and how much floor area your budget gets you.")

    with st.container(border=True):
        c_af1, c_af2 = st.columns([3, 2])
        b_budget = c_af1.slider("Your Budget (₹ in Lakhs)", min_value=20, max_value=300, value=75, step=5)
        b_bhk = c_af2.radio("Bedrooms", [1, 2, 3, 4], index=1, horizontal=True)

    master_df = load_master_dataset()
    if not master_df.empty:
        aff = match_affordable_cities(master_df, budget_lakhs=b_budget, bhk_no=b_bhk, min_sqft=600)
        if not aff.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Floor Area for ₹ {b_budget} Lakhs ({b_bhk} BHK)")
            st.caption("Estimated square footage you can purchase at prevailing local rates:")

            metros = ["Jaipur", "Chandigarh", "Kolkata", "Noida", "Chennai", "Bangalore", "Pune", "Gurgaon", "Mumbai"]
            m_chart = aff[aff["City"].isin(metros)][["City", "Achievable_SqFt"]].set_index("City")
            if not m_chart.empty:
                st.bar_chart(m_chart["Achievable_SqFt"])

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Top Cities in Budget")
            top_aff = aff.head(10)[["City", "Affordability_Pct", "Median_Price_Lakhs", "Achievable_SqFt", "Top_Localities"]]
            st.dataframe(
                top_aff,
                use_container_width=True,
                column_config={
                    "City": "City",
                    "Affordability_Pct": st.column_config.NumberColumn("In Budget (%)", format="%.0f%%"),
                    "Median_Price_Lakhs": st.column_config.NumberColumn("Median Price", format="₹ %.1f L"),
                    "Achievable_SqFt": st.column_config.NumberColumn("Est. Area", format="%d sq.ft."),
                    "Top_Localities": "Popular Localities",
                },
            )
    else:
        st.warning("Dataset not available. Run pipeline to generate it.")


# =============================================================================
# TAB 3: MORTGAGE EMI
# =============================================================================
with tab3:
    if pred:
        st.subheader("Mortgage & EMI Estimator")
        st.caption(f"Based on {pred['formatted_price']} valuation for {property_name} in {city}:")

        total_inr = pred["predicted_price_lakhs"] * 100000.0

        col_emi1, col_emi2, col_emi3 = st.columns(3)
        down_pct = col_emi1.slider("Down Payment (%)", 10, 50, 20, 5)
        rate_annual = col_emi2.slider("Interest Rate (%)", 7.0, 11.0, 8.5, 0.25)
        tenure = col_emi3.slider("Tenure (Years)", 5, 30, 20, 5)

        down_amt = total_inr * (down_pct / 100.0)
        loan_amt = total_inr - down_amt
        r_month = (rate_annual / 12.0) / 100.0
        n_months = tenure * 12
        emi = loan_amt * r_month * ((1 + r_month) ** n_months) / (((1 + r_month) ** n_months) - 1) if r_month > 0 else loan_amt / n_months

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            e1, e2, e3 = st.columns(3)
            e1.metric("Monthly EMI", f"₹ {emi:,.0f} / mo")
            e2.metric("Down Payment", f"₹ {down_amt/100000:.2f} Lakhs")
            e3.metric("Loan Amount", f"₹ {loan_amt/100000:.2f} Lakhs")

        st.caption(f"Tip: Recommended household monthly income for this EMI is approximately ₹ {emi * 2:,.0f} / month.")


# =============================================================================
# TAB 4: MODEL & DATA
# =============================================================================
with tab4:
    st.subheader("Model Performance & Data")

    metrics_df = load_metrics()
    if not metrics_df.empty:
        with st.container(border=True):
            st.caption("Model Benchmark Scorecard:")
            st.dataframe(
                metrics_df[["Model", "Test_R2", "CV_R2_Mean", "RMSE", "MAE"]],
                use_container_width=True,
                column_config={
                    "Model": "Algorithm",
                    "Test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
                    "CV_R2_Mean": st.column_config.NumberColumn("5-Fold CV R²", format="%.4f"),
                    "RMSE": st.column_config.NumberColumn("RMSE (Lakhs)", format="₹ %.2f L"),
                    "MAE": st.column_config.NumberColumn("MAE (Lakhs)", format="₹ %.2f L"),
                },
            )

    fi_df = load_feature_importance()
    if not fi_df.empty:
        with st.expander("Top Valuation Drivers (Feature Importance)"):
            st.bar_chart(fi_df.head(10).set_index("Feature")["Importance"])

    with st.expander("Download Datasets for Power BI / Tableau"):
        datasets = [
            ("predictions.csv", "Predictions and 80% interval bounds"),
            ("valuation_index.csv", "City x BHK valuation index"),
            ("cleaned_engineered_dataset.csv", "Cleaned master property inventory"),
        ]
        for fname, desc in datasets:
            fpath = POWERBI_DIR / fname
            if fpath.exists():
                c_d1, c_d2 = st.columns([3, 2])
                c_d1.caption(f"{fname} — {desc}")
                with open(fpath, "rb") as f:
                    c_d2.download_button(f"Download {fname}", f.read(), fname, "text/csv", key=f"dl_{fname}")
