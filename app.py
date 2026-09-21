"""
Indian Real Estate Valuation & Deal Analyzer.
Consumer-grade full-page calculator layout:
- Top search & specification card (like Zillow / 99acres)
- Instant fair-value valuation & deal checker
- Tabs for Loan EMI, City Comparisons, Affordability Search, and Model Data
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
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Clean, modern consumer calculator layout
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
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


# Application Header
st.title("Indian Real Estate Valuation")
st.caption("Fair-value price estimates, deal check, and city affordability across major Indian metropolitan markets.")


# =============================================================================
# 1. TOP SPECIFICATION CARD (Full-Page Calculator)
# =============================================================================
with st.container(border=True):
    col_c1, col_c2, col_c3 = st.columns([2, 2, 2])

    with col_c1:
        city = st.selectbox(
            "City",
            sorted(TOP_CITIES),
            index=sorted(TOP_CITIES).index("Bangalore") if "Bangalore" in TOP_CITIES else 0,
        )

    with col_c2:
        config_choice = st.selectbox(
            "Layout / Bedrooms",
            ["1 RK (Studio)", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"],
            index=2,
        )
        if "RK" in config_choice:
            layout = "RK"
            bhk = 1
        else:
            layout = "BHK"
            bhk = int(config_choice.split()[0])

    with col_c3:
        sqft = st.number_input(
            "Area (Sq. Ft.)",
            min_value=250,
            max_value=8000,
            value=1250,
            step=50,
        )

    # Optional detailed toggles
    with st.expander("Additional Property Details (Status, RERA, Seller)"):
        col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
        ready_choice = col_ex1.selectbox("Status", ["Ready to Move", "Under Construction"])
        ready = 1 if ready_choice == "Ready to Move" else 0
        under_construction = 1 - ready

        rera_choice = col_ex2.selectbox("RERA Status", ["RERA Approved", "Not Approved"])
        rera = 1 if rera_choice == "RERA Approved" else 0

        resale_choice = col_ex3.selectbox("Sale Type", ["Resale Property", "New Builder Sale"])
        resale = 1 if resale_choice == "Resale Property" else 0

        posted_by = col_ex4.selectbox("Posted By", ["Dealer", "Owner", "Builder"])


# Run Model Prediction
service = IndianValuationService.get_instance()
try:
    pred = service.predict(
        city=city,
        bhk_no=int(bhk),
        square_ft=float(sqft),
        posted_by=posted_by,
        bhk_or_rk=layout,
        rera=rera,
        ready_to_move=ready,
        resale=resale,
        under_construction=under_construction,
    )
except Exception as err:
    pred = None
    st.error(f"Valuation error: {err}")


# =============================================================================
# 2. INSTANT VALUATION & DEAL RESULT CARD
# =============================================================================
if pred:
    with st.container(border=True):
        col_val, col_deal = st.columns([3, 2])

        with col_val:
            st.caption(f"Estimated for: {city} • {config_choice} • {sqft:,} sq.ft.")
            st.metric(
                label="Estimated Fair Market Value",
                value=pred["formatted_price"],
                delta=f"₹ {pred['price_per_sqft']:,.0f} / sqft",
                delta_color="off",
            )
            st.caption(f"80% Confidence Range: {pred['formatted_lower']} – {pred['formatted_upper']}")

        with col_deal:
            st.caption("Deal Check (Listing Comparison)")
            asking = st.number_input(
                "Asking Price (₹ in Lakhs)",
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
                rera=rera,
                ready_to_move=ready,
                resale=resale,
                square_ft=float(sqft),
            )

            st.metric(
                label=f"Verdict: {deal['verdict']}",
                value=f"₹ {deal['asking_price_lakhs']:.2f} L",
                delta=f"{deal['delta_lakhs']:+.2f} L ({deal['pct_diff']:+.1f}%)",
                delta_color="inverse",
            )
            st.caption(f"Deal Rating: {deal['deal_score']}/100 • {deal['interval_status']}")

    st.markdown("<br>", unsafe_allow_html=True)


    # =============================================================================
    # 3. INTERACTIVE TOOLS & DEEP DIVE TABS
    # =============================================================================
    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "Loan & EMI",
        "Compare Cities",
        "Affordability Search",
        "Model & Data Hub",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Loan & EMI
    # -------------------------------------------------------------------------
    with subtab1:
        st.subheader("Financing & Mortgage Planner")
        st.caption(f"Based on current valuation of {pred['formatted_price']} in {city}:")

        total_inr = pred["predicted_price_lakhs"] * 100000.0
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
            e2.metric("Down Payment", f"₹ {down_amt/100000:.2f} L")
            e3.metric("Loan Principal", f"₹ {loan_amt/100000:.2f} L")

        st.caption(f"Tip: Recommended monthly household income is approximately ₹ {emi * 2:,.0f} / month.")

    # -------------------------------------------------------------------------
    # TAB 2: Compare Cities
    # -------------------------------------------------------------------------
    with subtab2:
        st.subheader("Price in Other Major Cities")
        st.caption(f"Estimated value of this {config_choice} ({sqft:,} sq.ft.) specification across Indian urban centers:")

        compare_cities = ["Mumbai", "Bangalore", "Pune", "Noida", "Gurgaon", "Chennai", "Kolkata", "Hyderabad"]
        comp_rows = []
        for c in compare_cities:
            res = service.predict(
                city=c,
                bhk_no=int(bhk),
                square_ft=float(sqft),
                posted_by=posted_by,
                bhk_or_rk=layout,
                rera=rera,
                ready_to_move=ready,
                resale=resale,
                under_construction=under_construction,
            )
            comp_rows.append({"City": c, "Price (₹ Lakhs)": res["predicted_price_lakhs"]})
        comp_df = pd.DataFrame(comp_rows).set_index("City")
        st.bar_chart(comp_df["Price (₹ Lakhs)"])

    # -------------------------------------------------------------------------
    # TAB 3: Affordability Search
    # -------------------------------------------------------------------------
    with subtab3:
        st.subheader("Where can I afford to buy?")
        st.caption("See which cities maximize your floor area for a given budget.")

        c_af1, c_af2 = st.columns([3, 2])
        b_budget = c_af1.slider("Target Budget (₹ in Lakhs)", min_value=20, max_value=300, value=75, step=5)
        b_bhk = c_af2.radio("Bedrooms", [1, 2, 3, 4], index=1, horizontal=True)

        master_df = load_master_dataset()
        if not master_df.empty:
            aff = match_affordable_cities(master_df, budget_lakhs=b_budget, bhk_no=b_bhk, min_sqft=600)
            if not aff.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader(f"Floor Area You Can Afford for ₹ {b_budget} Lakhs ({b_bhk} BHK)")
                metros = ["Jaipur", "Chandigarh", "Kolkata", "Noida", "Chennai", "Bangalore", "Pune", "Gurgaon", "Mumbai"]
                m_chart = aff[aff["City"].isin(metros)][["City", "Achievable_SqFt"]].set_index("City")
                if not m_chart.empty:
                    st.bar_chart(m_chart["Achievable_SqFt"])

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Top Cities in Budget")
                top_aff = aff.head(8)[["City", "Affordability_Pct", "Median_Price_Lakhs", "Achievable_SqFt", "Top_Localities"]]
                st.dataframe(
                    top_aff,
                    use_container_width=True,
                    column_config={
                        "City": "City",
                        "Affordability_Pct": st.column_config.NumberColumn("In Budget (%)", format="%.0f%%"),
                        "Median_Price_Lakhs": st.column_config.NumberColumn("Median Price", format="₹ %.1f L"),
                        "Achievable_SqFt": st.column_config.NumberColumn("Est. Area", format="%d sqft"),
                        "Top_Localities": "Popular Localities",
                    },
                )

    # -------------------------------------------------------------------------
    # TAB 4: Model & Data Hub
    # -------------------------------------------------------------------------
    with subtab4:
        st.subheader("Model Performance & Data Hub")

        metrics_df = load_metrics()
        if not metrics_df.empty:
            with st.container(border=True):
                st.caption("Algorithm Benchmarks (5-Fold Cross-Validation):")
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
                    col_f1, col_f2 = st.columns([3, 2])
                    col_f1.caption(f"{fname} — {desc}")
                    with open(fpath, "rb") as f:
                        col_f2.download_button(f"Download {fname}", f.read(), fname, "text/csv", key=f"dl_{fname}")
