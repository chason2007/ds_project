"""
Streamlit Web Application for Indian House Price Prediction & Valuation Index.
Provides real-time property price estimation, city valuation indices,
mortgage calculations, model performance benchmarks, and Power BI CSV exports.
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
    CHARTS_DIR,
    POWERBI_DIR,
    TOP_CITIES,
)
from src.predict_service import IndianValuationService
from src.affordability import (
    match_affordable_cities,
    evaluate_deal,
    get_city_localities_in_budget,
)
from src.main import run_pipeline

st.set_page_config(
    page_title="Indian House Price Prediction & Valuation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, clean CSS styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value-lg {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_metrics() -> pd.DataFrame:
    path = POWERBI_DIR / "model_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_valuation_index() -> pd.DataFrame:
    path = POWERBI_DIR / "valuation_index.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "City" in df.columns:
            return df
    return pd.DataFrame()


def load_feature_importance() -> pd.DataFrame:
    path = POWERBI_DIR / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_predictions() -> pd.DataFrame:
    path = POWERBI_DIR / "predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_master_dataset() -> pd.DataFrame:
    path = POWERBI_DIR / "cleaned_engineered_dataset.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# Session state for input presets
if "city" not in st.session_state:
    st.session_state.city = "Bangalore"
if "bhk" not in st.session_state:
    st.session_state.bhk = 2
if "sqft" not in st.session_state:
    st.session_state.sqft = 1250
if "posted_by" not in st.session_state:
    st.session_state.posted_by = "Dealer"
if "layout" not in st.session_state:
    st.session_state.layout = "BHK"
if "rera" not in st.session_state:
    st.session_state.rera = True
if "ready" not in st.session_state:
    st.session_state.ready = True
if "resale" not in st.session_state:
    st.session_state.resale = True


# Application Header
st.title("Indian Real Estate Valuation & Price Prediction")
st.caption("Trained on 29,451 property listings across Indian metropolitan areas (MachineHack / Kaggle challenge).")

# Sidebar Property Inputs
with st.sidebar:
    st.header("Property Inputs")

    city = st.selectbox("City", sorted(TOP_CITIES), key="city")
    layout = st.selectbox("Layout Type", ["BHK", "RK"], key="layout", help="BHK: Bedroom-Hall-Kitchen apartment; RK: Room-Kitchen studio apartment.")

    if layout == "BHK":
        bhk = st.radio("Bedrooms (BHK)", [1, 2, 3, 4, 5], horizontal=True, key="bhk")
    else:
        bhk = 1
        st.caption("RK layout is a single studio room (1 RK). Bedrooms fixed to 1.")

    sqft = st.slider("Area (Sq. Ft.)", min_value=250, max_value=8000, step=25, key="sqft")
    posted_by = st.selectbox("Posted By", ["Dealer", "Owner", "Builder"], key="posted_by")

    st.markdown("**Status**")
    rera = st.checkbox("RERA Approved", key="rera")
    ready = st.checkbox("Ready To Move", key="ready")
    resale = st.checkbox("Resale", key="resale")
    under_construction = not ready

    st.markdown("---")
    st.caption("Model: XGBoost Regressor | Metric: Price in Lakhs INR")


# Run prediction
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
    st.error(f"Prediction error: {err}")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Price Estimator",
    "Affordability Matcher",
    "City Price Index",
    "Mortgage Calculator",
    "Model Evaluation",
    "Feature Importance",
    "Data & Exports",
])

# -----------------------------------------------------------------------------
# TAB 1: Price Estimator
# -----------------------------------------------------------------------------
with tab1:
    # Example presets
    st.caption("Load quick example:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    if col_p1.button("Mumbai 2 BHK (950 sqft)", use_container_width=True):
        st.session_state.city = "Mumbai"
        st.session_state.layout = "BHK"
        st.session_state.bhk = 2
        st.session_state.sqft = 950
        st.session_state.posted_by = "Dealer"
        st.rerun()

    if col_p2.button("Bangalore 3 BHK (1,650 sqft)", use_container_width=True):
        st.session_state.city = "Bangalore"
        st.session_state.layout = "BHK"
        st.session_state.bhk = 3
        st.session_state.sqft = 1650
        st.session_state.posted_by = "Dealer"
        st.rerun()

    if col_p3.button("Gurgaon 4 BHK (2,800 sqft)", use_container_width=True):
        st.session_state.city = "Gurgaon"
        st.session_state.layout = "BHK"
        st.session_state.bhk = 4
        st.session_state.sqft = 2800
        st.session_state.posted_by = "Builder"
        st.rerun()

    if col_p4.button("Mumbai 1 RK Studio (450 sqft)", use_container_width=True):
        st.session_state.city = "Mumbai"
        st.session_state.layout = "RK"
        st.session_state.bhk = 1
        st.session_state.sqft = 450
        st.session_state.posted_by = "Owner"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if pred:
        # Primary Metric Cards
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Price</div>
                    <div class="metric-value-lg">{pred['formatted_price']}</div>
                    <div class="metric-sub">Base prediction: ₹ {pred['predicted_price_lakhs']:.2f} Lakhs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">80% Prediction Interval</div>
                    <div class="metric-value-lg" style="font-size: 1.5rem; color: #1e40af; margin-top: 10px;">
                        {pred['formatted_lower']} - {pred['formatted_upper']}
                    </div>
                    <div class="metric-sub">10th to 90th percentile bounds</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Rate per Sq. Ft.</div>
                    <div class="metric-value-lg" style="color: #475569;">₹ {pred['price_per_sqft']:,.0f}</div>
                    <div class="metric-sub">Calculated for {sqft:,} sq.ft.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Deal Valuation & Arbitrage Meter
        st.subheader("Deal Valuation & Arbitrage Meter")
        st.caption("Evaluate an active listing asking price against estimated fair market value to identify discount opportunities or asking premiums.")

        col_d_in1, col_d_in2 = st.columns([2, 3])
        with col_d_in1:
            asking_val = st.number_input(
                "Listing Asking Price (₹ in Lakhs)",
                min_value=1.0,
                max_value=5000.0,
                value=float(round(pred["predicted_price_lakhs"], 2)),
                step=1.0,
                help="Enter the actual asking price for this property to analyze valuation spread and deal rating.",
            )

        deal_res = evaluate_deal(
            predicted_price_lakhs=pred["predicted_price_lakhs"],
            asking_price_lakhs=asking_val,
            lower_bound_lakhs=pred["lower_bound_lakhs"],
            upper_bound_lakhs=pred["upper_bound_lakhs"],
            rera=1 if rera else 0,
            ready_to_move=1 if ready else 0,
            resale=1 if resale else 0,
            square_ft=float(sqft),
        )

        with col_d_in2:
            st.markdown(f"**Deal Verdict:** `{deal_res['verdict']}`")
            st.caption(deal_res["explanation"])
            st.caption(f"Risk Interval Position: **{deal_res['interval_status']}**")

        dm1, dm2, dm3, dm4 = st.columns(4)
        dm1.metric("Asking Price", f"₹ {deal_res['asking_price_lakhs']:.2f} L", f"Fair: ₹ {pred['predicted_price_lakhs']:.2f} L")
        dm2.metric(
            "Valuation Spread",
            f"₹ {deal_res['delta_lakhs']:+.2f} L",
            f"{deal_res['pct_diff']:+.1f}% vs fair",
            delta_color="inverse",
        )
        dm3.metric("Asking Rate / Sq.Ft.", f"₹ {deal_res['asking_rate_sqft']:,.0f}", f"{deal_res['rate_delta_sqft']:+,.0f} diff", delta_color="inverse")
        dm4.metric("Deal Score", f"{deal_res['deal_score']} / 100", "0 (Overpriced) to 100 (Bargain)")

        # Benchmark comparison bar chart
        deal_chart_df = pd.DataFrame([
            {"Benchmark": "Lower 80% Bound", "Price (₹ Lakhs)": pred["lower_bound_lakhs"]},
            {"Benchmark": "Fair Market Value", "Price (₹ Lakhs)": pred["predicted_price_lakhs"]},
            {"Benchmark": "Asking Price", "Price (₹ Lakhs)": deal_res["asking_price_lakhs"]},
            {"Benchmark": "Upper 80% Bound", "Price (₹ Lakhs)": pred["upper_bound_lakhs"]},
        ]).set_index("Benchmark")
        st.bar_chart(deal_chart_df["Price (₹ Lakhs)"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Cross-city comparison
        property_label = f"{bhk} BHK" if layout == "BHK" else "1 RK Studio"
        st.subheader("Price Comparison Across Major Cities")
        st.caption(f"Estimated value of this {property_label} ({sqft:,} sq.ft.) specification in other urban centers:")

        cities_compare = ["Mumbai", "Bangalore", "Pune", "Noida", "Gurgaon", "Chennai", "Kolkata", "Hyderabad"]
        comp_rows = []
        for c in cities_compare:
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
            comp_rows.append({"City": c, "Estimated Price (Lakhs)": res["predicted_price_lakhs"]})
        comp_df = pd.DataFrame(comp_rows).set_index("City")
        st.bar_chart(comp_df["Estimated Price (Lakhs)"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Sensitivity curve
        st.subheader(f"Price vs. Area (Sq. Ft.) in {city}")
        st.caption(f"Predicted price curve and 80% interval bounds for {property_label} units in {city}:")

        areas = [600, 900, 1200, 1500, 1800, 2200, 2800, 3500]
        curve_rows = []
        for a in areas:
            res = service.predict(
                city=city,
                bhk_no=int(bhk),
                square_ft=float(a),
                posted_by=posted_by,
                bhk_or_rk=layout,
                rera=1 if rera else 0,
                ready_to_move=1 if ready else 0,
                resale=1 if resale else 0,
                under_construction=1 if under_construction else 0,
            )
            curve_rows.append({
                "Area": a,
                "Predicted Price": res["predicted_price_lakhs"],
                "Lower Bound": res["lower_bound_lakhs"],
                "Upper Bound": res["upper_bound_lakhs"],
            })
        curve_df = pd.DataFrame(curve_rows).set_index("Area")
        st.line_chart(curve_df)


# -----------------------------------------------------------------------------
# TAB 2: Affordability Matcher
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Reverse Affordability & City Matcher")
    st.caption("Input your target budget and room requirements to identify which Indian cities and micro-markets maximize your purchasing power and floor area.")

    c_b1, c_b2, c_b3, c_b4 = st.columns([3, 2, 2, 2])
    budget_val = c_b1.slider("Target Budget (₹ in Lakhs)", min_value=15, max_value=350, value=75, step=5)
    bhk_sel = c_b2.radio("Configuration", [1, 2, 3, 4, 5], index=1, horizontal=True, key="aff_bhk_radio")
    min_area = c_b3.number_input("Min. Floor Area (Sq.Ft.)", min_value=300, max_value=4000, value=850, step=50)
    rera_req = c_b4.checkbox("RERA Certified Only", value=False, key="aff_rera_chk")

    master_df = load_master_dataset()
    if not master_df.empty:
        aff_cities = match_affordable_cities(
            master_df,
            budget_lakhs=budget_val,
            bhk_no=bhk_sel,
            min_sqft=min_area,
            rera_only=rera_req,
        )

        if not aff_cities.empty:
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Cities with Inventory", f"{len(aff_cities)} Cities", f"{bhk_sel} BHK >= {min_area} sqft")

            high_aff_count = len(aff_cities[aff_cities["Affordability_Pct"] >= 75.0])
            kpi2.metric("High Affordability Hubs", f"{high_aff_count} Cities", ">= 75% listings in budget")

            top_sqft_row = aff_cities.iloc[aff_cities["Achievable_SqFt"].argmax()]
            kpi3.metric("Max Purchasing Space", f"{top_sqft_row['Achievable_SqFt']:,.0f} sq.ft.", f"In {top_sqft_row['City']}")

            # Major metros subset
            major_names = ["Bangalore", "Pune", "Noida", "Chennai", "Kolkata", "Mumbai", "Gurgaon", "Hyderabad"]
            metro_sub = aff_cities[aff_cities["City"].isin(major_names)]
            if not metro_sub.empty:
                best_metro = metro_sub.sort_values("Affordability_Pct", ascending=False).iloc[0]
                kpi4.metric("Top Affordable Metro", f"{best_metro['City']}", f"{best_metro['Affordability_Pct']}% affordable")
            else:
                kpi4.metric("Average Market Rate", f"₹ {aff_cities['Median_Rate_SqFt'].median():,.0f}/sqft", "Median across cities")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Purchasing Power Comparison: Achievable Floor Area for ₹ {budget_val} Lakhs")
            st.caption(f"Estimated square footage a ₹ {budget_val} Lakhs budget can purchase at prevailing median rates across major urban centers:")

            # Metro comparison chart
            compare_cities = ["Jaipur", "Chandigarh", "Kolkata", "Noida", "Chennai", "Bangalore", "Pune", "Gurgaon", "Mumbai"]
            metro_chart_data = aff_cities[aff_cities["City"].isin(compare_cities)][["City", "Achievable_SqFt"]].set_index("City")
            if not metro_chart_data.empty:
                st.bar_chart(metro_chart_data["Achievable_SqFt"])

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("City-by-City Inventory & Affordability Breakdown")

            search_query = st.text_input("Filter city results by name:", placeholder="e.g. Bangalore, Pune, Noida...")
            display_aff = aff_cities.copy()
            if search_query.strip():
                display_aff = display_aff[display_aff["City"].str.contains(search_query.strip(), case=False, na=False)]

            st.dataframe(
                display_aff[[
                    "City", "Affordability_Pct", "Median_Price_Lakhs", "Budget_Margin_Lakhs",
                    "Achievable_SqFt", "Median_Rate_SqFt", "Total_Listings", "Top_Localities"
                ]],
                use_container_width=True,
                column_config={
                    "City": "City",
                    "Affordability_Pct": st.column_config.NumberColumn("Affordability Rate", format="%.1f%%"),
                    "Median_Price_Lakhs": st.column_config.NumberColumn("Median Price", format="₹ %.2f L"),
                    "Budget_Margin_Lakhs": st.column_config.NumberColumn("Budget Surplus", format="₹ %+.2f L"),
                    "Achievable_SqFt": st.column_config.NumberColumn("Est. Area for Budget", format="%d sq.ft."),
                    "Median_Rate_SqFt": st.column_config.NumberColumn("Median Rate (₹/SqFt)", format="₹ %d"),
                    "Total_Listings": st.column_config.NumberColumn("Market Sample", format="%d"),
                    "Top_Localities": "Top Affordable Localities",
                },
            )

            # Micro-Market Drilldown
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Micro-Market Locality Drilldown")
            st.caption("Inspect specific neighborhoods and localities within a city that offer listings matching your budget.")

            drill_cities = sorted(aff_cities["City"].unique().tolist())
            default_idx = drill_cities.index("Bangalore") if "Bangalore" in drill_cities else 0
            selected_drill_city = st.selectbox("Select City for Locality Breakdown", drill_cities, index=default_idx)

            loc_results = get_city_localities_in_budget(master_df, selected_drill_city, budget_val, bhk_sel)
            if not loc_results.empty:
                st.dataframe(
                    loc_results,
                    use_container_width=True,
                    column_config={
                        "Locality": "Locality / Micro-Market",
                        "Affordable_Count": st.column_config.NumberColumn("Affordable Listings", format="%d"),
                        "Avg_Price_Lakhs": st.column_config.NumberColumn("Avg Price (Lakhs)", format="₹ %.2f L"),
                        "Median_Price_Lakhs": st.column_config.NumberColumn("Median Price (Lakhs)", format="₹ %.2f L"),
                        "Avg_SqFt": st.column_config.NumberColumn("Avg Area", format="%d sq.ft."),
                        "Avg_Rate_SqFt": st.column_config.NumberColumn("Avg Rate", format="₹ %d/sqft"),
                    },
                )
            else:
                st.info(f"No listings found in {selected_drill_city} for {bhk_sel} BHK within ₹ {budget_val} Lakhs.")
        else:
            st.warning("No cities found matching these exact criteria. Try adjusting the budget or minimum area.")
    else:
        st.warning("Master dataset not available. Run the pipeline to generate it.")


# -----------------------------------------------------------------------------
# TAB 3: City Price Index
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("City Valuation Index")
    st.caption("Normalized price-to-rate metrics (Base 100.0 = National Median rate per sq.ft.) grouped by city and BHK.")

    val_idx_df = load_valuation_index()

    if not val_idx_df.empty and "City" in val_idx_df.columns:
        f_col1, f_col2, f_col3 = st.columns([3, 2, 2])
        available = sorted(val_idx_df["City"].unique().tolist())
        sel_cities = f_col1.multiselect("Filter Cities", available, default=[c for c in ["Bangalore", "Mumbai", "Pune", "Noida", "Kolkata", "Chennai"] if c in available])
        sel_bhk = f_col2.multiselect("Filter BHK", [1, 2, 3, 4, 5], default=[1, 2, 3, 4])
        metric_choice = f_col3.selectbox("Comparison Metric", ["Valuation Index (Base 100)", "Average Rate (₹/SqFt)", "Median Price (₹ Lakhs)"])

        metric_col_map = {
            "Valuation Index (Base 100)": "ValuationIndex",
            "Average Rate (₹/SqFt)": "AvgPricePerSqFt",
            "Median Price (₹ Lakhs)": "MedianPrice_Lakhs",
        }
        active_metric_col = metric_col_map[metric_choice]

        filtered = val_idx_df.copy()
        if sel_cities:
            filtered = filtered[filtered["City"].isin(sel_cities)]
        if sel_bhk:
            filtered = filtered[filtered["BHK_NO"].isin(sel_bhk)]

        # Interactive dynamic multi-series line chart
        if not filtered.empty and sel_cities:
            piv = filtered.pivot_table(index="BHK_NO", columns="City", values=active_metric_col, aggfunc="mean")
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Interactive Comparison: {metric_choice} by BHK")
            st.caption("Hover over lines to view values. Chart updates live when cities or BHK filters change.")
            st.line_chart(piv)

        st.markdown("<br>", unsafe_allow_html=True)
        query = st.text_input("Filter table by city name:", placeholder="Search city...")
        display_filtered = filtered.copy()
        if query.strip():
            display_filtered = display_filtered[display_filtered["City"].str.contains(query.strip(), case=False, na=False)]

        st.dataframe(
            display_filtered,
            use_container_width=True,
            column_config={
                "City": "City",
                "BHK_NO": "BHK",
                "ValuationIndex": st.column_config.NumberColumn("Valuation Index", format="%.2f"),
                "AvgPrice_Lakhs": st.column_config.NumberColumn("Avg Price (Lakhs)", format="₹ %.2f L"),
                "MedianPrice_Lakhs": st.column_config.NumberColumn("Median Price (Lakhs)", format="₹ %.2f L"),
                "AvgPricePerSqFt": st.column_config.NumberColumn("Avg Rate (₹/SqFt)", format="₹ %d"),
                "PropertyCount": st.column_config.NumberColumn("Listings Count", format="%d"),
            },
        )
    else:
        st.warning("Valuation index file not found. Run the pipeline to generate it.")


# -----------------------------------------------------------------------------
# TAB 4: Mortgage Calculator
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Mortgage & Rental Yield Calculator")
    st.caption("Estimate monthly loan commitments and expected rental returns for the current property valuation.")

    if pred:
        price_lakhs = pred["predicted_price_lakhs"]
        total_inr = price_lakhs * 100000.0

        col_m1, col_m2, col_m3 = st.columns(3)
        down_pct = col_m1.slider("Down Payment (%)", min_value=10, max_value=50, value=20, step=5)
        rate_annual = col_m2.slider("Interest Rate (%)", min_value=6.5, max_value=12.0, value=8.5, step=0.25)
        tenure = col_m3.slider("Loan Tenure (Years)", min_value=5, max_value=30, value=20, step=5)

        down_amt = total_inr * (down_pct / 100.0)
        loan_amt = total_inr - down_amt
        r_month = (rate_annual / 12.0) / 100.0
        n_months = tenure * 12

        if r_month > 0:
            emi = loan_amt * r_month * ((1 + r_month) ** n_months) / (((1 + r_month) ** n_months) - 1)
        else:
            emi = loan_amt / n_months

        monthly_rent = (total_inr * 0.028) / 12.0

        st.markdown("<br>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Down Payment", f"₹ {down_amt/100000:.2f} Lakhs", f"{down_pct}% of total")
        k2.metric("Loan Amount", f"₹ {loan_amt/100000:.2f} Lakhs", f"Tenure: {tenure} yrs")
        k3.metric("Monthly EMI", f"₹ {emi:,.0f} / mo", f"At {rate_annual}% p.a.")
        k4.metric("Est. Monthly Rent", f"₹ {monthly_rent:,.0f} / mo", "~2.8% yield")

        st.caption(f"Note: Recommended minimum net monthly income for this EMI is approximately ₹ {emi * 2:,.0f}.")


# -----------------------------------------------------------------------------
# TAB 5: Model Evaluation
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Model Evaluation & Benchmarking")
    st.caption("Cross-validation and test set performance across Linear Regression, Random Forest, and XGBoost.")

    metrics_df = load_metrics()

    if not metrics_df.empty:
        st.dataframe(
            metrics_df,
            use_container_width=True,
            column_config={
                "Model": "Model",
                "Train_R2": st.column_config.NumberColumn("Train R²", format="%.4f"),
                "Test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
                "CV_R2_Mean": st.column_config.NumberColumn("5-Fold CV R² (Mean)", format="%.4f"),
                "CV_R2_Std": st.column_config.NumberColumn("CV R² Std Dev", format="%.4f"),
                "RMSE": st.column_config.NumberColumn("RMSE (Lakhs)", format="₹ %.2f L"),
                "MAE": st.column_config.NumberColumn("MAE (Lakhs)", format="₹ %.2f L"),
            },
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader("R² Score Benchmark")
            st.caption("Test R² vs. 5-Fold Cross-Validation R²:")
            st.bar_chart(metrics_df.set_index("Model")[["Test_R2", "CV_R2_Mean"]])

        with col_c2:
            st.subheader("Error Comparison (₹ Lakhs)")
            st.caption("Root Mean Squared Error (RMSE) vs. Mean Absolute Error (MAE):")
            st.bar_chart(metrics_df.set_index("Model")[["RMSE", "MAE"]])

        st.markdown("<br>", unsafe_allow_html=True)
        preds_df = load_predictions()
        if not preds_df.empty:
            st.subheader("Interactive Actual vs. Predicted Prices (Test Set)")
            st.caption("Hover over data points to inspect property predictions. Color-coded by city, sized by square footage.")
            clean_preds = preds_df[(preds_df["ActualPrice"] <= 400) & (preds_df["PredictedPrice"] <= 400)]
            sample_preds = clean_preds.sample(n=min(800, len(clean_preds)), random_state=42)
            st.scatter_chart(
                sample_preds,
                x="ActualPrice",
                y="PredictedPrice",
                color="City",
                size="SQUARE_FT",
            )
    else:
        st.warning("Model metrics not found. Run the training pipeline first.")


# -----------------------------------------------------------------------------
# TAB 6: Feature Importance
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("Feature Importance & Explainability")
    st.caption("Global feature importance weights and SHAP TreeExplainer attribution plots.")

    fi_df = load_feature_importance()
    if not fi_df.empty:
        st.subheader("Top 20 Valuation Drivers (XGBoost Feature Importance)")
        st.caption("Interactive bar chart: hover over features to view exact split and gain contribution.")
        st.bar_chart(fi_df.head(20).set_index("Feature")["Importance"])

    st.markdown("<br>", unsafe_allow_html=True)
    shap_img = CHARTS_DIR / "shap_summary.png"
    if shap_img.exists():
        st.subheader("SHAP Feature Attribution (Summary Plot)")
        st.caption("Displays the direction and magnitude of each feature's impact on property price (in ₹ Lakhs) across test listings.")
        st.image(str(shap_img), use_container_width=True)

    if not fi_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Feature Importance Table")
        st.dataframe(fi_df, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 7: Data & Exports
# -----------------------------------------------------------------------------
with tab7:
    st.subheader("Power BI Datasets & File Downloads")
    st.caption("Clean relational CSV files exported by the pipeline for Power BI or Tableau import:")

    datasets = [
        ("predictions.csv", "Out-of-sample test property predictions, 80% interval bounds, and residuals."),
        ("valuation_index.csv", "Aggregated city and BHK index table with average rates and listing volume."),
        ("feature_importance.csv", "Top 20 feature importances extracted from the model."),
        ("model_metrics.csv", "Benchmark scorecard for all evaluated algorithms."),
        ("cleaned_engineered_dataset.csv", "Master cleaned property inventory table with 29,216 rows."),
    ]

    for fname, desc in datasets:
        fpath = POWERBI_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024.0
            col_d1, col_d2, col_d3 = st.columns([2, 5, 2])
            col_d1.markdown(f"**`{fname}`**")
            col_d2.caption(f"{desc} ({size_kb:,.1f} KB)")
            with open(fpath, "rb") as f:
                col_d3.download_button(
                    label="Download CSV",
                    data=f.read(),
                    file_name=fname,
                    mime="text/csv",
                    key=f"dl_{fname}",
                    use_container_width=True,
                )
            st.markdown("<hr style='margin: 4px 0;'>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Pipeline Management")
    st.caption("Re-run the training pipeline from end to end:")

    if st.button("Run Full Pipeline", use_container_width=True):
        with st.spinner("Running pipeline..."):
            try:
                run_pipeline()
                st.success("Pipeline execution completed successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Error during execution: {e}")
