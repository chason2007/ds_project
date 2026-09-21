"""
Streamlit Web Application for Indian House Price Prediction & Valuation.
Features:
- Property Valuer & Deal Arbitrage Meter
- Reverse Affordability & City Matcher
- Market Trends & Mortgage Calculator
- Model Benchmarks & Power BI Data Hub
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
    page_title="Indian Real Estate Valuation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Polished, minimal UI styling
st.markdown(
    """
    <style>
    /* Global layout tuning */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    /* Clean metric card */
    .hero-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .hero-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .hero-price {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 0.9rem;
        color: #475569;
    }
    .deal-badge {
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        margin-top: 6px;
    }
    .badge-favorable {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-fair {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .badge-premium {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-overpriced {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
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


# State defaults for input presets
if "city" not in st.session_state:
    st.session_state.city = "Bangalore"
if "layout" not in st.session_state:
    st.session_state.layout = "BHK"
if "bhk" not in st.session_state:
    st.session_state.bhk = 2
if "sqft" not in st.session_state:
    st.session_state.sqft = 1250
if "posted_by" not in st.session_state:
    st.session_state.posted_by = "Dealer"
if "rera" not in st.session_state:
    st.session_state.rera = True
if "ready" not in st.session_state:
    st.session_state.ready = True
if "resale" not in st.session_state:
    st.session_state.resale = True


# Sidebar: Property Inputs
with st.sidebar:
    st.subheader("Property Specification")

    city = st.selectbox("City", sorted(TOP_CITIES), key="city")
    layout = st.selectbox("Layout Type", ["BHK", "RK"], key="layout")

    if layout == "BHK":
        bhk = st.radio("Bedrooms (BHK)", [1, 2, 3, 4, 5], horizontal=True, key="bhk")
    else:
        bhk = 1
        st.caption("Studio Room-Kitchen (1 RK Layout)")

    sqft = st.slider("Area (Sq. Ft.)", min_value=250, max_value=6000, step=25, key="sqft")
    posted_by = st.selectbox("Posted By", ["Dealer", "Owner", "Builder"], key="posted_by")

    st.markdown("---")
    st.caption("Attributes")
    c_s1, c_s2 = st.columns(2)
    rera = c_s1.checkbox("RERA Approved", key="rera")
    ready = c_s2.checkbox("Ready To Move", key="ready")
    resale = c_s1.checkbox("Resale Market", key="resale")
    under_construction = not ready

    st.markdown("---")
    st.caption("Model: XGBoost Regressor (Test R2: 0.757)")


# Run Live Valuation
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

property_label = f"{bhk} BHK" if layout == "BHK" else "1 RK Studio"


# Header
col_h1, col_h2 = st.columns([4, 1])
col_h1.title("Indian Real Estate Valuation")
col_h1.caption("Predictive pricing, deal arbitrage analysis, and city-by-city affordability intelligence.")


# 4 Clean Navigation Tabs
tab_val, tab_afford, tab_market, tab_model = st.tabs([
    "Property Valuer",
    "Affordability Matcher",
    "Market Trends & EMI",
    "Model & Data Hub",
])


# =============================================================================
# TAB 1: PROPERTY VALUER & DEAL ARBITRAGE
# =============================================================================
with tab_val:
    # Quick Example Presets
    col_p0, col_p1, col_p2, col_p3, col_p4 = st.columns([1.5, 2, 2, 2, 2])
    col_p0.caption("Quick Presets:")
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
        # Primary Hero Valuation & Deal Check Card
        with st.container(border=True):
            col_v1, col_v2 = st.columns([3, 2])

            with col_v1:
                st.markdown(
                    f"""
                    <div class="hero-label">Estimated Fair Market Value</div>
                    <div class="hero-price">{pred['formatted_price']}</div>
                    <div class="hero-sub">
                        <b>80% Range:</b> {pred['formatted_lower']} – {pred['formatted_upper']}<br>
                        <b>Rate:</b> ₹ {pred['price_per_sqft']:,.0f} / sq.ft. for {property_label} ({sqft:,} sq.ft. in {city})
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_v2:
                asking_price_val = st.number_input(
                    "Listing Asking Price (₹ in Lakhs)",
                    min_value=1.0,
                    max_value=5000.0,
                    value=float(round(pred["predicted_price_lakhs"], 2)),
                    step=1.0,
                    help="Enter an actual asking price to benchmark it against market fair value.",
                )

                deal = evaluate_deal(
                    predicted_price_lakhs=pred["predicted_price_lakhs"],
                    asking_price_lakhs=asking_price_val,
                    lower_bound_lakhs=pred["lower_bound_lakhs"],
                    upper_bound_lakhs=pred["upper_bound_lakhs"],
                    rera=1 if rera else 0,
                    ready_to_move=1 if ready else 0,
                    resale=1 if resale else 0,
                    square_ft=float(sqft),
                )

                badge_class = "badge-fair"
                if "Discount" in deal["verdict"] or "Below" in deal["verdict"]:
                    badge_class = "badge-favorable"
                elif "Premium" in deal["verdict"]:
                    badge_class = "badge-premium"
                elif "Overpriced" in deal["verdict"]:
                    badge_class = "badge-overpriced"

                st.markdown(
                    f"""
                    <div class="deal-badge {badge_class}">{deal['verdict']}</div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                        Valuation Spread: <b>{deal['delta_lakhs']:+.2f} Lakhs ({deal['pct_diff']:+.1f}%)</b><br>
                        Deal Rating: <b>{deal['deal_score']} / 100</b> | {deal['interval_status']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 2-Column Visual Breakdown: Benchmark Range & Cross-City Comparison
        c_ch1, c_ch2 = st.columns(2)

        with c_ch1:
            with st.container(border=True):
                st.subheader("Price Benchmark Range")
                st.caption(f"Asking Price vs. Model Fair Value and 80% Confidence Interval for {city}:")
                benchmark_df = pd.DataFrame([
                    {"Benchmark": "Lower 80% Bound", "Price (₹ Lakhs)": pred["lower_bound_lakhs"]},
                    {"Benchmark": "Fair Market Value", "Price (₹ Lakhs)": pred["predicted_price_lakhs"]},
                    {"Benchmark": "Asking Price", "Price (₹ Lakhs)": deal["asking_price_lakhs"]},
                    {"Benchmark": "Upper 80% Bound", "Price (₹ Lakhs)": pred["upper_bound_lakhs"]},
                ]).set_index("Benchmark")
                st.bar_chart(benchmark_df["Price (₹ Lakhs)"])

        with c_ch2:
            with st.container(border=True):
                st.subheader("Cross-City Value Comparison")
                st.caption(f"Estimated price for this {property_label} ({sqft:,} sq.ft.) across Indian metros:")
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
                    comp_rows.append({"City": c, "Price (Lakhs)": res["predicted_price_lakhs"]})
                comp_df = pd.DataFrame(comp_rows).set_index("City")
                st.bar_chart(comp_df["Price (Lakhs)"])

        # Collapsible Detailed Sensitivity Curve
        with st.expander(f"View Price vs. Area Curve for {city}"):
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
                    "Area (Sq.Ft.)": a,
                    "Predicted Fair Value": res["predicted_price_lakhs"],
                    "Lower Bound": res["lower_bound_lakhs"],
                    "Upper Bound": res["upper_bound_lakhs"],
                })
            curve_df = pd.DataFrame(curve_rows).set_index("Area (Sq.Ft.)")
            st.line_chart(curve_df)


# =============================================================================
# TAB 2: AFFORDABILITY MATCHER (REVERSE SEARCH)
# =============================================================================
with tab_afford:
    st.subheader("Reverse Affordability & City Matcher")
    st.caption("Discover which Indian cities maximize your purchasing power and floor area for your budget.")

    with st.container(border=True):
        c_b1, c_b2, c_b3, c_b4 = st.columns([3, 2, 2, 2])
        budget_val = c_b1.slider("Target Budget (₹ in Lakhs)", min_value=15, max_value=350, value=75, step=5)
        bhk_sel = c_b2.radio("Bedrooms", [1, 2, 3, 4, 5], index=1, horizontal=True, key="aff_bhk_radio")
        min_area = c_b3.number_input("Min. Area (Sq.Ft.)", min_value=300, max_value=4000, value=850, step=50)
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
            # 3 High-Impact KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric("Cities in Budget", f"{len(aff_cities)} Cities", f"{bhk_sel} BHK >= {min_area} sqft")

            top_sqft_row = aff_cities.iloc[aff_cities["Achievable_SqFt"].argmax()]
            k2.metric("Maximum Space", f"{top_sqft_row['Achievable_SqFt']:,.0f} sq.ft.", f"In {top_sqft_row['City']}")

            major_names = ["Bangalore", "Pune", "Noida", "Chennai", "Kolkata", "Mumbai", "Gurgaon"]
            metro_sub = aff_cities[aff_cities["City"].isin(major_names)]
            if not metro_sub.empty:
                best_metro = metro_sub.sort_values("Affordability_Pct", ascending=False).iloc[0]
                k3.metric("Top Affordable Metro", f"{best_metro['City']}", f"{best_metro['Affordability_Pct']}% of listings in budget")

            st.markdown("<br>", unsafe_allow_html=True)

            # Metro Comparison Chart
            with st.container(border=True):
                st.subheader(f"Purchasing Power: Achievable Floor Area for ₹ {budget_val} Lakhs")
                st.caption(f"Estimated square footage ₹ {budget_val} Lakhs buys at local median rates across key metros:")
                compare_cities = ["Jaipur", "Chandigarh", "Kolkata", "Noida", "Chennai", "Bangalore", "Pune", "Gurgaon", "Mumbai"]
                metro_chart_data = aff_cities[aff_cities["City"].isin(compare_cities)][["City", "Achievable_SqFt"]].set_index("City")
                if not metro_chart_data.empty:
                    st.bar_chart(metro_chart_data["Achievable_SqFt"])

            # Filterable Table
            st.markdown("<br>", unsafe_allow_html=True)
            search_query = st.text_input("Filter results by city name:", placeholder="Type city...")
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
                    "Median_Rate_SqFt": st.column_config.NumberColumn("Rate (₹/SqFt)", format="₹ %d"),
                    "Total_Listings": st.column_config.NumberColumn("Listings Count", format="%d"),
                    "Top_Localities": "Top Localities in Budget",
                },
            )

            # Collapsible Micro-Market Neighborhood Drilldown
            with st.expander("Explore Specific Neighborhoods & Localities"):
                drill_cities = sorted(aff_cities["City"].unique().tolist())
                default_idx = drill_cities.index("Bangalore") if "Bangalore" in drill_cities else 0
                selected_drill_city = st.selectbox("Select City", drill_cities, index=default_idx)

                loc_results = get_city_localities_in_budget(master_df, selected_drill_city, budget_val, bhk_sel)
                if not loc_results.empty:
                    st.dataframe(
                        loc_results,
                        use_container_width=True,
                        column_config={
                            "Locality": "Neighborhood",
                            "Affordable_Count": st.column_config.NumberColumn("Listings Count", format="%d"),
                            "Avg_Price_Lakhs": st.column_config.NumberColumn("Avg Price (Lakhs)", format="₹ %.2f L"),
                            "Median_Price_Lakhs": st.column_config.NumberColumn("Median Price (Lakhs)", format="₹ %.2f L"),
                            "Avg_SqFt": st.column_config.NumberColumn("Avg Area", format="%d sq.ft."),
                            "Avg_Rate_SqFt": st.column_config.NumberColumn("Avg Rate", format="₹ %d/sqft"),
                        },
                    )
                else:
                    st.info(f"No listings found in {selected_drill_city} for {bhk_sel} BHK within ₹ {budget_val} Lakhs.")
        else:
            st.warning("No cities match these exact criteria. Adjust the budget or minimum area.")
    else:
        st.warning("Master dataset not available. Run pipeline to generate it.")


# =============================================================================
# TAB 3: MARKET TRENDS & EMI CALCULATOR
# =============================================================================
with tab_market:
    # Top Section: City Valuation Index
    with st.container(border=True):
        st.subheader("City Valuation & BHK Rate Index")
        st.caption("Compare property price rates across major Indian cities and bedroom configurations:")

        val_idx_df = load_valuation_index()
        if not val_idx_df.empty and "City" in val_idx_df.columns:
            f_col1, f_col2, f_col3 = st.columns([3, 2, 2])
            available = sorted(val_idx_df["City"].unique().tolist())
            sel_cities = f_col1.multiselect("Cities to Compare", available, default=[c for c in ["Bangalore", "Mumbai", "Pune", "Noida", "Kolkata", "Chennai"] if c in available])
            sel_bhk = f_col2.multiselect("BHK Tiers", [1, 2, 3, 4, 5], default=[1, 2, 3, 4])
            metric_choice = f_col3.selectbox("Metric", ["Valuation Index (Base 100)", "Average Rate (₹/SqFt)", "Median Price (₹ Lakhs)"])

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

            if not filtered.empty and sel_cities:
                piv = filtered.pivot_table(index="BHK_NO", columns="City", values=active_metric_col, aggfunc="mean")
                st.line_chart(piv)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bottom Section: Mortgage & Rental Yield Planner
    with st.container(border=True):
        st.subheader("Mortgage & Rental Yield Planner")
        st.caption("Financing and rental estimation based on current property valuation:")

        if pred:
            price_lakhs = pred["predicted_price_lakhs"]
            total_inr = price_lakhs * 100000.0

            col_m1, col_m2, col_m3 = st.columns(3)
            down_pct = col_m1.slider("Down Payment (%)", min_value=10, max_value=50, value=20, step=5)
            rate_annual = col_m2.slider("Interest Rate (% p.a.)", min_value=6.5, max_value=12.0, value=8.5, step=0.25)
            tenure = col_m3.slider("Tenure (Years)", min_value=5, max_value=30, value=20, step=5)

            down_amt = total_inr * (down_pct / 100.0)
            loan_amt = total_inr - down_amt
            r_month = (rate_annual / 12.0) / 100.0
            n_months = tenure * 12

            emi = loan_amt * r_month * ((1 + r_month) ** n_months) / (((1 + r_month) ** n_months) - 1) if r_month > 0 else loan_amt / n_months
            monthly_rent = (total_inr * 0.028) / 12.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Down Payment", f"₹ {down_amt/100000:.2f} Lakhs", f"{down_pct}% of property")
            m2.metric("Loan Principal", f"₹ {loan_amt/100000:.2f} Lakhs", f"{tenure} years")
            m3.metric("Monthly EMI", f"₹ {emi:,.0f} / mo", f"{rate_annual}% interest")
            m4.metric("Est. Rental Yield", f"₹ {monthly_rent:,.0f} / mo", "~2.8% yield")

            st.caption(f"Recommended minimum household net income for this EMI is approximately ₹ {emi * 2:,.0f} / month.")


# =============================================================================
# TAB 4: MODEL & DATA HUB
# =============================================================================
with tab_model:
    st.subheader("Model Evaluation & Data Exports")
    st.caption("Verification metrics, explainability, and analytical datasets for Power BI / Tableau.")

    metrics_df = load_metrics()
    if not metrics_df.empty:
        # Benchmark Table
        with st.container(border=True):
            st.subheader("Algorithm Benchmarks (5-Fold Cross Validation)")
            st.dataframe(
                metrics_df,
                use_container_width=True,
                column_config={
                    "Model": "Model",
                    "Train_R2": st.column_config.NumberColumn("Train R²", format="%.4f"),
                    "Test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
                    "CV_R2_Mean": st.column_config.NumberColumn("5-Fold CV R²", format="%.4f"),
                    "CV_R2_Std": st.column_config.NumberColumn("CV Std Dev", format="%.4f"),
                    "RMSE": st.column_config.NumberColumn("RMSE (Lakhs)", format="₹ %.2f L"),
                    "MAE": st.column_config.NumberColumn("MAE (Lakhs)", format="₹ %.2f L"),
                },
            )

        # Performance Charts
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            with st.container(border=True):
                st.subheader("R² Score Benchmark")
                st.bar_chart(metrics_df.set_index("Model")[["Test_R2", "CV_R2_Mean"]])

        with c_m2:
            with st.container(border=True):
                st.subheader("Error Benchmark (₹ in Lakhs)")
                st.bar_chart(metrics_df.set_index("Model")[["RMSE", "MAE"]])

        # Collapsible Actual vs Predicted Test Data Scatter
        with st.expander("Inspect Actual vs. Predicted Prices (Test Set Scatter)"):
            preds_df = load_predictions()
            if not preds_df.empty:
                clean_preds = preds_df[(preds_df["ActualPrice"] <= 400) & (preds_df["PredictedPrice"] <= 400)]
                sample_preds = clean_preds.sample(n=min(800, len(clean_preds)), random_state=42)
                st.scatter_chart(
                    sample_preds,
                    x="ActualPrice",
                    y="PredictedPrice",
                    color="City",
                    size="SQUARE_FT",
                )

    # Feature Importance
    fi_df = load_feature_importance()
    if not fi_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Top Valuation Drivers (Feature Importance)")
            st.caption("Relative gain / split contribution extracted from XGBoost:")
            st.bar_chart(fi_df.head(15).set_index("Feature")["Importance"])

    # Collapsible SHAP beeswarm plot
    shap_img = CHARTS_DIR / "shap_summary.png"
    if shap_img.exists():
        with st.expander("View SHAP Feature Attribution Beeswarm Plot"):
            st.caption("Direction and magnitude of each feature's marginal impact on property price across test observations:")
            st.image(str(shap_img), use_container_width=True)

    # Power BI Datasets Export
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Power BI / Tableau Relational Exports")
        st.caption("Download structured CSV tables generated by the pipeline:")

        datasets = [
            ("predictions.csv", "Test set predictions with 80% interval bounds and residuals."),
            ("valuation_index.csv", "Aggregated city and BHK index table with rates and listing volumes."),
            ("feature_importance.csv", "Top feature importances extracted from the model."),
            ("model_metrics.csv", "Benchmark evaluation metrics across all tested models."),
            ("cleaned_engineered_dataset.csv", "Master property inventory dataset with 29,216 rows."),
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

    # Pipeline trigger
    with st.expander("Pipeline Maintenance"):
        st.caption("Re-execute the end-to-end data processing and model training pipeline:")
        if st.button("Run Full Pipeline", use_container_width=True):
            with st.spinner("Executing pipeline..."):
                try:
                    run_pipeline()
                    st.success("Pipeline executed successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Execution error: {e}")
