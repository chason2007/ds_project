"""
Affordability Matcher and Deal Arbitrage Evaluation module.
Provides reverse-budget city matching and deal valuation intelligence.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def match_affordable_cities(
    df: pd.DataFrame,
    budget_lakhs: float,
    bhk_no: int,
    min_sqft: float = 0.0,
    rera_only: bool = False,
    min_listings: int = 5,
) -> pd.DataFrame:
    """
    Ranks Indian cities based on affordability and purchasing power
    for a given budget, BHK requirement, and minimum square footage.
    """
    if df.empty:
        return pd.DataFrame()

    sub = df[df["BHK_NO"] == bhk_no].copy()
    if min_sqft > 0:
        sub = sub[sub["SQUARE_FT"] >= min_sqft]
    if rera_only:
        sub = sub[sub["RERA"] == 1]

    if sub.empty:
        return pd.DataFrame()

    records = []
    for city, grp in sub.groupby("City"):
        total_count = len(grp)
        if total_count < min_listings:
            continue

        affordable_grp = grp[grp["Price_Lakhs"] <= budget_lakhs]
        aff_count = len(affordable_grp)
        aff_pct = (aff_count / total_count) * 100.0

        med_price = float(grp["Price_Lakhs"].median())
        avg_price = float(grp["Price_Lakhs"].mean())
        med_rate = float(grp["Price_Per_SqFt"].median())
        
        # Achievable square footage at the median rate in this city
        achievable_sqft = (budget_lakhs * 100000.0) / med_rate if med_rate > 0 else 0.0

        # Top 3 localities with listings within budget
        if not affordable_grp.empty and "Locality" in affordable_grp.columns:
            top_locs = affordable_grp["Locality"].value_counts().head(3).index.tolist()
            top_locs_str = ", ".join(top_locs)
        else:
            top_locs_str = "None in budget"

        records.append({
            "City": city,
            "Total_Listings": total_count,
            "Affordable_Listings": aff_count,
            "Affordability_Pct": round(aff_pct, 1),
            "Median_Price_Lakhs": round(med_price, 2),
            "Avg_Price_Lakhs": round(avg_price, 2),
            "Median_Rate_SqFt": round(med_rate, 0),
            "Achievable_SqFt": round(achievable_sqft, 0),
            "Budget_Margin_Lakhs": round(budget_lakhs - med_price, 2),
            "Top_Localities": top_locs_str,
        })

    if not records:
        return pd.DataFrame()

    res_df = pd.DataFrame(records)
    # Sort primarily by Affordability Percentage, secondarily by Achievable SqFt
    res_df = res_df.sort_values(by=["Affordability_Pct", "Achievable_SqFt"], ascending=[False, False]).reset_index(drop=True)
    return res_df


def get_city_localities_in_budget(
    df: pd.DataFrame,
    city: str,
    budget_lakhs: float,
    bhk_no: int,
) -> pd.DataFrame:
    """
    Extracts localities within a chosen city that have listings fitting the budget.
    """
    if df.empty or "City" not in df.columns or "Locality" not in df.columns:
        return pd.DataFrame()

    sub = df[(df["City"] == city) & (df["BHK_NO"] == bhk_no)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub_aff = sub[sub["Price_Lakhs"] <= budget_lakhs]
    if sub_aff.empty:
        return pd.DataFrame()

    loc_df = (
        sub_aff.groupby("Locality")
        .agg(
            Affordable_Count=("Price_Lakhs", "count"),
            Avg_Price_Lakhs=("Price_Lakhs", "mean"),
            Median_Price_Lakhs=("Price_Lakhs", "median"),
            Avg_SqFt=("SQUARE_FT", "mean"),
            Avg_Rate_SqFt=("Price_Per_SqFt", "mean"),
        )
        .reset_index()
    )

    loc_df["Avg_Price_Lakhs"] = loc_df["Avg_Price_Lakhs"].round(2)
    loc_df["Median_Price_Lakhs"] = loc_df["Median_Price_Lakhs"].round(2)
    loc_df["Avg_SqFt"] = loc_df["Avg_SqFt"].round(0)
    loc_df["Avg_Rate_SqFt"] = loc_df["Avg_Rate_SqFt"].round(0)

    loc_df = loc_df.sort_values(by="Affordable_Count", ascending=False).reset_index(drop=True)
    return loc_df


def evaluate_deal(
    predicted_price_lakhs: float,
    asking_price_lakhs: float,
    lower_bound_lakhs: float,
    upper_bound_lakhs: float,
    rera: int = 1,
    ready_to_move: int = 1,
    resale: int = 1,
    square_ft: float = 1000.0,
) -> Dict[str, Any]:
    """
    Evaluates an asking price against the fair market valuation and risk bounds.
    Returns delta, percentage difference, deal verdict, and deal quality score (0-100).
    """
    delta_lakhs = asking_price_lakhs - predicted_price_lakhs
    pct_diff = (delta_lakhs / max(predicted_price_lakhs, 1.0)) * 100.0

    # Classify verdict
    if pct_diff <= -12.0:
        verdict = "High Discount Opportunity"
        verdict_color = "green"
        explanation = f"Asking price is ₹ {abs(delta_lakhs):.2f} Lakhs ({abs(pct_diff):.1f}%) below estimated fair market value. Strong buyer advantage."
    elif -12.0 < pct_diff <= -4.0:
        verdict = "Below Market Valuation"
        verdict_color = "green"
        explanation = f"Asking price is ₹ {abs(delta_lakhs):.2f} Lakhs ({abs(pct_diff):.1f}%) below fair value. Favorable negotiation position."
    elif -4.0 < pct_diff <= 4.0:
        verdict = "Fair Market Value"
        verdict_color = "blue"
        explanation = "Asking price is well aligned with prevailing machine learning valuation and historical comparable sales."
    elif 4.0 < pct_diff <= 12.0:
        verdict = "Slight Asking Premium"
        verdict_color = "orange"
        explanation = f"Asking price carries a ₹ {delta_lakhs:.2f} Lakhs ({pct_diff:.1f}%) markup over fair market value. Room for buyer negotiation."
    else:
        verdict = "Overpriced Listing"
        verdict_color = "red"
        explanation = f"Asking price is ₹ {delta_lakhs:.2f} Lakhs ({pct_diff:.1f}%) above estimated market fair value."

    # Prediction interval positioning
    if asking_price_lakhs < lower_bound_lakhs:
        interval_status = "Below 80% Lower Bound (Distress / Deep Discount)"
    elif asking_price_lakhs > upper_bound_lakhs:
        interval_status = "Above 80% Upper Bound (Outlier Premium)"
    else:
        interval_status = "Within 80% Expected Market Range"

    # Deal Quality Score (0 to 100)
    base_score = 50.0
    # Price component: +1.5 pts per 1% discount, clamped to [-30, +30]
    price_component = float(np.clip(-pct_diff * 1.5, -30.0, 30.0))
    # Risk/Trust component: RERA (+10), Ready-to-move (+10)
    rera_component = 10.0 if rera == 1 else 0.0
    ready_component = 10.0 if ready_to_move == 1 else 0.0

    deal_score = int(np.clip(base_score + price_component + rera_component + ready_component, 5.0, 98.0))

    asking_rate_sqft = (asking_price_lakhs * 100000.0) / max(square_ft, 1.0)
    fair_rate_sqft = (predicted_price_lakhs * 100000.0) / max(square_ft, 1.0)

    return {
        "asking_price_lakhs": round(asking_price_lakhs, 2),
        "predicted_price_lakhs": round(predicted_price_lakhs, 2),
        "delta_lakhs": round(delta_lakhs, 2),
        "pct_diff": round(pct_diff, 1),
        "verdict": verdict,
        "verdict_color": verdict_color,
        "explanation": explanation,
        "interval_status": interval_status,
        "deal_score": deal_score,
        "asking_rate_sqft": round(asking_rate_sqft, 0),
        "fair_rate_sqft": round(fair_rate_sqft, 0),
        "rate_delta_sqft": round(asking_rate_sqft - fair_rate_sqft, 0),
    }
