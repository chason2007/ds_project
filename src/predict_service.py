"""
Real-time Property Valuation Service for All-India Real Estate Engine.
Accepts user property attributes, aligns with model schema, and computes
predicted price (₹ Lakhs & ₹ Crores), 80% interval bounds, and ₹/sqft rate.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from src.config import MODELS_DIR, TOP_CITIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Approximate coordinates for Indian metro city centers (latitude, longitude)
CITY_COORDINATES = {
    "Bangalore": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Noida": (28.5355, 77.3910),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Ghaziabad": (28.6692, 77.4538),
    "Jaipur": (26.9124, 75.7873),
    "Chandigarh": (30.7333, 76.7794),
    "Faridabad": (28.4089, 77.3178),
    "Mohali": (30.7046, 76.7179),
    "Gurgaon": (28.4595, 77.0266),
    "Vadodara": (22.3072, 73.1812),
    "Surat": (21.1702, 72.8311),
    "Nagpur": (21.1458, 79.0882),
    "Lucknow": (26.8467, 80.9462),
    "Indore": (22.7196, 75.8577),
    "Bhubaneswar": (20.2961, 85.8245),
    "Hyderabad": (17.3850, 78.4867),
    "Kochi": (9.9312, 76.2673),
    "Lalitpur": (24.6883, 78.4124),
    "Maharashtra": (19.7515, 75.7139),
}


class IndianValuationService:
    """
    Cached model inference service for property valuation.
    """
    _instance: Optional["IndianValuationService"] = None

    def __init__(self):
        self.model = None
        self.q_lower = None
        self.q_upper = None
        self.feature_columns = None
        self.load_models()

    def load_models(self) -> None:
        """
        Loads serialized joblib models into memory.
        """
        try:
            self.model = joblib.load(MODELS_DIR / "best_model.joblib")
            self.q_lower = joblib.load(MODELS_DIR / "quantile_lower.joblib")
            self.q_upper = joblib.load(MODELS_DIR / "quantile_upper.joblib")
            self.feature_columns = joblib.load(MODELS_DIR / "model_feature_columns.joblib")
            logger.info("Successfully loaded Indian valuation models into memory.")
        except Exception as e:
            logger.warning(f"Models not fully loaded yet: {e}")

    @classmethod
    def get_instance(cls) -> "IndianValuationService":
        if cls._instance is None:
            cls._instance = IndianValuationService()
        return cls._instance

    def predict(
        self,
        city: str,
        bhk_no: int,
        square_ft: float,
        posted_by: str = "Owner",
        bhk_or_rk: str = "BHK",
        rera: int = 1,
        ready_to_move: int = 1,
        resale: int = 1,
        under_construction: int = 0,
    ) -> Dict[str, Any]:
        """
        Computes property price prediction, 80% interval bounds, and price per sqft.
        """
        if self.model is None or self.feature_columns is None:
            self.load_models()
            if self.model is None:
                raise RuntimeError("Models are not trained or available in outputs/models/.")

        # Assign coordinates based on city
        lat, lon = CITY_COORDINATES.get(city, (20.5937, 78.9629))

        # Engineer features
        log_sqft = float(np.log1p(square_ft))
        bhk_sqft_ratio = float(square_ft / max(bhk_no, 1))
        is_rk = 1 if bhk_or_rk.upper() == "RK" else 0
        is_dealer_or_builder = 1 if posted_by in ["Dealer", "Builder"] else 0
        is_uc_rera = 1 if (under_construction == 1 and rera == 1) else 0
        is_resale_ready = 1 if (resale == 1 and ready_to_move == 1) else 0

        city_grouped = city if city in TOP_CITIES else "Other_City"

        # Build feature dict
        raw_vals = {
            "SQUARE_FT": square_ft,
            "BHK_NO": bhk_no,
            "LATITUDE": lat,
            "LONGITUDE": lon,
            "Log_Sqft": log_sqft,
            "BHK_Sqft_Ratio": bhk_sqft_ratio,
            "UNDER_CONSTRUCTION": under_construction,
            "RERA": rera,
            "READY_TO_MOVE": ready_to_move,
            "RESALE": resale,
            "Is_RK": is_rk,
            "Is_Dealer_Or_Builder": is_dealer_or_builder,
            "Is_UC_RERA": is_uc_rera,
            "Is_Resale_Ready": is_resale_ready,
        }

        # Initialize input series with zeros for all trained features
        row = pd.Series(0.0, index=self.feature_columns, dtype="float32")

        # Fill direct numerical / binary features
        for k, v in raw_vals.items():
            if k in row.index:
                row[k] = v

        # Fill one-hot categorical flags
        posted_col = f"POSTED_BY_{posted_by}"
        if posted_col in row.index:
            row[posted_col] = 1.0

        layout_col = f"BHK_OR_RK_{bhk_or_rk.upper()}"
        if layout_col in row.index:
            row[layout_col] = 1.0

        city_col = f"City_Grouped_{city_grouped}"
        if city_col in row.index:
            row[city_col] = 1.0

        # Create 1-row DataFrame
        input_df = pd.DataFrame([row])

        # Inference
        pred_lakhs = float(np.maximum(1.0, self.model.predict(input_df)[0]))
        lower_lakhs = float(np.maximum(1.0, self.q_lower.predict(input_df)[0]))
        upper_lakhs = float(np.maximum(lower_lakhs, self.q_upper.predict(input_df)[0]))

        price_per_sqft = (pred_lakhs * 100000.0) / square_ft

        def format_inr(val_lakhs: float) -> str:
            if val_lakhs >= 100.0:
                crores = val_lakhs / 100.0
                return f"₹ {crores:.2f} Cr"
            return f"₹ {val_lakhs:.2f} Lakhs"

        return {
            "predicted_price_lakhs": round(pred_lakhs, 2),
            "lower_bound_lakhs": round(lower_lakhs, 2),
            "upper_bound_lakhs": round(upper_lakhs, 2),
            "price_per_sqft": round(price_per_sqft, 2),
            "formatted_price": format_inr(pred_lakhs),
            "formatted_lower": format_inr(lower_lakhs),
            "formatted_upper": format_inr(upper_lakhs),
            "city": city,
            "bhk_no": bhk_no,
            "square_ft": square_ft,
        }
