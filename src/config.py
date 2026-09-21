"""
Pipeline Configuration module for All-India Real Estate Valuation and Price Prediction Engine.
Defines system paths, modeling constants, column schemas, and city groupings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "train.csv"
DATA_DESCRIPTION_PATH = DATA_DIR / "data_description.txt"

OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
POWERBI_DIR = OUTPUTS_DIR / "powerbi_data"
MODELS_DIR = OUTPUTS_DIR / "models"

# Ensure all output directories exist
for path in [DATA_DIR, OUTPUTS_DIR, CHARTS_DIR, POWERBI_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Data URL for automatic downloading if missing locally
URL_TRAIN = "https://raw.githubusercontent.com/Ankur789453/House-Price-Prediction/main/train.csv"

# Modeling Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Price_Lakhs"
RAW_TARGET = "TARGET(PRICE_IN_LACS)"

# Feature groups for modeling
CATEGORICAL_COLS = ["POSTED_BY", "BHK_OR_RK", "City_Grouped"]
BINARY_COLS = ["UNDER_CONSTRUCTION", "RERA", "READY_TO_MOVE", "RESALE"]
NUMERICAL_COLS = ["SQUARE_FT", "BHK_NO", "LATITUDE", "LONGITUDE", "Log_Sqft", "BHK_Sqft_Ratio"]

# Outlier filtering limits for realistic residential property estimation
MIN_SQUARE_FT = 150.0
MAX_SQUARE_FT = 12000.0
MIN_PRICE_LAKHS = 3.0       # ₹ 3 Lakhs minimum
MAX_PRICE_LAKHS = 4000.0    # ₹ 40 Crore maximum
MAX_BHK = 8

# Top Indian Cities for categorical grouping (others assigned to 'Other_City')
TOP_CITIES = [
    "Bangalore",
    "Mumbai",
    "Pune",
    "Noida",
    "Kolkata",
    "Chennai",
    "Ghaziabad",
    "Jaipur",
    "Chandigarh",
    "Faridabad",
    "Mohali",
    "Gurgaon",
    "Vadodara",
    "Surat",
    "Nagpur",
    "Lucknow",
    "Indore",
    "Bhubaneswar",
    "Hyderabad",
    "Kochi",
    "Lalitpur",
    "Maharashtra",
]

# Metadata columns preserved for Power BI reporting and slicing (not used directly as raw model features)
META_COLS = [
    "Property_Id",
    "City",
    "Locality",
    "BHK_NO",
    "BHK_OR_RK",
    "SQUARE_FT",
    "POSTED_BY",
    "RERA",
    "READY_TO_MOVE",
    "RESALE",
]
