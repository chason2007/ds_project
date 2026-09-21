# Indian Housing Price Prediction & Market Analysis

A machine learning pipeline and Streamlit dashboard for predicting residential real estate prices across major Indian cities using housing listings data.

The project evaluates Linear Regression, Random Forest, and XGBoost regressors, constructs 80% prediction intervals using quantile gradient boosting, provides feature attribution via SHAP values, and exports relational tables for Power BI analysis.

---

## Dataset

- **Source**: MachineHack / Kaggle Indian House Price Prediction Challenge (~29,451 listings across Indian urban centers).
- **Target Variable**: `Price_Lakhs` (listing price in Indian Rupees, formatted in Lakhs and Crores).
- **Features**:
  - `City` and `Locality`: Parsed from listing addresses across Bangalore, Mumbai, Pune, Noida, Chennai, Kolkata, Gurgaon, Hyderabad, and other metropolitan areas.
  - `BHK_NO`: Number of bedrooms (1 to 5+ BHK).
  - `BHK_OR_RK`: Room configuration (BHK apartment or RK studio).
  - `SQUARE_FT`: Carpet / super built-up area in square feet.
  - `POSTED_BY`: Listing creator category (Dealer / Broker, Owner, Builder).
  - `RERA`: Regulatory compliance approval status.
  - `READY_TO_MOVE` / `UNDER_CONSTRUCTION`: Construction and handover stage.
  - `RESALE`: Primary developer sale vs. secondary resale market.
  - `LATITUDE` & `LONGITUDE`: Geographic coordinates of property locations.

---

## Model Benchmark Results

Evaluated across **29,216 cleaned residential listings** (80/20 train/test split: 23,372 training, 5,844 test):

| Model | Train $R^2$ | Test $R^2$ | 5-Fold CV $R^2$ ($\mu \pm \sigma$) | RMSE (₹ Lakhs) | MAE (₹ Lakhs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | 0.4712 | 0.4649 | 0.4676 $\pm$ 0.0290 | ₹ 102.56 L | ₹ 49.43 L |
| Random Forest Regressor | 0.9239 | 0.7475 | 0.7601 $\pm$ 0.0403 | ₹ 70.45 L | ₹ 26.95 L |
| **XGBoost Regressor** | **0.9406** | **0.7569** | **0.7751 $\pm$ 0.0264** | **₹ 69.13 L** | **₹ 25.10 L** |

### Prediction Intervals (Quantile Regression)
- Implemented using dual `HistGradientBoostingRegressor` models at the 10th ($\alpha = 0.10$) and 90th ($\alpha = 0.90$) percentiles.
- **Empirical Coverage**: **77.84%** of out-of-sample test listings fall within the predicted 80% interval bounds.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── run.py                      # Pipeline CLI entrypoint
├── app.py                      # Streamlit interactive dashboard
├── data/
│   ├── train.csv               # Raw Indian housing dataset (29,451 rows)
│   └── data_description.txt
├── src/
│   ├── config.py               # Paths, constants, and city lists
│   ├── data_loader.py          # Data ingestion and verification
│   ├── preprocessing.py        # Address parsing and outlier filtering
│   ├── feature_engineering.py  # Ratio metrics and layout indicators
│   ├── eda.py                  # Exploratory diagnostic charts
│   ├── dataset.py              # Feature matrix preparation and splits
│   ├── train.py                # Model training, CV, and quantile intervals
│   ├── explain.py              # Feature importance and SHAP analysis
│   ├── valuation_index.py      # City x BHK aggregation and test predictions
│   ├── predict_service.py      # Inference service for real-time valuation
│   └── main.py                 # Pipeline execution runner
└── outputs/
    ├── charts/                 # Saved diagnostic visualizations (PNG)
    ├── models/                 # Saved model artifacts (.joblib)
    └── powerbi_data/           # Exported CSV tables for Power BI / Tableau
```

---

## Installation & Setup

### 1. Requirements
- Python 3.10+
- Recommended: virtual environment (`venv` or `conda`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Streamlit Web Dashboard
Launch the interactive web interface:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The dashboard includes:
- **Price Estimator**: Real-time property valuation, 80% interval bounds, and city comparisons.
- **City Price Index**: Benchmark valuation index by city and BHK tier.
- **Mortgage Calculator**: Estimated monthly EMI and rental yields.
- **Model Evaluation**: Benchmark comparison charts, actual vs. predicted plots, and residual diagnostics.
- **Feature Importance**: Top 20 feature importances and SHAP attribution plots.
- **Data & Exports**: Direct CSV downloads for Power BI / Tableau.

### Run Full Batch Pipeline
To re-run data processing, model training, and export generation:
```bash
python run.py
```
*(Or `python src/main.py`)*

### Run Individual Modules
Modules inside `src/` can also be executed independently:
```bash
# Data loading check
python src/data_loader.py

# Preprocessing & locality extraction
python src/preprocessing.py

# Feature engineering
python src/feature_engineering.py

# EDA figure generation
python src/eda.py

# Model training and benchmark
python src/train.py

# Feature importance & SHAP plots
python src/explain.py

# Valuation index computation
python src/valuation_index.py
```

---

## Power BI / Tableau Exports (`outputs/powerbi_data/`)

The pipeline generates five structured CSV files ready for import into business intelligence tools:

| File Name | Rows | Columns | Description |
| :--- | :--- | :--- | :--- |
| `predictions.csv` | 5,844 | 17 | Test set predictions with actual price, predicted price, 80% lower/upper bounds, and error metrics. |
| `valuation_index.csv` | 299 | 8 | Aggregated city and BHK index table with average and median rates per square foot. |
| `feature_importance.csv` | 20 | 3 | Top 20 valuation drivers ranked by relative importance (gain). |
| `model_metrics.csv` | 3 | 7 | Benchmark evaluation metrics ($R^2$, CV mean/std, RMSE, MAE) for all evaluated models. |
| `cleaned_engineered_dataset.csv` | 29,216 | 24 | Cleaned dataset including engineered features, location groupings, and calculated rates. |
