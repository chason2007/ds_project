# All-India Real Estate Valuation Index & Price Prediction Engine

An enterprise-grade, modular Python engine for automated real estate valuation, predictive pricing, metropolitan BHK indexing, and risk interval estimation using the **All-India House Price Dataset** (29,451 residential listings across India's premier urban centers).

The engine implements domain-driven address parsing, Indian real estate feature engineering (BHK, RERA compliance, construction stage, rate per sq.ft.), multi-model benchmarking (Linear Regression, Random Forest, XGBoost), 5-fold cross-validation, 80% prediction interval modeling via quantile gradient boosting, SHAP explainability, an interactive **Streamlit Web GUI**, and automated generation of **Power BI-ready analytics tables**.

---

## Key Features

- **Realistic Indian Real Estate Attributes**:
  - Currency: Prices in **₹ Lakhs** and formatted in **₹ Crores** (1 Lakh = ₹ 100,000 INR; 1 Crore = ₹ 10,000,000 INR).
  - Configurations: **1 BHK, 2 BHK, 3 BHK, 4 BHK, 5+ BHK**, and **RK (Studio)** layouts.
  - Spatial Metrics: Area in **Square Feet** (`SQUARE_FT`) and implied rate in **₹ / Sq. Ft.** (`Price_Per_SqFt`).
  - Regulatory & Trust: **RERA Approval status** (Real Estate Regulatory Authority certification).
  - Market Timing: **Ready to Move** vs. **Under Construction** flags.
  - Transaction Type: **Resale Market** vs. **New Developer Booking**.
  - Listing Source: **Owner**, **Dealer (Broker)**, or **Builder (Developer)**.
  - Geographic Coverage: **Bangalore, Mumbai, Pune, Noida, Kolkata, Chennai, Ghaziabad, Jaipur, Chandigarh, Gurgaon, Hyderabad, Kochi, Vadodara, Surat, etc.**

- **Advanced Feature Engineering**:
  - `Log_Sqft`: Logarithmic transformation of property area.
  - `BHK_Sqft_Ratio`: Space intensity (average sq.ft. per bedroom room).
  - `Is_RK`: Studio Room-Kitchen configuration indicator.
  - `Is_Dealer_Or_Builder`: Professional brokerage / developer listing flag.
  - `Is_UC_RERA`: High-trust development flag (Under Construction + RERA certified).
  - `Is_Resale_Ready`: Ready-to-occupy secondary resale property indicator.
  - `Price_Per_SqFt`: Implied rate per square foot in INR (strictly isolated from model inputs to prevent target leakage).

- **Multi-Model Benchmark & Cross-Validation**:
  - Baseline Ordinary Least Squares (Linear Regression)
  - Random Forest Regressor (150 trees, `max_depth=12`)
  - Extreme Gradient Boosting (`XGBRegressor`, 300 trees, `max_depth=6`, `learning_rate=0.08`)
  - Evaluated on Train $R^2$, Test $R^2$, 5-Fold Cross-Validation $R^2$ ($\mu \pm \sigma$), Root Mean Squared Error (RMSE in ₹ Lakhs), and Mean Absolute Error (MAE in ₹ Lakhs).

- **Quantile Prediction Intervals (Risk Estimation)**:
  - Dual `HistGradientBoostingRegressor` quantile models ($\alpha = 0.10$ and $\alpha = 0.90$) forming an asymmetric 80% prediction interval in ₹ Lakhs.
  - Achieves **77.84% empirical coverage** across out-of-sample test listings.

- **Model Explainability & Attribution**:
  - Top 20 relative valuation drivers exported to CSV and styled bar chart.
  - SHAP `TreeExplainer` summary plot evaluated across test observations.

- **Indian Metro Valuation Index**:
  - Aggregates transaction metrics by `City x BHK_NO`.
  - Normalizes valuation index to $100.00$ relative to national median rate per sq.ft.
  - Generates multi-series comparison chart for top 6 metro markets (Mumbai, Bangalore, Pune, Noida, Kolkata, Chennai).

- **Power BI Ingestion Hub**:
  - Automatically outputs 5 clean, non-empty, relational CSV tables directly into `outputs/powerbi_data/`.

---

## Project Structure

```
real-estate-pipeline/
├── README.md                   # Complete system documentation & data catalog
├── requirements.txt            # Pinned project dependencies
├── run.py                      # Thin CLI entrypoint to execute full batch pipeline
├── app.py                      # Streamlit interactive Web GUI application
├── data/
│   ├── train.csv               # Raw Indian housing dataset (29,451 rows, 12 columns)
│   └── data_description.txt    # Field and attribute definitions
├── src/
│   ├── __init__.py             # Package declaration
│   ├── config.py               # Paths, constants, city lists, outlier thresholds
│   ├── data_loader.py          # Auto-download, ingestion, and schema verification
│   ├── preprocessing.py        # Address parsing, city grouping, and outlier filtering
│   ├── feature_engineering.py  # BHK ratios, non-linear transforms, composite flags
│   ├── eda.py                  # Headless-safe visual diagnostics (4 charts)
│   ├── dataset.py              # Categorical encoding, metadata isolation, 80/20 split
│   ├── train.py                # Model benchmarking, 5-fold CV, quantile intervals
│   ├── explain.py              # Feature importances & SHAP TreeExplainer
│   ├── valuation_index.py      # City x BHK valuation index & test predictions
│   ├── predict_service.py      # Real-time inference service (₹ Lakhs & ₹ Crores)
│   └── main.py                 # End-to-end orchestration and execution summary
└── outputs/
    ├── charts/                 # Headless publication-quality PNG charts
    │   ├── actual_vs_predicted.png
    │   ├── city_price_summary.png
    │   ├── correlation_heatmap.png
    │   ├── feature_importance_top20.png
    │   ├── market_overview_panels.png
    │   ├── missingness_summary.png
    │   ├── model_comparison.png
    │   ├── residuals_plot.png
    │   ├── shap_summary.png
    │   ├── target_distribution.png
    │   └── valuation_index_top6.png
    ├── models/                 # Serialized joblib model artifacts
    │   ├── best_model.joblib
    │   ├── model_feature_columns.joblib
    │   ├── quantile_lower.joblib
    │   └── quantile_upper.joblib
    └── powerbi_data/           # 5 Curated CSV datasets for Power BI import
        ├── cleaned_engineered_dataset.csv
        ├── feature_importance.csv
        ├── model_metrics.csv
        ├── predictions.csv
        └── valuation_index.csv
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13, 3.14)
- Git (optional)

### 2. Environment Setup
Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Dependencies: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `joblib`.*

---

## How to Run

### 1. Launch Interactive Web GUI (Streamlit Dashboard)
To launch the interactive web application featuring the live Indian property price calculator, confidence interval bounds, city valuation index explorer, and model diagnostics:

```bash
streamlit run app.py
```
This opens the dashboard in your default browser at **`http://localhost:8501`**.

### 2. Run Headless Full End-to-End Pipeline
Execute the single thin entrypoint command from the project root to run the automated batch pipeline:

```bash
python run.py
```

*(Alternatively: `python src/main.py`)*

### 3. Run Modular Components Independently
Each module inside `src/` can also be executed independently:

```bash
# Verify data ingestion
python src/data_loader.py

# Test preprocessing and locality parsing
python src/preprocessing.py

# Test Indian real estate feature engineering
python src/feature_engineering.py

# Regenerate diagnostic EDA visualizations
python src/eda.py

# Test dataset encoding and train/test splitting
python src/dataset.py

# Benchmark models, 5-fold CV, and quantile intervals
python src/train.py

# Generate feature importance and SHAP summary plots
python src/explain.py

# Compute valuation index and test predictions
python src/valuation_index.py
```

---

## Model Benchmark Results

Evaluated across **29,216 cleaned residential listings** (23,372 training properties, 5,844 test properties):

| Model | Train $R^2$ | Test $R^2$ | 5-Fold CV $R^2$ ($\mu \pm \sigma$) | RMSE (₹ in Lakhs) | MAE (₹ in Lakhs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline OLS)** | 0.4712 | 0.4649 | 0.4676 $\pm$ 0.0290 | ₹ 102.56 L | ₹ 49.43 L |
| **Random Forest Regressor** | 0.9239 | 0.7475 | 0.7601 $\pm$ 0.0403 | ₹ 70.45 L | ₹ 26.95 L |
| **XGBoost Regressor (Champion)** | **0.9406** | **0.7569** | **0.7751 $\pm$ 0.0264** | **₹ 69.13 L** | **₹ 25.10 L** |

### 80% Prediction Interval Performance
- **Methodology**: Dual `HistGradientBoostingRegressor` models ($\alpha = 0.10$ and $\alpha = 0.90$).
- **Empirical Coverage on Test Set**: **77.84%** of out-of-sample properties fall within predicted bounds.

---

## Power BI Data Catalog (`outputs/powerbi_data/`)

All 5 generated CSV files are formatted for immediate import into Microsoft Power BI or Tableau:

| File Name | Rows | Columns | Primary Key / Grain | Description & Use in Power BI |
| :--- | :--- | :--- | :--- | :--- |
| **`predictions.csv`** | 5,844 | 17 | `Property_Id` (Test Property) | Holds out-of-sample test property predictions from XGBoost. Contains `ActualPrice`, `PredictedPrice`, `LowerBound_80`, `UpperBound_80` (all in ₹ Lakhs), `AbsoluteError`, `PercentageError`, and `WithinInterval_80`. Includes slicer attributes (`City`, `Locality`, `BHK_NO`, `SQUARE_FT`, `POSTED_BY`, `RERA`, `READY_TO_MOVE`, `RESALE`). |
| **`valuation_index.csv`** | 299 | 8 | `City` $\times$ `BHK_NO` | Indian housing valuation table aggregated at city-BHK level. Contains `AvgPrice_Lakhs`, `MedianPrice_Lakhs`, `AvgPricePerSqFt`, `MedianPricePerSqFt`, `PropertyCount`, and `ValuationIndex` (Base 100.0 = National Median ₹/SqFt). Used for city price sensitivity and BHK premium line charts. |
| **`feature_importance.csv`** | 20 | 3 | `Rank` (1 to 20) | Ranked list of the top 20 property valuation drivers extracted from XGBoost (`Feature`, `Importance`, `Rank`). Used in horizontal bar charts illustrating price drivers. |
| **`model_metrics.csv`** | 3 | 7 | `Model` | Performance summary across Linear Regression, Random Forest, and XGBoost reporting Train $R^2$, Test $R^2$, 5-Fold CV $R^2$ Mean/Std, RMSE (₹ Lakhs), and MAE (₹ Lakhs). Used in executive KPI cards. |
| **`cleaned_engineered_dataset.csv`** | 29,216 | 24 | `Property_Id` (Listing) | Master property inventory combining all cleaned features, engineered flags (`Is_RK`, `Is_UC_RERA`, `Is_Resale_Ready`), rate per sq.ft., and target prices in ₹ Lakhs. Serves as the primary fact table. |
