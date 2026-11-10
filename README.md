# Bus Passenger-Demand Forecasting: Decision-Support Solution
**Algions AI Labs Private Limited | AI & Machine Learning Advanced Assessment**

![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![License](https://img.shields.io/badge/License-CC--BY--4.0-green.svg)
![Tests](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)
![Model](https://img.shields.io/badge/Champion%20Model-LightGBM-orange.svg)
![WAPE](https://img.shields.io/badge/Test%20WAPE-8.89%25-success.svg)

---

## 1. Project Overview & Business Problem
In intercity and commuter bus booking, scheduling empty buses wastes significant capital in fuel and crew costs, while underestimating demand leads to denied boarding and customer churn. 

This solution is a **decision-support forecasting engine** that predicts passenger ridership for every scheduled trip departure $(Route, Departure\_Datetime)$ at the **D-1 18:00 cut-off (24 hours prior to departure)**. It provides operations dispatchers with:
- **Real-time passenger demand forecasts** and 95% confidence intervals.
- **Comparison against historical route/time heuristics**.
- **Actionable capacity-aware status flags** (`OPTIMAL`, `NEAR_CAPACITY`, `CRITICAL_OVERFLOW`, `LOW_DEMAND_CONSIDER_MERGE`).
- **Feature attribution and explicit operational limitations** (human-in-the-loop oversight).

---

## 2. Benchmark Evaluation Summary (Held-out Test Set)

Evaluated across **2,614 strictly held-out forward trips** (representing an unseen 2-month winter operational period):

| Model Architecture | Test WAPE (%) | Test MAE (Passengers) | Test RMSE | Test $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Historical Route/Time Baseline** | 27.64% | 8.188 | 10.051 | 0.2450 |
| **Ridge Regression** | 9.40% | 2.783 | 3.613 | 0.9025 |
| **Random Forest Regressor** | 8.93% | 2.644 | 3.456 | 0.9108 |
| **LightGBM Regressor (Champion)** | **8.89%** | **2.634** | **3.458** | **0.9107** |

> **Key Result:** The champion LightGBM model achieves a **67.8% relative error reduction** over the transit heuristic baseline, reducing average passenger error to **2.63 passengers per trip**.

---

## 3. Repository Directory Structure
```
bus-passenger-demand-forecasting/
├── app/
│   ├── api.py                    # Production FastAPI REST API (/predict, /batch, /routes, /metrics)
│   └── dashboard.py              # Interactive Streamlit Decision-Support Dashboard
├── artifacts/
│   ├── best_model.joblib         # Serialized champion LightGBM model
│   ├── baseline_model.joblib     # Serialized historical route/time baseline model
│   ├── preprocessor.joblib       # Serialized feature engineering pipeline
│   ├── metrics_summary.json      # Test evaluation metrics & improvement summary
│   ├── route_metrics.csv         # Stratified per-route error metrics
│   └── peak_offpeak_metrics.csv  # Stratified rush-hour vs off-peak error metrics
├── data/
│   ├── raw/
│   │   └── bus_trips_historical.csv  # 17,425 trip-level observations (14 months)
│   └── processed/
│       ├── train.csv             # Chronological train split (12,197 records, 70%)
│       ├── val.csv               # Chronological validation split (2,614 records, 15%)
│       └── test.csv              # Held-out forward test split (2,614 records, 15%)
├── notebooks/
│   ├── eda_analysis.py           # Comprehensive EDA script generating profiling plots
│   └── exploratory_data_analysis.ipynb # Interactive Jupyter analysis notebook
├── reports/
│   ├── TECHNICAL_REPORT.md       # Full formal technical report
│   └── figures/                  # Publication-quality evaluation and EDA charts
│       ├── model_comparison_bar.png
│       ├── actual_vs_predicted_scatter.png
│       ├── residuals_distribution.png
│       ├── feature_importance.png
│       └── eda_route_distribution.png
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, random seed (42), route metadata, capacity thresholds
│   ├── data_generator.py         # Realistic synthetic bus trip data generator
│   ├── data_loader.py            # Data quality validation & chronological splitter
│   ├── preprocessing.py          # Cyclical trigonometric & categorical transformers
│   ├── baseline.py               # Historical Route x Time Slot baseline implementation
│   ├── models.py                 # Model factory (Ridge, Random Forest, LightGBM)
│   ├── evaluate.py               # WAPE, MAE, RMSE, R2, and stratified breakdowns
│   ├── train.py                  # End-to-end reproducible training pipeline
│   ├── inference.py              # Real artifact-backed inference service
│   └── explain.py                # Local attribution, uncertainty bounds, limitations
├── tests/
│   ├── test_api.py               # FastAPI test client integration tests
│   ├── test_baseline.py          # Baseline model lookup tests
│   ├── test_data_generation.py   # Dataset schema, non-negativity, and lag safety tests
│   ├── test_inference.py         # Dynamic inference, non-hardcoded output tests
│   └── test_preprocessing.py     # Cyclical transforms & missing value fallback tests
├── requirements.txt              # Pinned Python package dependencies
├── .env.example                  # Environment configuration template
├── .gitignore                    # Version control exclusions
└── README.md                     # Project documentation & execution guide
```

---

## 4. Setup & Quickstart Guide

### 4.1 Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 4.2 Installation
Clone or navigate to the project repository:
```bash
cd bus-passenger-demand-forecasting

# Create and activate a Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
# source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. End-to-End Reproducibility Guide

Run the complete pipeline from scratch using the following commands:

### Step 1: Generate Dataset & Chronological Splits
```bash
# Generate 17,425 trip-level records (14 months across 6 route corridors)
python -m src.data_generator

# Perform data validation and strict chronological 70/15/15 train/val/test split
python -m src.data_loader
```

### Step 2: Run Exploratory Data Analysis
```bash
# Generates EDA figures in reports/figures/
python -m notebooks.eda_analysis
```

### Step 3: Train and Evaluate All Models
```bash
# Trains Baseline, Ridge, Random Forest, and LightGBM;
# Evaluates on held-out test split, exports metrics and visualizations
python -m src.train
```

### Step 4: Run Test Suite
```bash
# Executes 16 unit and integration tests across data, preprocessing, models, and API
pytest tests/ -v
```

---

## 6. Launching Inference Interfaces

### Interface A: Interactive Streamlit Decision-Support Dashboard
```bash
streamlit run app/dashboard.py
```
Opens in your browser at `http://localhost:8501`. Features include:
- Interactive Route, Date, Time, Weather, and Advance Bookings selectors.
- Real-time demand forecast, 95% confidence interval, and capacity occupancy gauge.
- Actionable Dispatch Decision Banner (`OPTIMAL`, `NEAR_CAPACITY`, `CRITICAL_OVERFLOW`, `LOW_DEMAND`).
- "What-If" Sensitivity Simulator for adverse weather storms and event rushes.
- Full model benchmark comparisons and stratified error breakdowns.

### Interface B: Production FastAPI REST Service
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI available at: `http://localhost:8000/docs`

#### Sample Prediction Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "route_id": "ROUTE_101",
       "departure_time": "08:00",
       "departure_hour": 8,
       "departure_minute": 0,
       "day_of_week": 1,
       "month": 4,
       "bus_capacity": 50,
       "advance_bookings_d1": 36,
       "weather_condition": "Clear",
       "special_event": "None"
     }'
```

#### Real API Response Output:
```json
{
  "trip_metadata": {
    "route_id": "ROUTE_101",
    "route_name": "Metro Downtown - Tech Park Commuter",
    "departure_time": "08:00",
    "bus_type": "Standard_50",
    "bus_capacity": 50,
    "advance_bookings_d1": 36,
    "weather": "Clear",
    "special_event": "None"
  },
  "forecast": {
    "model_name": "LightGBM",
    "predicted_passenger_count": 46.1,
    "baseline_passenger_count": 36.3,
    "prediction_interval_95pct": {
      "lower": 39.3,
      "upper": 52.9,
      "margin_error": 6.8
    },
    "expected_seat_occupancy": "92.2%"
  },
  "capacity_decision_support": {
    "predicted_demand": 46.1,
    "bus_capacity": 50,
    "occupancy_rate": 0.922,
    "occupancy_percentage": "92.2%",
    "status_code": "NEAR_CAPACITY_MONITOR",
    "action_recommendation": "High passenger load. Close promotional discounts and verify vehicle readiness.",
    "severity": "WARNING",
    "excess_passengers": 0.0,
    "available_empty_seats": 4
  },
  "explainability": {
    "key_drivers": [
      {
        "factor": "D-1 Advance Bookings",
        "impact": "Primary Baseline Driver",
        "direction": "Positive",
        "detail": "36 confirmed tickets booked 24h before departure (~75-85% final share)."
      },
      {
        "factor": "Rush-Hour Commute Peak",
        "impact": "High Positive Surge",
        "direction": "Positive",
        "detail": "Departure at 08:00 aligns with high-volume commuter rush window."
      }
    ],
    "limitations_and_failure_modes": [
      "Cold-Start Route: Model relies heavily on historical slot averages and advance bookings; new routes lack 7d/14d lag data.",
      "Unplanned Service Disruptions: Sudden road closures, transit strikes, or spontaneous weather events occurring <24h before departure are outside the model horizon.",
      "Extreme Outlier Events: Multi-day severe blizzard or city-wide emergency will distort normal booking conversion rates.",
      "Human Oversight Rule: This tool provides decision support. Fleet dispatchers must manually verify fleet and driver availability before cancelling or adding runs."
    ]
  }
}
```

---

## 7. Acceptance Test Verification Matrix

| Acceptance Criteria | Status | Verification Detail |
| :--- | :---: | :--- |
| **1. Reproducible Split & Training** | **PASSED** | Fixed seed (42), chronological boundary split (70/15/15), repeatable with `python -m src.train`. |
| **2. Baseline vs Candidate Comparison** | **PASSED** | Historical Baseline (27.64% WAPE) compared against Ridge (9.40%), Random Forest (8.93%), and LightGBM (8.89%). |
| **3. Real Non-Hardcoded Prediction** | **PASSED** | Loads serialized `best_model.joblib`. Changing inputs dynamically alters predictions (verified via `test_inference_is_non_hardcoded_and_dynamic`). |
| **4. Explainability & Uncertainty** | **PASSED** | Surfaced via local factor attribution, 95% confidence intervals, and capacity risk warnings. |
| **5. Zero Target Leakage or PII** | **PASSED** | Lags calculated strictly on past trips $\ge 24$ hours prior; zero personal identifiable data used. |

---

## 8. Responsible AI & Operational Limitations
- **Human-in-the-Loop:** This system provides advisory decision support. It does not automate coach cancellations without fleet dispatcher sign-off.
- **Cold-Start Handling:** For newly created routes, the model falls back to Route Type medians and provides wider uncertainty intervals.
- **Acute Disruptions:** Unheralded disruptions (flash transit strikes, sudden highway accidents $<24$ hours out) cannot be inferred from D-1 bookings; dispatchers must apply manual contingency overrides.
- **Privacy:** Operates strictly on aggregate trip and ticket volumes. No passenger PII is collected or stored.

---
*Created for ALGIONS AI LABS PRIVATE LIMITED — Advanced Individual Project Assessment*
