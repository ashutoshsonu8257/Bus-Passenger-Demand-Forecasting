# Bus Passenger-Demand Forecasting — Technical Evaluation Report
**Algions AI Labs Private Limited | AI & Machine Learning Advanced Assessment**  
**Role:** AI & Machine Learning | **Domain:** Bus Booking & Fleet Capacity Optimization  
**Project:** Bus Passenger-Demand Forecasting  

---

## 1. Executive Summary & Problem Framing

### 1.1 Business Decision Context
In regional, intercity, and commuter bus operations, dispatch managers must balance two opposing operational costs:
1. **Under-capacity (Spillover & Lost Revenue):** Turning away paying passengers or leaving travelers stranded when passenger demand exceeds bus seating capacity ($>100\%$).
2. **Over-capacity (Deadhead / Fuel / Crew Inefficiency):** Operating full-sized 50-seat coaches on trips carrying fewer than 15 passengers ($<30\%$ occupancy), wasting diesel, driver hours, and vehicle wear.

To optimize fleet scheduling, operators implement a **cut-off at D-1 18:00 (24 hours prior to departure)**. At this decision gate, operations dispatchers must decide whether to:
- Dispatch the standard scheduled coach (e.g., 40–50 seats).
- Upgrade to a high-capacity or double-decker vehicle (55+ seats).
- Schedule a secondary overflow coach (duplicate departure).
- Consolidate / merge two low-demand departures into an adjacent slot.

### 1.2 ML Formulation
- **ML Problem Type:** Supervised Time-Series Tabular Regression with Operational Constraints.
- **Prediction Unit:** A scheduled trip departure $(Route\_ID, Departure\_Datetime)$.
- **Target Variable ($y$):** `passenger_count` (non-negative integer representing the actual boarded passengers on the trip). Seat occupancy percentage is derived as $\text{Occupancy} = \frac{\hat{y}}{\text{Capacity}} \times 100\%$.
- **Prediction Horizon:** 24 hours prior to scheduled departure time (D-1 at 18:00 local time).
- **Point in Time:** All input signals are strictly restricted to information available $\ge 24$ hours before departure.

---

## 2. Dataset Sourcing, Profiling & Leakage Safeguards

### 2.1 Dataset Provenance & Characteristics
A licensed synthetic dataset mirroring regional transit booking patterns was generated spanning **14 continuous calendar months** (January 1, 2025 to March 1, 2026).
- **Total Records:** 17,425 trip-level observations across 6 distinct route archetypes.
- **Minimum Threshold Requirement:** Met and exceeded (assessment requested $\ge 1,000$ records; 17,425 provided).
- **License / Attribution:** Open Synthetic Transit Ridership Dataset (Algions AI Labs ML Assessment Benchmark).

### 2.2 Route Archetypes
| Route ID | Name | Type | Distance (km) | Default Capacity | Base Fare | Demand Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ROUTE_101` | Metro Downtown - Tech Park | Commuter | 42 km | 50 seats | $12.00 | Heavy weekday rush hour peaks (07:00–09:30, 16:30–19:30); low weekend demand. |
| `ROUTE_102` | City Center - University Campus | University | 28 km | 45 seats | $8.00 | Midday academic peaks; seasonal drops during summer break (Jun–Aug) and winter recess. |
| `ROUTE_103` | Capital City - Harbor Port | Intercity | 210 km | 50 seats | $45.00 | Friday evening & Sunday afternoon surges; pre-holiday surges. |
| `ROUTE_104` | Central Hub - Airport Express | Airport | 55 km | 40 seats | $25.00 | Consistent early morning (05:00–08:00) and evening flight connection waves. |
| `ROUTE_105` | Metro Valley - Mountain Resort | Tourist | 165 km | 55 seats | $38.00 | Extreme weekend and summer peaks; severe dampening during winter storms. |
| `ROUTE_106` | Southern Suburb - Industrial District | Feeder | 35 km | 40 seats | $10.00 | Industrial shift change surges (06:45, 15:45, 16:45). |

### 2.3 Strict Target & Future Leakage Prevention
Data leakage is the single greatest failure mode in demand forecasting. The pipeline enforces three architectural guarantees:
1. **D-1 Booking Cut-Off:** Only tickets confirmed $\ge 24$ hours prior (`advance_bookings_d1`) are exposed to the model. Final walk-up bookings and same-day cancellations are never exposed.
2. **Lag Calculation Barrier:** Historical lag features (`lag_7d_demand`, `lag_14d_demand`, `rolling_mean_route_4w`) are calculated strictly on past trips occurring at least 24 hours prior.
3. **No Target Contamination:** No derived features contain target averages computed over the validation or test splits. Preprocessors and imputation statistics are fitted **exclusively on the training split**.

---

## 3. Feature Engineering Pipeline

The preprocessing pipeline (`BusDemandPreprocessor`) transforms 11 raw inputs into 47 engineered features:

1. **Cyclical Temporal Encodings:**
   To capture diurnal and annual periodicities without artificial boundary discontinuities (e.g. 23:59 to 00:00), trigonometric encodings are computed:
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
   $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
   $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

2. **Categorical Encodings:**
   One-hot encoding for `route_id`, `route_type`, `weather_condition`, `special_event`, and `bus_type` with robust handling for unknown / out-of-vocabulary categories at inference time.

3. **External Proxies:**
   - `weather_impact_score`: Numeric severity proxy (Clear = 0.0, Rain = 0.08, Heavy Rain = 0.20, Snow = 0.30).
   - `days_to_holiday`: Continuous distance in days to the nearest major public holiday.
   - `special_event`: Scheduled entertainment or sporting events at destination.

---

## 4. Chronological Validation Strategy

Random $k$-fold cross-validation is **invalid** for time-series forecasting because it allows future observations to predict past events (lookahead leakage).

We employ a strict **chronological train/validation/test split**:
- **Train Set (70%, 12,197 trips):** `2025-01-01 05:30:00` to `2025-10-25 13:00:00`
- **Validation Set (15%, 2,614 trips):** `2025-10-25 13:30:00` to `2025-12-28 08:00:00`
- **Held-out Test Set (15%, 2,614 trips):** `2025-12-28 08:45:00` to `2026-03-01 21:30:00`

```
Timeline: [============= Train (70%) =============][== Val (15%) ==][== Test (15%) ==]
Dates:    Jan 1, 2025               Oct 25, 2025   Dec 28, 2025      Mar 1, 2026
```
The held-out test set represents an unseen forward-looking 2-month winter operational period used exclusively for final unbiased reporting.

---

## 5. Model Architectures & Benchmark Comparison

We evaluated four distinct approaches ranging from domain heuristics to gradient-boosted ensembles:

1. **Historical Route $\times$ Time Slot Baseline:**
   The standard transit industry heuristic. Calculates the historical average demand for each $(Route, Departure\_Time)$ slot with hierarchical fallback to route average and global mean.
2. **Candidate 1 — Ridge Regression ($L_2$ Regularized Linear):**
   Interpretable linear model with regularized coefficients preventing collinearity overfitting.
3. **Candidate 2 — Random Forest Regressor:**
   Non-linear ensemble of 100 decision trees capturing feature threshold interactions.
4. **Candidate 3 — LightGBM Regressor (Champion):**
   State-of-the-art leaf-wise gradient boosted decision tree optimized for fast inference, handling both cyclical features and non-linear holiday surges.

### 5.1 Comprehensive Evaluation Metrics
- **WAPE (Weighted Absolute Percentage Error):**
  $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\%$$
- **MAE (Mean Absolute Error):** Average passenger discrepancy per departure.
- **RMSE (Root Mean Squared Error):** Penalizes large forecast misses (vital for capacity planning).
- **$R^2$ Score:** Proportion of demand variance explained by the model.

### 5.2 Held-Out Test Set Performance Table

| Model Architecture | Val WAPE (%) | Held-out Test WAPE (%) | Test MAE (pax) | Test RMSE | Test $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Route/Time Baseline** | 26.41% | 27.64% | 8.188 | 10.051 | 0.2450 |
| **Ridge Regression** | 9.40% | 9.40% | 2.783 | 3.613 | 0.9025 |
| **Random Forest Regressor** | 9.09% | 8.93% | 2.644 | 3.456 | 0.9108 |
| **LightGBM Regressor (Champion)** | **9.04%** | **8.89%** | **2.634** | **3.458** | **0.9107** |

### 5.3 Business Impact & Quantified Improvement
- **WAPE Error Reduction:** LightGBM achieved an **8.89% WAPE**, representing a **67.8% relative error reduction** compared to the historical baseline (27.64%).
- **MAE Reduction:** Average prediction error dropped from **8.19 passengers to 2.63 passengers**, giving dispatchers accurate single-seat level granularity.
- **Variance Explained:** $R^2$ improved from 0.2450 (baseline) to 0.9107 (champion model).

---

## 6. Granular Stratified Breakdowns

### 6.1 Per-Route Breakdown (Held-Out Test Set)

| Route ID | Route Type | Test Samples | Mean Actual | Mean Pred | MAE | RMSE | WAPE (%) | $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ROUTE_101` | Commuter | 573 | 34.6 | 34.7 | 3.143 | 4.057 | 9.08% | 0.9102 |
| `ROUTE_102` | University | 447 | 27.1 | 26.8 | 2.652 | 3.375 | 9.79% | 0.8686 |
| `ROUTE_103` | Intercity | 319 | 35.9 | 35.8 | 2.594 | 3.428 | **7.23%** | 0.8238 |
| `ROUTE_104` | Airport Express | 574 | 31.7 | 31.7 | 2.488 | 3.197 | **7.86%** | 0.8083 |
| `ROUTE_105` | Tourist Scenic | 319 | 17.9 | 18.1 | **1.647** | **2.121** | 9.19% | **0.9559** |
| `ROUTE_106` | Suburb Feeder | 382 | 26.5 | 26.4 | 2.924 | 3.856 | 11.02% | 0.8653 |

**Key Route Insights:**
- Long-haul Intercity (`ROUTE_103`) and Airport Express (`ROUTE_104`) exhibit the lowest WAPE errors (7.23% and 7.86%) because ticket reservations are planned well in advance.
- Suburban Feeder (`ROUTE_106`) exhibits slightly higher WAPE (11.02%) due to higher walk-up variability.

### 6.2 Peak vs. Off-Peak Rush Hour Breakdown

| Departure Window | Test Sample Count | Mean Actual | Mean Pred | MAE (pax) | RMSE | WAPE (%) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Peak Period (07–09 & 17–19)** | 1,530 | 29.8 | 29.8 | 2.730 | 3.552 | 9.16% | 0.9206 |
| **Off-Peak Period** | 1,084 | 29.4 | 29.3 | 2.498 | 3.321 | **8.51%** | 0.8880 |

Both peak and off-peak periods maintain sub-10% WAPE and sub-3-passenger MAE, confirming the model does not suffer from peak-hour distortion.

---

## 7. Residual & Error Analysis

Inspection of test residuals $(y - \hat{y})$ reveals:
- **Mean Residual:** $-0.021$ passengers (unbiased; zero systematic over/under-forecasting).
- **Residual Distribution:** Highly symmetric bell curve centered at zero with minimal kurtosis.
- **P90 Absolute Error:** $5.3$ passengers, confirming that 90% of all trip forecasts deviate by fewer than 5 passengers from actual boardings.

---

## 8. Capacity-Aware Decision-Support Framework

The system maps demand predictions directly into operational dispatch actions:

| Occupancy Range | Status Code | Severity | Recommended Dispatch Action |
| :--- | :--- | :--- | :--- |
| **$< 35\%$** | `LOW_DEMAND_CONSIDER_MERGE` | `INFO` | Consider downsizing vehicle or consolidating with adjacent departure to save fuel and crew costs. |
| **$35\% - 70\%$** | `OPTIMAL_CAPACITY` | `SUCCESS` | Standard scheduled dispatch. Capacity matches expected volume comfortably. |
| **$70\% - 85\%$** | `NEAR_CAPACITY_MONITOR` | `WARNING` | High passenger load. Close promotional discounts and verify vehicle readiness. |
| **$> 85\%$** | `CRITICAL_OVERFLOW_RISK_DEPLOY_BACKUP` | `DANGER` | Projected overflow. Deploy a secondary backup coach or reroute bookings to earlier/later service. |

### Prediction Uncertainty Interval
Every prediction includes a **95% Confidence Interval**:
$$\hat{y} \pm 1.96 \times \text{RMSE}_{\text{test}} \approx \hat{y} \pm 6.8 \text{ passengers}$$
This ensures dispatchers know the lower and upper bounds before committing rolling stock.

---

## 9. Responsible AI, Ethics & System Limitations

1. **Advisory Decision Support, Not Automated Actuation:**
   The model does not autonomously cancel trips or assign buses. Fleet managers remain in full control and must verify driver hour regulations (Hours of Service) and bus maintenance status.
2. **Cold-Start Routes:**
   For newly introduced routes lacking 7-day or 14-day history, the pipeline falls back to route-type medians and D-1 advance booking curves. Dispatchers are alerted with an expanded uncertainty interval.
3. **Unplanned Acute Disruptions:**
   Flash flooding, transit strikes, or sudden road closures occurring inside the 24-hour horizon cannot be detected by D-1 booking curves. Dispatchers must manually override forecasts during known emergencies.
4. **Data Privacy (Zero PII):**
   The system operates entirely on aggregate trip-level counts. No passenger names, credit cards, or personal data are ingested or stored.

---

## 10. Verification of Acceptance Tests

| Acceptance Test Criteria | Status | Evidence / Verification Method |
| :--- | :---: | :--- |
| 1. Reproduce dataset split and training run | **PASSED** | Repeatable with `python -m src.train` using `RANDOM_SEED = 42`. |
| 2. Compare baseline & candidate models on held-out test set | **PASSED** | Baseline (27.64% WAPE) vs LightGBM (8.89% WAPE) documented in report and artifacts. |
| 3. Submit real prediction through API/App (no hardcoded stubs) | **PASSED** | Validated via `tests/test_inference.py` and `tests/test_api.py`. Dynamic output verified. |
| 4. Explain single prediction, top inputs, and limitations | **PASSED** | Implemented in `src/explain.py` and surfaced in both API and Streamlit UI. |
| 5. Demonstrate zero target leakage or PII | **PASSED** | Chronological boundary enforcement verified; unit tests confirm lag safety. |

---
*Report Author: AI & Machine Learning Assessment Candidate*  
*Submission to: Algions AI Labs Private Limited*
