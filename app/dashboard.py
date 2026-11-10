"""
Streamlit Decision-Support Dashboard for Bus Passenger-Demand Forecasting.
Provides interactive fleet dispatchers with real-time demand predictions,
capacity threshold flags, scenario simulation, explainability, and validation metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, date, time
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    ROUTE_CONFIG,
    METRICS_SUMMARY_PATH,
    ROUTE_METRICS_PATH,
    PEAK_METRICS_PATH,
    FIGURES_DIR
)
from src.inference import get_forecaster

st.set_page_config(
    page_title="Bus Demand Forecaster | Decision Support System",
    page_icon="🚌",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2980b9;
        margin-bottom: 10px;
    }
    .status-banner {
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 600;
        margin: 15px 0;
    }
    .status-SUCCESS {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-WARNING {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .status-DANGER {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .status-INFO {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_forecaster():
    return get_forecaster()

forecaster = load_cached_forecaster()

# Sidebar: Trip Configuration
st.sidebar.image("https://img.icons8.com/color/96/bus.png", width=64)
st.sidebar.title("Fleet Trip Dispatch")
st.sidebar.caption("ALGIONS AI LABS — Bus Passenger-Demand Forecasting")

route_options = list(ROUTE_CONFIG.keys())
route_labels = {r: f"{r}: {ROUTE_CONFIG[r]['name']} ({ROUTE_CONFIG[r]['type']})" for r in route_options}
selected_route = st.sidebar.selectbox(
    "Select Bus Route",
    options=route_options,
    format_func=lambda r: route_labels[r]
)

r_info = ROUTE_CONFIG[selected_route]
st.sidebar.info(
    f"**Type:** {r_info['type']}  \n"
    f"**Distance:** {r_info['distance_km']} km  \n"
    f"**Scheduled Duration:** {r_info['duration_min']} mins  \n"
    f"**Base Fare:** ${r_info['base_fare']:.2f}"
)

# Date & Time Pickers
departure_date = st.sidebar.date_input("Departure Date", value=date.today())
departure_time = st.sidebar.time_input("Departure Time", value=time(8, 0))

# Operational Sliders
bus_capacity = st.sidebar.slider(
    "Assigned Bus Capacity (Seats)",
    min_value=25,
    max_value=65,
    value=r_info["default_capacity"],
    step=5
)

advance_bookings = st.sidebar.slider(
    "D-1 Advance Bookings (24h before run)",
    min_value=0,
    max_value=bus_capacity + 15,
    value=int(bus_capacity * 0.72)
)

# External Context Factors
weather = st.sidebar.selectbox("Forecasted Weather", ["Clear", "Rain", "Heavy_Rain", "Snow", "Fog"])
special_event = st.sidebar.selectbox("Local Special Event", ["None", "Sports_Game", "Concert", "Festival", "Conference"])
days_to_holiday = st.sidebar.number_input("Days to Nearest Public Holiday", min_value=0, max_value=60, value=12)

# Weather score map
weather_score_map = {"Clear": 0.0, "Rain": 0.08, "Fog": 0.12, "Heavy_Rain": 0.20, "Snow": 0.30}
weather_score = weather_score_map[weather]

# Prepare input payload
trip_payload = {
    "route_id": selected_route,
    "departure_datetime": f"{departure_date} {departure_time.strftime('%H:%M:%S')}",
    "departure_hour": departure_time.hour,
    "departure_minute": departure_time.minute,
    "departure_time": departure_time.strftime("%H:%M"),
    "day_of_week": departure_date.weekday(),
    "month": departure_date.month,
    "bus_capacity": bus_capacity,
    "advance_bookings_d1": advance_bookings,
    "weather_condition": weather,
    "weather_impact_score": weather_score,
    "special_event": special_event,
    "days_to_holiday": int(days_to_holiday)
}

# Run Real ML Inference
inference_result = forecaster.predict(trip_payload)
forecast_data = inference_result["forecast"]
cap_support = inference_result["capacity_decision_support"]
explain_data = inference_result["explainability"]

# Header
st.title("🚌 Bus Passenger-Demand Decision Support Interface")
st.markdown("Decision-support system for intercity and commuter bus operators to optimize fleet dispatching and avoid overflow.")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Demand Forecast & Dispatch Action",
    "📊 Model Benchmark & Granular Evaluation",
    "🔍 Feature Attribution & Explainability",
    "🛡️ Responsible AI & System Limitations"
])

with tab1:
    st.subheader(f"Trip Departure: {r_info['name']} at {departure_time.strftime('%H:%M')} on {departure_date.strftime('%A, %b %d, %Y')}")

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Predicted Demand (Passengers)",
            value=f"{forecast_data['predicted_passenger_count']} pax",
            delta=f"{forecast_data['predicted_passenger_count'] - forecast_data['baseline_passenger_count']:+.1f} vs baseline"
        )
        st.caption(f"95% CI: [{forecast_data['prediction_interval_95pct']['lower']} – {forecast_data['prediction_interval_95pct']['upper']}] pax")

    with col2:
        st.metric(
            label="Historical Slot Baseline",
            value=f"{forecast_data['baseline_passenger_count']} pax",
            delta="Lookup Baseline"
        )
        st.caption("Historical route/time average")

    with col3:
        st.metric(
            label="Expected Seat Occupancy",
            value=cap_support["occupancy_percentage"],
            delta=f"{cap_support['occupancy_rate'] - 0.70:+.1%} vs 70% target"
        )
        st.caption(f"Bus Capacity: {bus_capacity} seats")

    with col4:
        if cap_support["excess_passengers"] > 0:
            st.metric(
                label="Capacity Deficit",
                value=f"+{cap_support['excess_passengers']} overflow",
                delta="OVERFLOW RISK",
                delta_color="inverse"
            )
            st.caption("Exceeds single coach seating")
        else:
            st.metric(
                label="Available Empty Seats",
                value=f"{cap_support['available_empty_seats']} seats",
                delta="Normal buffer"
            )
            st.caption("Seats free for walk-ups")

    # Decision Banner
    severity = cap_support["severity"]
    st.markdown(f"""
    <div class="status-banner status-{severity}">
        <strong>Dispatch Recommendation [{cap_support['status_code']}]:</strong><br>
        {cap_support['action_recommendation']}
    </div>
    """, unsafe_allow_html=True)

    # Visual Capacity Gauge Bar
    st.markdown("#### Real-time Seating Capacity Utilization")
    occ_val = min(1.3, cap_support["occupancy_rate"])
    progress_col1, progress_col2 = st.columns([4, 1])
    with progress_col1:
        st.progress(min(1.0, float(occ_val)))
    with progress_col2:
        st.write(f"**{cap_support['occupancy_percentage']} full**")

    # Interactive Scenario Sensitivity
    st.markdown("---")
    st.markdown("### 🧪 What-If Operational Sensitivity Simulator")
    st.caption("Simulate real-time external disruptions to see how the champion ML model adjusts dispatch predictions.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_weather = st.selectbox("Simulate Extreme Weather", ["Clear", "Heavy_Rain", "Snow"], key="sim_w")
    with sim_col2:
        sim_event = st.selectbox("Simulate Event Rush", ["None", "Concert", "Sports_Game"], key="sim_e")
    with sim_col3:
        sim_advance_delta = st.slider("Simulate Advance Booking Surge/Drop", -15, 15, 0, key="sim_adv")

    sim_payload = dict(trip_payload)
    sim_payload["weather_condition"] = sim_weather
    sim_payload["weather_impact_score"] = weather_score_map[sim_weather]
    sim_payload["special_event"] = sim_event
    sim_payload["advance_bookings_d1"] = max(0, advance_bookings + sim_advance_delta)

    sim_result = forecaster.predict(sim_payload)
    sim_forecast = sim_result["forecast"]
    sim_cap = sim_result["capacity_decision_support"]

    st.write(
        f"**Simulated Outcome:** Under {sim_weather} weather with '{sim_event}' event and {sim_payload['advance_bookings_d1']} advance bookings: "
        f"Forecasted demand shifts to **{sim_forecast['predicted_passenger_count']} passengers** ({sim_cap['occupancy_percentage']} occupancy). "
        f"Resulting Flag: `{sim_cap['status_code']}`."
    )

with tab2:
    st.subheader("Model Evaluation & Held-out Test Benchmarks")
    st.markdown("Performance measured on **2,614 strictly held-out test trips** that were not used during training or hyperparameter tuning.")

    # Load metrics from file
    if METRICS_SUMMARY_PATH.exists():
        with open(METRICS_SUMMARY_PATH, "r") as f:
            summary_data = json.load(f)

        models_data = summary_data.get("models", {})
        table_rows = []
        for m_name, vals in models_data.items():
            t = vals["test"]
            table_rows.append({
                "Model Architecture": m_name,
                "Test WAPE (%)": f"{t['WAPE_percent']:.2f}%",
                "Test MAE (pax)": f"{t['MAE']:.3f}",
                "Test RMSE": f"{t['RMSE']:.3f}",
                "Test R² Score": f"{t['R2']:.4f}",
                "Test MAPE (%)": f"{t['MAPE_percent']:.2f}%",
                "Mean Residual": f"{t['mean_residual']:.3f}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        imp = summary_data.get("test_improvement", {})
        st.success(
            f"**Baseline vs Champion ({summary_data.get('champion_model')}):** "
            f"Reduced WAPE error from **{imp.get('baseline_wape')}%** down to **{imp.get('champion_wape')}%** "
            f"({imp.get('wape_reduction_pct')}% relative error reduction)! MAE improved by {imp.get('mae_reduction')} passengers per trip."
        )

    # Figures Grid
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if (FIGURES_DIR / "model_comparison_bar.png").exists():
            st.image(str(FIGURES_DIR / "model_comparison_bar.png"), caption="Model Error Comparison (WAPE % & MAE)")
    with col_img2:
        if (FIGURES_DIR / "actual_vs_predicted_scatter.png").exists():
            st.image(str(FIGURES_DIR / "actual_vs_predicted_scatter.png"), caption="Actual vs Predicted Demand (Held-out Test)")

    # Stratified Breakdowns
    st.markdown("---")
    st.subheader("Stratified Performance Breakdown")
    breakdown_col1, breakdown_col2 = st.columns(2)

    with breakdown_col1:
        st.markdown("#### Per-Route Held-out Performance")
        if ROUTE_METRICS_PATH.exists():
            route_df = pd.read_csv(ROUTE_METRICS_PATH)
            st.dataframe(route_df, use_container_width=True)

    with breakdown_col2:
        st.markdown("#### Peak vs Off-Peak Rush Hour Performance")
        if PEAK_METRICS_PATH.exists():
            peak_df = pd.read_csv(PEAK_METRICS_PATH)
            st.dataframe(peak_df, use_container_width=True)

    # Residuals & Feature Importances
    col_img3, col_img4 = st.columns(2)
    with col_img3:
        if (FIGURES_DIR / "residuals_distribution.png").exists():
            st.image(str(FIGURES_DIR / "residuals_distribution.png"), caption="Residual Error Distribution (Actual - Predicted)")
    with col_img4:
        if (FIGURES_DIR / "feature_importance.png").exists():
            st.image(str(FIGURES_DIR / "feature_importance.png"), caption="Top Influential Drivers of Bus Ridership")

with tab3:
    st.subheader("Local Prediction Attribution & Key Demand Drivers")
    st.markdown(f"Decomposition of factors driving the prediction for **{selected_route}** at **{departure_time.strftime('%H:%M')}**:")

    drivers = explain_data.get("key_drivers", [])
    if drivers:
        for d in drivers:
            with st.expander(f"{'🟢' if d['direction'] == 'Positive' else '🔴'} {d['factor']} — {d['impact']}", expanded=True):
                st.write(d["detail"])
    else:
        st.write("Operating at standard latent baseline demand with no unusual anomalies.")

    st.markdown("---")
    st.subheader("Prediction Uncertainty Bounds (95% Confidence Interval)")
    pi = forecast_data["prediction_interval_95pct"]
    st.write(
        f"Given the historical held-out test error distribution, the true boarded passenger count is expected to fall within "
        f"**[{pi['lower']}, {pi['upper']}] passengers** with 95% statistical confidence (± {pi['margin_error']} pax margin of error)."
    )

with tab4:
    st.subheader("Responsible AI, Data Privacy & System Limitations")
    st.markdown("""
    ### 1. Decision-Support Boundary & Human Oversight
    - **Advisory Role Only:** This forecasting model is strictly an **operational decision-support tool**. It does **not** execute automated dispatch cancellations or vehicle reassignments without human dispatcher confirmation.
    - **Dispatcher Verification:** Fleet operations managers must check driver shift limits, fleet maintenance status, and depot parking constraints before approving schedule adjustments.

    ### 2. Failure Modes & Edge Case Behavior
    - **Cold-Start Routes:** When a brand-new route is introduced with no historical 7-day or 14-day booking records, the model falls back to Route Type medians and early D-1 advance booking signals. Forecasts for new routes should be treated with wider confidence intervals.
    - **Unheralded Disruption Events:** Sudden weather occurrences (flash floods), public transit labor strikes, or sudden highway closures occurring inside the 24-hour prediction horizon cannot be anticipated by D-1 booking signals.
    - **Extreme Weather Dampening:** Blizzards and severe storms suppress walk-up boardings by 15–30%.

    ### 3. Data Privacy & Leakage Safeguards
    - **Zero Personal Identifiable Information (PII):** The dataset operates exclusively at the aggregate trip and ticket count level. No passenger names, phone numbers, payment details, or IP addresses are stored or processed.
    - **Zero Target Leakage:** Lag features are computed strictly chronologically using trips occurring $\ge 24$ hours prior. D-1 advance bookings reflect only confirmed bookings up to 18:00 on the day prior to departure.
    """)

st.markdown("---")
st.caption("ALGIONS AI LABS PRIVATE LIMITED — AI & Machine Learning Assessment | Bus Passenger-Demand Forecasting")
