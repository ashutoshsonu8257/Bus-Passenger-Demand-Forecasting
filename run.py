"""
Runner script for Bus Passenger-Demand Forecasting.
Executes sample real-time inference and displays decision-support recommendations.
"""

import sys
import json
from pathlib import Path

# Set up project path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.inference import get_forecaster
from src.config import ROUTE_CONFIG

def main():
    print("=" * 70)
    print(" 🚌 BUS PASSENGER-DEMAND FORECASTING — DECISION SUPPORT ENGINE")
    print("=" * 70)

    forecaster = get_forecaster()
    print(f"\n[Artifacts Loaded] Model: {forecaster.champion_name} | RMSE: {forecaster.test_rmse}")

    sample_trip = {
        "route_id": "ROUTE_101",
        "route_name": "Metro Downtown - Tech Park Commuter",
        "departure_time": "08:00",
        "departure_hour": 8,
        "departure_minute": 0,
        "day_of_week": 1, # Tuesday
        "month": 4,
        "bus_capacity": 50,
        "advance_bookings_d1": 38,
        "weather_condition": "Clear",
        "special_event": "None",
        "days_to_holiday": 18
    }

    print("\n--- Scheduled Trip Query ---")
    for k, v in sample_trip.items():
        print(f"  {k}: {v}")

    result = forecaster.predict(sample_trip)
    f = result["forecast"]
    c = result["capacity_decision_support"]
    e = result["explainability"]

    print("\n" + "=" * 70)
    print(f" FORECAST RESULTS (Model: {f['model_name']})")
    print("=" * 70)
    print(f"  ► Predicted Passenger Count : {f['predicted_passenger_count']} passengers")
    print(f"  ► Historical Slot Baseline  : {f['baseline_passenger_count']} passengers")
    print(f"  ► 95% Confidence Interval   : [{f['prediction_interval_95pct']['lower']} – {f['prediction_interval_95pct']['upper']}] passengers (margin: ±{f['prediction_interval_95pct']['margin_error']})")
    print(f"  ► Expected Seat Occupancy   : {f['expected_seat_occupancy']} (Capacity: {c['bus_capacity']} seats)")
    print(f"  ► Available Empty Seats     : {c['available_empty_seats']} seats")
    print(f"  ► Operational Status Flag   : [{c['severity']}] {c['status_code']}")
    print(f"  ► Action Recommendation     : {c['action_recommendation']}")

    print("\n--- Key Prediction Drivers ---")
    for d in e["key_drivers"]:
        print(f"  • {d['factor']} ({d['direction']}): {d['detail']}")

    print("\n--- System Limitations & Oversight ---")
    for lim in e["limitations_and_failure_modes"][:2]:
        print(f"  ⚠ {lim}")

    print("\n" + "=" * 70)
    print(" SERVICES AVAILABLE TO LAUNCH:")
    print("  1. Interactive Web Dashboard : streamlit run app/dashboard.py")
    print("  2. FastAPI REST Endpoint     : uvicorn app.api:app --reload --port 8000")
    print("=" * 70)

if __name__ == "__main__":
    main()
