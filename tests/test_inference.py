"""
Integration tests for BusDemandForecaster inference engine.
Verifies real artifact loading, non-hardcoded predictions, and capacity flags.
"""

import pytest
from src.inference import get_forecaster
from src.config import BEST_MODEL_PATH

def test_inference_loads_real_model():
    forecaster = get_forecaster()
    assert forecaster.model is not None
    assert forecaster.preprocessor is not None
    assert forecaster.baseline is not None

def test_inference_prediction_structure_and_values():
    forecaster = get_forecaster()
    sample = {
        "route_id": "ROUTE_101",
        "departure_time": "08:00",
        "departure_hour": 8,
        "departure_minute": 0,
        "day_of_week": 1,
        "is_weekend": 0,
        "month": 3,
        "advance_bookings_d1": 35,
        "bus_capacity": 50,
        "weather_condition": "Clear",
        "special_event": "None"
    }
    result = forecaster.predict(sample)

    assert "forecast" in result
    assert "capacity_decision_support" in result
    assert "explainability" in result

    pred = result["forecast"]["predicted_passenger_count"]
    assert isinstance(pred, (int, float))
    assert pred > 0

    # Interval checks
    interval = result["forecast"]["prediction_interval_95pct"]
    assert interval["lower"] <= pred <= interval["upper"]

def test_inference_is_non_hardcoded_and_dynamic():
    forecaster = get_forecaster()
    sample_low = {
        "route_id": "ROUTE_101",
        "departure_time": "08:00",
        "departure_hour": 8,
        "departure_minute": 0,
        "day_of_week": 1,
        "advance_bookings_d1": 10,
        "bus_capacity": 50
    }
    sample_high = {
        "route_id": "ROUTE_101",
        "departure_time": "08:00",
        "departure_hour": 8,
        "departure_minute": 0,
        "day_of_week": 1,
        "advance_bookings_d1": 45,
        "bus_capacity": 50
    }

    res_low = forecaster.predict(sample_low)
    res_high = forecaster.predict(sample_high)

    pred_low = res_low["forecast"]["predicted_passenger_count"]
    pred_high = res_high["forecast"]["predicted_passenger_count"]

    # Model must output significantly different values
    assert pred_high > pred_low + 15

def test_capacity_overflow_flag():
    forecaster = get_forecaster()
    sample_overflow = {
        "route_id": "ROUTE_101",
        "departure_time": "08:00",
        "departure_hour": 8,
        "departure_minute": 0,
        "day_of_week": 1,
        "advance_bookings_d1": 55,
        "bus_capacity": 40  # Under-sized bus
    }
    result = forecaster.predict(sample_overflow)
    cap = result["capacity_decision_support"]
    assert cap["occupancy_rate"] > 1.0
    assert cap["severity"] == "DANGER"
    assert "CRITICAL_OVERFLOW" in cap["status_code"]
