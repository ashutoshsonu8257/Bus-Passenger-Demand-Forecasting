"""
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "loaded_model" in data

def test_routes_endpoint():
    response = client.get("/routes")
    assert response.status_code == 200
    data = response.json()
    assert data["route_count"] >= 6
    assert "ROUTE_101" in data["routes"]

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model_comparison" in data
    assert "champion_model" in data
    assert "route_breakdown" in data
    assert "peak_offpeak_breakdown" in data

def test_predict_endpoint_success():
    payload = {
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
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert "capacity_decision_support" in data
    assert "explainability" in data
    assert data["forecast"]["predicted_passenger_count"] > 0

def test_batch_predict_endpoint():
    payload = {
        "trips": [
            {
                "route_id": "ROUTE_101",
                "departure_time": "08:00",
                "bus_capacity": 50,
                "advance_bookings_d1": 35
            },
            {
                "route_id": "ROUTE_103",
                "departure_time": "16:00",
                "bus_capacity": 50,
                "advance_bookings_d1": 42
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["forecasts"]) == 2

def test_redbus_ui_endpoint():
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "busPulse" in response.text
    for expected_key in (
        "predicted_passenger_count",
        "baseline_passenger_count",
        "prediction_interval_95pct",
        "occupancy_rate",
        "action_recommendation",
        "key_drivers",
        "WAPE_percent",
        "WAPE_%",
    ):
        assert expected_key in response.text
