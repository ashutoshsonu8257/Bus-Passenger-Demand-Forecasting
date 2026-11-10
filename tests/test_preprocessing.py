"""
Unit tests for BusDemandPreprocessor and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import BusDemandPreprocessor

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame([
        {
            "route_id": "ROUTE_101",
            "route_type": "Commuter",
            "departure_datetime": "2025-05-10 08:00:00",
            "departure_hour": 8,
            "departure_minute": 0,
            "day_of_week": 5,
            "month": 5,
            "is_weekend": 1,
            "is_holiday": 0,
            "is_peak_hour": 1,
            "advance_bookings_d1": 30,
            "route_distance_km": 42.0,
            "scheduled_duration_min": 55,
            "standard_fare": 12.0,
            "bus_capacity": 50,
            "days_to_holiday": 16,
            "weather_impact_score": 0.0,
            "lag_7d_demand": 35.0,
            "lag_14d_demand": 34.0,
            "rolling_mean_route_4w": 33.0,
            "weather_condition": "Clear",
            "special_event": "None",
            "bus_type": "Standard_City"
        },
        {
            "route_id": "ROUTE_103",
            "route_type": "Intercity",
            "departure_datetime": "2025-05-10 16:30:00",
            "departure_hour": 16,
            "departure_minute": 30,
            "day_of_week": 5,
            "month": 5,
            "is_weekend": 1,
            "is_holiday": 0,
            "is_peak_hour": 0,
            "advance_bookings_d1": 42,
            "route_distance_km": 210.0,
            "scheduled_duration_min": 180,
            "standard_fare": 45.0,
            "bus_capacity": 50,
            "days_to_holiday": 16,
            "weather_impact_score": 0.2,
            "lag_7d_demand": 40.0,
            "lag_14d_demand": 38.0,
            "rolling_mean_route_4w": 39.0,
            "weather_condition": "Heavy_Rain",
            "special_event": "Concert",
            "bus_type": "Deluxe_Intercity"
        }
    ])

def test_preprocessor_fit_transform(sample_raw_data):
    prep = BusDemandPreprocessor()
    prep.fit(sample_raw_data)
    transformed = prep.transform(sample_raw_data)

    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 2
    # Check cyclical feature creation
    assert "hour_sin" in transformed.columns
    assert "hour_cos" in transformed.columns
    assert "dow_sin" in transformed.columns
    assert "dow_cos" in transformed.columns
    assert not transformed.isnull().any().any()

def test_preprocessor_handles_unseen_categories(sample_raw_data):
    prep = BusDemandPreprocessor()
    prep.fit(sample_raw_data)

    unseen_df = pd.DataFrame([{
        "route_id": "ROUTE_NEW_UNKNOWN",
        "route_type": "UnknownType",
        "departure_datetime": "2025-06-01 12:00:00",
        "departure_hour": 12,
        "departure_minute": 0,
        "day_of_week": 6,
        "month": 6,
        "is_weekend": 1,
        "advance_bookings_d1": 25,
        "bus_capacity": 40
    }])
    transformed = prep.transform(unseen_df)
    assert transformed.shape[1] == len(prep.get_feature_names())
    assert not transformed.isnull().any().any()
