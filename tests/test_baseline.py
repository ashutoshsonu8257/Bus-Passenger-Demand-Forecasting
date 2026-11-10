"""
Unit tests for HistoricalRouteTimeBaseline benchmark model.
"""

import pytest
import pandas as pd
import numpy as np
from src.baseline import HistoricalRouteTimeBaseline

def test_baseline_fit_and_predict():
    train_df = pd.DataFrame([
        {"route_id": "R1", "departure_time": "08:00", "passenger_count": 40},
        {"route_id": "R1", "departure_time": "08:00", "passenger_count": 44},
        {"route_id": "R1", "departure_time": "12:00", "passenger_count": 20},
        {"route_id": "R2", "departure_time": "08:00", "passenger_count": 30}
    ])
    baseline = HistoricalRouteTimeBaseline(target_column="passenger_count")
    baseline.fit(train_df)

    test_df = pd.DataFrame([
        {"route_id": "R1", "departure_time": "08:00"},
        {"route_id": "R1", "departure_time": "12:00"},
        {"route_id": "R2", "departure_time": "08:00"},
        {"route_id": "R1", "departure_time": "18:00"}, # Unknown slot for R1 -> falls back to R1 mean
        {"route_id": "R3", "departure_time": "08:00"}  # Unknown route -> falls back to global mean
    ])

    preds = baseline.predict(test_df)
    assert len(preds) == 5
    assert preds[0] == 42.0 # (40+44)/2
    assert preds[1] == 20.0
    assert preds[2] == 30.0
    assert preds[3] == (40 + 44 + 20) / 3 # Route 1 mean = 34.66
    assert preds[4] == (40 + 44 + 20 + 30) / 4 # Global mean = 33.5
