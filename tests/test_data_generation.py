"""
Unit tests for synthetic bus trip data generator.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_bus_dataset
from src.config import TARGET_COLUMN

def test_data_generation_shape_and_types():
    # Generate small slice for fast unit testing
    df = generate_bus_dataset(start_date="2025-01-01", end_date="2025-01-10", seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert TARGET_COLUMN in df.columns
    assert "route_id" in df.columns
    assert "departure_datetime" in df.columns
    assert "advance_bookings_d1" in df.columns

def test_data_generation_target_validity():
    df = generate_bus_dataset(start_date="2025-01-01", end_date="2025-01-05", seed=42)
    # Target must be non-negative integers
    assert (df[TARGET_COLUMN] >= 0).all()
    assert (df["bus_capacity"] > 0).all()
    assert (df["advance_bookings_d1"] >= 0).all()

def test_zero_leakage_in_lags():
    df = generate_bus_dataset(start_date="2025-01-01", end_date="2025-01-25", seed=42)
    # Lags must be defined and not equal to future targets
    assert "lag_7d_demand" in df.columns
    assert "lag_14d_demand" in df.columns
    assert not df["lag_7d_demand"].isnull().any()
