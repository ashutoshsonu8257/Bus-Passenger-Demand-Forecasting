"""
Historical Route x Time Slot Baseline Model.
Serves as the domain-standard benchmark required by the assessment.
Predicts the historical average passenger count for each route and scheduled departure time.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Dict, Tuple, Optional
import joblib

class HistoricalRouteTimeBaseline(BaseEstimator, RegressorMixin):
    """
    Heuristic baseline model that looks up historical mean demand
    grouped by (route_id, departure_time).
    Provides hierarchical fallback:
      1. Route + Time Slot mean
      2. Route-level overall mean
      3. Global dataset mean
    """
    def __init__(self, target_column: str = "passenger_count"):
        self.target_column = target_column
        self.slot_lookup: Dict[Tuple[str, str], float] = {}
        self.route_lookup: Dict[str, float] = {}
        self.global_mean: float = 30.0
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Fit baseline on historical training data."""
        df = X.copy()
        if y is not None:
            df["_target"] = np.array(y)
        elif self.target_column in df.columns:
            df["_target"] = df[self.target_column]
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in X or y.")

        # Ensure departure_time string column is available
        if "departure_time" not in df.columns and "departure_datetime" in df.columns:
            df["departure_time"] = pd.to_datetime(df["departure_datetime"]).dt.strftime("%H:%M")

        self.global_mean = float(df["_target"].mean())
        self.route_lookup = df.groupby("route_id")["_target"].mean().to_dict()

        slot_grouped = df.groupby(["route_id", "departure_time"])["_target"].mean()
        self.slot_lookup = {k: float(v) for k, v in slot_grouped.items()}

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected passenger demand based on historical route x time lookup."""
        if not self.is_fitted:
            raise RuntimeError("HistoricalRouteTimeBaseline is not fitted yet.")

        df = X.copy()
        if "departure_time" not in df.columns and "departure_datetime" in df.columns:
            df["departure_time"] = pd.to_datetime(df["departure_datetime"]).dt.strftime("%H:%M")

        preds = []
        for _, row in df.iterrows():
            r_id = str(row.get("route_id", ""))
            dep_time = str(row.get("departure_time", ""))

            key = (r_id, dep_time)
            if key in self.slot_lookup:
                pred = self.slot_lookup[key]
            elif r_id in self.route_lookup:
                pred = self.route_lookup[r_id]
            else:
                pred = self.global_mean
            preds.append(pred)

        return np.array(preds)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath) -> "HistoricalRouteTimeBaseline":
        return joblib.load(filepath)
