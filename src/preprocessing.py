"""
Feature Engineering and Preprocessing Pipeline for Bus Passenger-Demand Forecasting.
Handles cyclical time encodings, one-hot encodings, scaling, and leakage-safe transformations.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List, Tuple, Dict, Any, Optional
import joblib

CATEGORICAL_FEATURES = [
    "route_id",
    "route_type",
    "weather_condition",
    "special_event",
    "bus_type"
]

NUMERICAL_FEATURES = [
    "advance_bookings_d1",
    "route_distance_km",
    "scheduled_duration_min",
    "standard_fare",
    "bus_capacity",
    "days_to_holiday",
    "weather_impact_score",
    "lag_7d_demand",
    "lag_14d_demand",
    "rolling_mean_route_4w"
]

BINARY_FEATURES = [
    "is_weekend",
    "is_holiday",
    "is_peak_hour"
]

CYCLICAL_FEATURES = [
    "departure_hour",
    "day_of_week",
    "month"
]

class BusDemandPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free preprocessing pipeline that encodes cyclical temporal features,
    one-hot encodes route and operational categories, and scales numerical regressors.
    """
    def __init__(self, scale_numerics: bool = False):
        self.scale_numerics = scale_numerics
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler() if scale_numerics else None
        self.numerical_imputer_values: Dict[str, float] = {}
        self.categorical_defaults: Dict[str, str] = {
            "route_id": "ROUTE_101",
            "route_type": "Commuter",
            "weather_condition": "Clear",
            "special_event": "None",
            "bus_type": "Standard_City"
        }
        self.feature_names_: List[str] = []
        self.is_fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive cyclical and helper features from raw attributes."""
        df = df.copy()

        # Handle string or missing special event
        if "special_event" in df.columns:
            df["special_event"] = df["special_event"].fillna("None").astype(str)
            df.loc[df["special_event"].isin(["nan", "None", ""]), "special_event"] = "None"
        else:
            df["special_event"] = "None"

        if "weather_condition" in df.columns:
            df["weather_condition"] = df["weather_condition"].fillna("Clear").astype(str)
        else:
            df["weather_condition"] = "Clear"

        if "route_type" not in df.columns:
            df["route_type"] = "Commuter"

        if "bus_type" not in df.columns:
            df["bus_type"] = "Standard_City"

        # Parse departure datetime if present
        if "departure_datetime" in df.columns:
            try:
                dt = pd.to_datetime(df["departure_datetime"])
                if "departure_hour" not in df.columns:
                    df["departure_hour"] = dt.dt.hour
                if "day_of_week" not in df.columns:
                    df["day_of_week"] = dt.dt.weekday
                if "month" not in df.columns:
                    df["month"] = dt.dt.month
            except Exception:
                pass

        # Fallbacks for temporal features
        if "departure_hour" not in df.columns:
            df["departure_hour"] = 8
        else:
            df["departure_hour"] = df["departure_hour"].fillna(8).astype(int)

        if "day_of_week" not in df.columns:
            df["day_of_week"] = 1
        else:
            df["day_of_week"] = df["day_of_week"].fillna(1).astype(int)

        if "month" not in df.columns:
            df["month"] = 1
        else:
            df["month"] = df["month"].fillna(1).astype(int)

        if "is_weekend" not in df.columns:
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        else:
            df["is_weekend"] = df["is_weekend"].fillna(0).astype(int)

        # Peak hour indicator if not present
        if "is_peak_hour" not in df.columns:
            df["is_peak_hour"] = df["departure_hour"].apply(
                lambda h: 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0
            )
        else:
            df["is_peak_hour"] = df["is_peak_hour"].fillna(0).astype(int)

        # Cyclical Encodings (Hour: 24, Day of week: 7, Month: 12)
        df["hour_sin"] = np.sin(2 * np.pi * df["departure_hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["departure_hour"] / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

        return df

    def fit(self, X: pd.DataFrame, y=None):
        """Fit encodings and scalers strictly on training data."""
        df_eng = self._engineer_features(X)

        # Compute numerical imputers strictly on training split
        for col in NUMERICAL_FEATURES:
            if col in df_eng.columns:
                self.numerical_imputer_values[col] = float(df_eng[col].median())
            else:
                self.numerical_imputer_values[col] = 0.0

        # Fit categorical encoder
        for col in CATEGORICAL_FEATURES:
            if col not in df_eng.columns:
                df_eng[col] = self.categorical_defaults.get(col, "None")
        cat_df = df_eng[CATEGORICAL_FEATURES].fillna(self.categorical_defaults)
        self.encoder.fit(cat_df)

        cat_feature_names = self.encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        cyclical_names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        self.feature_names_ = (
            NUMERICAL_FEATURES +
            BINARY_FEATURES +
            cyclical_names +
            cat_feature_names
        )

        if self.scale_numerics and self.scaler:
            num_data = df_eng[NUMERICAL_FEATURES].fillna(self.numerical_imputer_values)
            self.scaler.fit(num_data)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input features using fitted statistics."""
        if not self.is_fitted:
            raise RuntimeError("BusDemandPreprocessor must be fitted before calling transform().")

        df_eng = self._engineer_features(X)

        # Numerical columns with training median fallback
        for col in NUMERICAL_FEATURES:
            if col not in df_eng.columns:
                df_eng[col] = self.numerical_imputer_values.get(col, 0.0)
            else:
                df_eng[col] = df_eng[col].fillna(self.numerical_imputer_values.get(col, 0.0))

        # Binary features
        for col in BINARY_FEATURES:
            if col not in df_eng.columns:
                df_eng[col] = 0
            else:
                df_eng[col] = df_eng[col].fillna(0).astype(int)

        # Categoricals
        for col in CATEGORICAL_FEATURES:
            if col not in df_eng.columns:
                df_eng[col] = self.categorical_defaults.get(col, "None")
        cat_df = df_eng[CATEGORICAL_FEATURES].fillna(self.categorical_defaults)
        encoded_cats = self.encoder.transform(cat_df)
        cat_feature_names = self.encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        encoded_cat_df = pd.DataFrame(encoded_cats, columns=cat_feature_names, index=X.index)

        # Cyclical features
        cyclical_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        cyc_df = df_eng[cyclical_cols]

        if self.scale_numerics and self.scaler:
            scaled_num = self.scaler.transform(df_eng[NUMERICAL_FEATURES])
            num_df = pd.DataFrame(scaled_num, columns=NUMERICAL_FEATURES, index=X.index)
        else:
            num_df = df_eng[NUMERICAL_FEATURES].copy()

        bin_df = df_eng[BINARY_FEATURES].copy()

        transformed_df = pd.concat([num_df, bin_df, cyc_df, encoded_cat_df], axis=1)
        # Ensure exact column alignment
        transformed_df = transformed_df[self.feature_names_]
        return transformed_df

    def get_feature_names(self) -> List[str]:
        return self.feature_names_

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath) -> "BusDemandPreprocessor":
        return joblib.load(filepath)
