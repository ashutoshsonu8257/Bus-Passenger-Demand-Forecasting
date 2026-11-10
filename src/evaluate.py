"""
Evaluation engine for Bus Passenger-Demand Forecasting.
Calculates MAE, RMSE, WAPE (Weighted Absolute Percentage Error), R2, and MAPE,
along with per-route and peak/off-peak stratified performance breakdowns.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error:
    WAPE = (sum(|y - y_hat|) / sum(y)) * 100%
    """
    denominator = np.sum(y_true)
    if denominator == 0:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / denominator) * 100.0)

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (avoiding divide-by-zero)."""
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute overall regression evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wape = float(calculate_wape(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(calculate_mape(y_true, y_pred))

    residuals = y_true - y_pred
    mean_residual = float(np.mean(residuals))
    p90_error = float(np.percentile(np.abs(residuals), 90))

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE_percent": round(wape, 2),
        "R2": round(r2, 4),
        "MAPE_percent": round(mape, 2),
        "mean_residual": round(mean_residual, 3),
        "p90_absolute_error": round(p90_error, 3)
    }

def evaluate_by_route(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """Calculate error breakdown per bus route."""
    rows = []
    for route_id, group in df.groupby("route_id"):
        y_t = group[y_true_col].values
        y_p = group[y_pred_col].values
        metrics = compute_regression_metrics(y_t, y_p)
        rows.append({
            "route_id": route_id,
            "sample_count": len(group),
            "mean_actual": round(float(np.mean(y_t)), 1),
            "mean_pred": round(float(np.mean(y_p)), 1),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE_%": metrics["WAPE_percent"],
            "R2": metrics["R2"]
        })
    return pd.DataFrame(rows).sort_values(by="route_id").reset_index(drop=True)

def evaluate_by_peak(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """Calculate error breakdown between peak and off-peak departure windows."""
    rows = []
    # If is_peak_hour not explicitly present, calculate it
    if "is_peak_hour" not in df.columns and "departure_hour" in df.columns:
        df["is_peak_hour"] = df["departure_hour"].apply(
            lambda h: 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0
        )

    for peak_flag in [1, 0]:
        label = "Peak Period (07-09 & 17-19)" if peak_flag == 1 else "Off-Peak Period"
        group = df[df["is_peak_hour"] == peak_flag]
        if len(group) == 0:
            continue
        y_t = group[y_true_col].values
        y_p = group[y_pred_col].values
        metrics = compute_regression_metrics(y_t, y_p)
        rows.append({
            "period": label,
            "is_peak": peak_flag,
            "sample_count": len(group),
            "mean_actual": round(float(np.mean(y_t)), 1),
            "mean_pred": round(float(np.mean(y_p)), 1),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE_%": metrics["WAPE_percent"],
            "R2": metrics["R2"]
        })
    return pd.DataFrame(rows).reset_index(drop=True)
