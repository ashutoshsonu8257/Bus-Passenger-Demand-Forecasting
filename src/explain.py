"""
Explainability and Feature Contribution Analysis for Bus Demand Forecasting.
Provides local prediction explanations, key drivers, and decision-support risk caveats.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

def explain_single_prediction(
    input_features: Dict[str, Any],
    predicted_demand: float,
    baseline_demand: float,
    bus_capacity: int,
    rmse: float = 3.46
) -> Dict[str, Any]:
    """
    Decompose a single trip prediction into key demand drivers, uncertainty bounds,
    and decision-support recommendations.
    """
    drivers = []

    # 1. Advance Bookings Driver
    advance = input_features.get("advance_bookings_d1", 0)
    if advance > 0:
        drivers.append({
            "factor": "D-1 Advance Bookings",
            "impact": "Primary Baseline Driver",
            "direction": "Positive",
            "detail": f"{advance} confirmed tickets booked 24h before departure (~75-85% final share)."
        })

    # 2. Peak vs Off-Peak
    is_peak = input_features.get("is_peak_hour", 0)
    hour = input_features.get("departure_hour", 12)
    if is_peak == 1:
        drivers.append({
            "factor": "Rush-Hour Commute Peak",
            "impact": "High Positive Surge",
            "direction": "Positive",
            "detail": f"Departure at {hour:02d}:00 aligns with high-volume commuter rush window."
        })
    elif hour < 6 or hour > 21:
        drivers.append({
            "factor": "Late Night / Early Morning Low Demand Window",
            "impact": "Moderate Suppressor",
            "direction": "Negative",
            "detail": f"Departure time {hour:02d}:00 experiences historically reduced ridership."
        })

    # 3. Weekend vs Weekday
    is_weekend = input_features.get("is_weekend", 0)
    route_type = input_features.get("route_type", "Commuter")
    if is_weekend == 1:
        if route_type in ("Commuter", "Feeder", "University"):
            drivers.append({
                "factor": "Weekend Commute Drop",
                "impact": "Strong Demand Reduction",
                "direction": "Negative",
                "detail": f"Commercial/office travel is minimal on weekends for route type '{route_type}'."
            })
        elif route_type in ("Tourist", "Intercity"):
            drivers.append({
                "factor": "Weekend Leisure Surge",
                "impact": "Strong Leisure Influx",
                "direction": "Positive",
                "detail": "High weekend getaway and leisure travel volume."
            })

    # 4. Special Event
    event = str(input_features.get("special_event", "None"))
    if event not in ("None", "", "nan"):
        drivers.append({
            "factor": f"Special Event: {event}",
            "impact": "High Event Demand Spurt",
            "direction": "Positive",
            "detail": f"Significant surge in destination travel attributed to scheduled '{event}'."
        })

    # 5. Weather Conditions
    weather = str(input_features.get("weather_condition", "Clear"))
    impact_score = float(input_features.get("weather_impact_score", 0.0))
    if weather in ("Snow", "Heavy_Rain") or impact_score >= 0.2:
        drivers.append({
            "factor": f"Adverse Weather: {weather}",
            "impact": "Demand Dampening",
            "direction": "Negative",
            "detail": f"Severe weather ({weather}) typically reduces discretionary walk-up bookings by 10-25%."
        })

    # 6. Holiday Proximity
    days_to_hol = input_features.get("days_to_holiday", 10)
    if days_to_hol <= 2:
        drivers.append({
            "factor": f"Holiday Proximity ({days_to_hol} days away)",
            "impact": "Pre-Holiday Travel Surge",
            "direction": "Positive",
            "detail": "Elevated intercity and airport travel ahead of major calendar holiday."
        })

    # Uncertainty Bounds (95% Confidence Interval ~ ±1.96 * RMSE)
    margin = 1.96 * rmse
    lower_bound = max(0.0, round(predicted_demand - margin, 1))
    upper_bound = round(predicted_demand + margin, 1)

    delta_from_baseline = round(predicted_demand - baseline_demand, 1)

    limitations = [
        "Cold-Start Route: Model relies heavily on historical slot averages and advance bookings; new routes lack 7d/14d lag data.",
        "Unplanned Service Disruptions: Sudden road closures, transit strikes, or spontaneous weather events occurring <24h before departure are outside the model horizon.",
        "Extreme Outlier Events: Multi-day severe blizzard or city-wide emergency will distort normal booking conversion rates.",
        "Human Oversight Rule: This tool provides decision support. Fleet dispatchers must manually verify fleet and driver availability before cancelling or adding runs."
    ]

    return {
        "predicted_demand": round(float(predicted_demand), 1),
        "baseline_demand": round(float(baseline_demand), 1),
        "delta_vs_baseline": delta_from_baseline,
        "prediction_interval_95pct": {
            "lower": lower_bound,
            "upper": upper_bound,
            "margin_error": round(margin, 1)
        },
        "key_drivers": drivers,
        "model_limitations": limitations
    }
