"""
Synthetic Bus Trip Ridership Data Generator.
Generates realistic, trip-level scheduled bus booking data with cyclical seasonality,
commuter vs. leisure route patterns, holiday surges, weather effects, and D-1 advance bookings.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Tuple, List, Dict
from src.config import (
    RANDOM_SEED,
    ROUTE_CONFIG,
    RAW_DATA_PATH
)

# Major public holidays in the calendar window (2025 - early 2026)
PUBLIC_HOLIDAYS = [
    datetime(2025, 1, 1).date(),   # New Year's Day
    datetime(2025, 1, 20).date(),  # MLK Day
    datetime(2025, 2, 17).date(),  # Presidents' Day
    datetime(2025, 5, 26).date(),  # Memorial Day
    datetime(2025, 7, 4).date(),   # Independence Day
    datetime(2025, 9, 1).date(),   # Labor Day
    datetime(2025, 11, 27).date(), # Thanksgiving
    datetime(2025, 11, 28).date(), # Black Friday
    datetime(2025, 12, 24).date(), # Christmas Eve
    datetime(2025, 12, 25).date(), # Christmas Day
    datetime(2025, 12, 31).date(), # New Year's Eve
    datetime(2026, 1, 1).date(),   # New Year's Day
    datetime(2026, 1, 19).date(),  # MLK Day
    datetime(2026, 2, 16).date()   # Presidents' Day
]

def get_season(month: int) -> str:
    """Return season for a given calendar month."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"

def calculate_days_to_holiday(d) -> int:
    """Return minimum days between date and any holiday."""
    return min(abs((d - h).days) for h in PUBLIC_HOLIDAYS)

def is_peak_slot(hour: int, minute: int) -> int:
    """1 if morning peak (07:00-09:30) or evening peak (16:30-19:30), else 0."""
    t = hour + minute / 60.0
    if (7.0 <= t <= 9.5) or (16.5 <= t <= 19.5):
        return 1
    return 0

def generate_bus_dataset(
    start_date: str = "2025-01-01",
    end_date: str = "2026-03-01",
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate synthetic trip-level bus dataset across multiple routes.
    """
    np.random.seed(seed)
    random.seed(seed)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    records = []

    # Standard departure schedule templates per route
    route_schedules = {
        "ROUTE_101": [(6, 30), (7, 15), (8, 0), (8, 45), (12, 0), (16, 30), (17, 15), (18, 0), (19, 0)],
        "ROUTE_102": [(7, 30), (9, 0), (11, 30), (14, 0), (16, 30), (18, 15), (20, 0)],
        "ROUTE_103": [(6, 0), (9, 30), (13, 0), (16, 0), (19, 30)],
        "ROUTE_104": [(5, 30), (7, 0), (9, 0), (11, 0), (13, 30), (15, 30), (17, 30), (19, 30), (21, 30)],
        "ROUTE_105": [(8, 0), (10, 30), (13, 30), (16, 30), (19, 0)],
        "ROUTE_106": [(6, 45), (7, 45), (8, 45), (15, 45), (16, 45), (17, 45)]
    }

    bus_type_options = {
        "ROUTE_101": ("Standard_City", 50),
        "ROUTE_102": ("Standard_Campus", 45),
        "ROUTE_103": ("Deluxe_Intercity", 50),
        "ROUTE_104": ("Airport_Coach", 40),
        "ROUTE_105": ("HighCapacity_Scenic", 55),
        "ROUTE_106": ("Standard_Feeder", 40)
    }

    weather_types = ["Clear", "Rain", "Heavy_Rain", "Snow", "Fog"]
    weather_probs = [0.65, 0.20, 0.05, 0.05, 0.05]
    event_types = ["None", "Sports_Game", "Concert", "Festival", "Conference"]
    event_probs = [0.84, 0.05, 0.04, 0.04, 0.03]

    for day_idx in range(total_days):
        current_date = (start_dt + timedelta(days=day_idx)).date()
        month = current_date.month
        day_of_week = current_date.weekday() # 0 = Mon, 6 = Sun
        day_name = current_date.strftime("%A")
        is_weekend = 1 if day_of_week in (5, 6) else 0
        is_holiday = 1 if current_date in PUBLIC_HOLIDAYS else 0
        days_to_hol = calculate_days_to_holiday(current_date)
        season = get_season(month)

        # Weather for this date across the region
        weather = np.random.choice(weather_types, p=weather_probs)
        if weather == "Clear":
            weather_impact = 0.0
        elif weather == "Rain":
            weather_impact = 0.08
        elif weather == "Fog":
            weather_impact = 0.12
        elif weather == "Heavy_Rain":
            weather_impact = 0.20
        else: # Snow
            weather_impact = 0.30

        # Special event indicator for today
        special_event = np.random.choice(event_types, p=event_probs)
        event_boost = 0.25 if special_event != "None" else 0.0

        for route_id, slots in route_schedules.items():
            r_cfg = ROUTE_CONFIG[route_id]
            bus_type, bus_cap = bus_type_options[route_id]

            for hour, minute in slots:
                dep_dt = datetime.combine(current_date, datetime.min.time()).replace(
                    hour=hour, minute=minute
                )
                trip_id = f"TRIP_{current_date.strftime('%Y%m%d')}_{route_id}_{hour:02d}{minute:02d}"
                peak_flag = is_peak_slot(hour, minute)

                # Route-specific latent baseline passenger volume
                if route_id == "ROUTE_101":  # Commuter Downtown - Tech Park
                    base_volume = 32.0
                    # Commuter peaks
                    if not is_weekend:
                        if peak_flag:
                            base_volume += 14.0
                        else:
                            base_volume += 2.0
                    else:
                        base_volume -= 16.0  # Much lower on weekends
                    # Weather slight reduction
                    base_volume -= weather_impact * 8.0

                elif route_id == "ROUTE_102":  # University Campus
                    base_volume = 26.0
                    # Term time seasonality (lower in summer June-August, low in late Dec)
                    if month in (6, 7, 8):
                        base_volume -= 12.0
                    elif month == 12 and current_date.day > 15:
                        base_volume -= 14.0
                    # Class schedule peak (8am - 18pm)
                    if not is_weekend and 8 <= hour <= 17:
                        base_volume += 11.0
                    elif is_weekend:
                        base_volume -= 8.0

                elif route_id == "ROUTE_103":  # Intercity
                    base_volume = 30.0
                    # Friday evening & Sunday afternoon surges
                    if day_of_week == 4 and hour >= 16:  # Friday PM
                        base_volume += 15.0
                    elif day_of_week == 6 and hour >= 13:  # Sunday PM
                        base_volume += 14.0
                    elif is_weekend:
                        base_volume += 6.0
                    # Pre-holiday surge
                    if days_to_hol <= 2:
                        base_volume += 12.0

                elif route_id == "ROUTE_104":  # Airport Express
                    base_volume = 25.0
                    # Early morning and late evening flight waves
                    if 5 <= hour <= 8 or 18 <= hour <= 21:
                        base_volume += 10.0
                    # Summer travel surge
                    if season == "Summer":
                        base_volume += 6.0
                    # Holiday travel
                    if days_to_hol <= 3:
                        base_volume += 8.0

                elif route_id == "ROUTE_105":  # Tourist Scenic
                    base_volume = 18.0
                    # Huge weekend and summer boost
                    if is_weekend:
                        base_volume += 22.0
                    if season == "Summer":
                        base_volume += 12.0
                    elif season == "Winter":
                        base_volume -= 6.0
                    # Severe weather discourages tourists
                    base_volume -= weather_impact * 15.0

                elif route_id == "ROUTE_106":  # Suburb - Industrial Feeder
                    base_volume = 24.0
                    # Shift change peaks: 6:45, 15:45, 16:45
                    if not is_weekend:
                        if hour in (6, 7, 15, 16):
                            base_volume += 12.0
                    else:
                        base_volume -= 10.0

                # Add event boost
                if event_boost > 0:
                    base_volume += event_boost * 15.0

                # Random gaussian noise
                noise = np.random.normal(loc=0.0, scale=3.2)
                latent_demand = base_volume + noise

                # Bound latent demand realistically
                latent_demand = max(2.0, latent_demand)

                # Compute D-1 advance booking (typically 65% - 85% of final passenger count)
                booking_fraction = np.clip(np.random.normal(loc=0.78, scale=0.06), 0.55, 0.95)
                # For airport/intercity, advance booking is higher; for feeder, walkup is higher
                if route_id in ("ROUTE_103", "ROUTE_104", "ROUTE_105"):
                    booking_fraction += 0.08
                elif route_id in ("ROUTE_101", "ROUTE_106"):
                    booking_fraction -= 0.08
                booking_fraction = np.clip(booking_fraction, 0.40, 0.92)

                advance_bookings_d1 = int(round(latent_demand * booking_fraction))
                # Walkups, last 24h bookings, minus no-shows
                walkups = np.random.poisson(lam=max(1.0, latent_demand * (1.0 - booking_fraction)))
                no_shows = np.random.binomial(n=advance_bookings_d1, p=0.05) if advance_bookings_d1 > 0 else 0
                
                final_boarded = advance_bookings_d1 + walkups - no_shows
                final_boarded = max(1, int(round(final_boarded)))

                records.append({
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "route_name": r_cfg["name"],
                    "route_type": r_cfg["type"],
                    "departure_datetime": dep_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "departure_date": str(current_date),
                    "departure_time": f"{hour:02d}:{minute:02d}",
                    "departure_hour": hour,
                    "departure_minute": minute,
                    "day_of_week": day_of_week,
                    "day_name": day_name,
                    "is_weekend": is_weekend,
                    "month": month,
                    "season": season,
                    "is_holiday": is_holiday,
                    "days_to_holiday": days_to_hol,
                    "is_peak_hour": peak_flag,
                    "route_distance_km": r_cfg["distance_km"],
                    "scheduled_duration_min": r_cfg["duration_min"],
                    "bus_type": bus_type,
                    "bus_capacity": bus_cap,
                    "standard_fare": r_cfg["base_fare"],
                    "weather_condition": weather,
                    "weather_impact_score": round(weather_impact, 2),
                    "special_event": special_event,
                    "advance_bookings_d1": advance_bookings_d1,
                    "passenger_count": final_boarded
                })

    df = pd.DataFrame(records)
    # Ensure sorted by datetime strictly
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"])
    df = df.sort_values(by=["departure_datetime", "route_id"]).reset_index(drop=True)

    # Compute strictly historical lag features (chronologically safe, >= 24h lag)
    # 7-day lag: passenger demand for this route and slot 7 days ago
    df = _add_chronological_lag_features(df)

    # Calculate seat occupancy rate
    df["seat_occupancy_rate"] = (df["passenger_count"] / df["bus_capacity"]).round(4)

    return df

def _add_chronological_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute strictly chronological lag and rolling features.
    Ensures zero future leakage by operating over past trip history.
    """
    df = df.copy()
    df["slot_key"] = df["route_id"] + "_" + df["departure_time"]

    # 7-day and 14-day lag per route & time slot
    df["lag_7d_demand"] = df.groupby("slot_key")["passenger_count"].shift(7)
    df["lag_14d_demand"] = df.groupby("slot_key")["passenger_count"].shift(14)

    # Historical expanding / rolling 28-trip average per route (prior to current trip)
    df["rolling_mean_route_4w"] = (
        df.groupby("route_id")["passenger_count"]
        .transform(lambda s: s.shift(1).rolling(window=28, min_periods=5).mean())
    )

    # Impute initial periods where 7/14 day lag is not yet available with slot expanding mean
    slot_means = df.groupby("slot_key")["passenger_count"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    overall_route_means = df.groupby("route_id")["passenger_count"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    df["lag_7d_demand"] = df["lag_7d_demand"].fillna(slot_means).fillna(overall_route_means).bfill().round(1)
    df["lag_14d_demand"] = df["lag_14d_demand"].fillna(slot_means).fillna(overall_route_means).bfill().round(1)
    df["rolling_mean_route_4w"] = df["rolling_mean_route_4w"].fillna(overall_route_means).bfill().round(1)

    df = df.drop(columns=["slot_key"])
    return df

def generate_and_save_data(output_path: str = str(RAW_DATA_PATH)) -> pd.DataFrame:
    """Generate and save the raw dataset to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = generate_bus_dataset()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} bus trip records saved to {output_path}")
    print(f"Date range: {df['departure_date'].min()} to {df['departure_date'].max()}")
    print(f"Routes: {df['route_id'].unique().tolist()}")
    print(f"Target distribution (passenger_count): Mean={df['passenger_count'].mean():.2f}, Min={df['passenger_count'].min()}, Max={df['passenger_count'].max()}")
    return df

if __name__ == "__main__":
    generate_and_save_data()
