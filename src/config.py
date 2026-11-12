"""
Configuration module for Bus Passenger-Demand Forecasting.
Defines constants, paths, random seeds, and operational capacity thresholds.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bus_trips_historical.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
BASELINE_MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METRICS_SUMMARY_PATH = ARTIFACTS_DIR / "metrics_summary.json"
ROUTE_METRICS_PATH = ARTIFACTS_DIR / "route_metrics.csv"
PEAK_METRICS_PATH = ARTIFACTS_DIR / "peak_offpeak_metrics.csv"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibility
RANDOM_SEED = 42

# Target Variable
TARGET_COLUMN = "passenger_count"

# Split Ratios (Chronological Split)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Route Metadata
ROUTE_CONFIG = {
    "ROUTE_101": {
        "name": "Metro Downtown - Tech Park Commuter",
        "origin": "Metro Downtown",
        "destination": "Tech Park",
        "type": "Commuter",
        "bus_type": "Standard_City",
        "distance_km": 42.0,
        "default_capacity": 50,
        "base_fare": 12.0,
        "duration_min": 55,
        "is_commuter_heavy": True,
        "schedules": ["06:30", "07:15", "08:00", "08:45", "12:00", "16:30", "17:15", "18:00", "19:00"]
    },
    "ROUTE_102": {
        "name": "City Center - University Campus",
        "origin": "City Center",
        "destination": "University Campus",
        "type": "University",
        "bus_type": "Standard_Campus",
        "distance_km": 28.0,
        "default_capacity": 45,
        "base_fare": 8.0,
        "duration_min": 40,
        "is_commuter_heavy": True,
        "schedules": ["07:30", "09:00", "11:30", "14:00", "16:30", "18:15", "20:00"]
    },
    "ROUTE_103": {
        "name": "Capital City - Harbor Port Intercity",
        "origin": "Capital City",
        "destination": "Harbor Port",
        "type": "Intercity",
        "bus_type": "Deluxe_Intercity",
        "distance_km": 210.0,
        "default_capacity": 50,
        "base_fare": 45.0,
        "duration_min": 180,
        "is_commuter_heavy": False,
        "schedules": ["06:00", "09:30", "13:00", "16:00", "19:30"]
    },
    "ROUTE_104": {
        "name": "Central Hub - International Airport Express",
        "origin": "Central Hub",
        "destination": "International Airport",
        "type": "Airport",
        "bus_type": "Airport_Coach",
        "distance_km": 55.0,
        "default_capacity": 40,
        "base_fare": 25.0,
        "duration_min": 65,
        "is_commuter_heavy": False,
        "schedules": ["05:30", "07:00", "09:00", "11:00", "13:30", "15:30", "17:30", "19:30", "21:30"]
    },
    "ROUTE_105": {
        "name": "Metro Valley - Mountain Resort Weekend Express",
        "origin": "Metro Valley",
        "destination": "Mountain Resort",
        "type": "Tourist",
        "bus_type": "HighCapacity_Scenic",
        "distance_km": 165.0,
        "default_capacity": 55,
        "base_fare": 38.0,
        "duration_min": 150,
        "is_commuter_heavy": False,
        "schedules": ["08:00", "10:30", "13:30", "16:30", "19:00"]
    },
    "ROUTE_106": {
        "name": "Southern Suburb - Industrial District Feeder",
        "origin": "Southern Suburb",
        "destination": "Industrial District",
        "type": "Feeder",
        "bus_type": "Standard_Feeder",
        "distance_km": 35.0,
        "default_capacity": 40,
        "base_fare": 10.0,
        "duration_min": 45,
        "is_commuter_heavy": True,
        "schedules": ["06:45", "07:45", "08:45", "15:45", "16:45", "17:45"]
    }
}

# Capacity Decision Support Thresholds (Occupancy %)
# Occupancy = Predicted Demand / Bus Capacity
THRESHOLD_LOW_DEMAND = 0.35       # < 35%: Potential trip merge or downsize coach
THRESHOLD_MODERATE_DEMAND = 0.70  # 35% - 70%: Normal scheduled operation
THRESHOLD_HIGH_DEMAND = 0.85      # 70% - 85%: High demand, monitor booking pace
THRESHOLD_OVERFLOW_RISK = 0.85    # > 85%: Dispatch second coach / upgrade to 55-seater

# Operational Status Enums
STATUS_MERGE_CANDIDATE = "LOW_DEMAND_CONSIDER_MERGE"
STATUS_BALANCED = "OPTIMAL_CAPACITY"
STATUS_HIGH_LOAD = "NEAR_CAPACITY_MONITOR"
STATUS_OVERFLOW_ALERT = "CRITICAL_OVERFLOW_RISK_DEPLOY_BACKUP"

def get_capacity_flag(predicted_demand: float, capacity: int) -> dict:
    """
    Generate decision-support action and risk flag based on capacity constraints.
    """
    if capacity <= 0:
        occupancy = 1.0
    else:
        occupancy = predicted_demand / capacity

    if occupancy < THRESHOLD_LOW_DEMAND:
        status = STATUS_MERGE_CANDIDATE
        action = "Consider downsizing vehicle or consolidating with adjacent slot to minimize operational fuel/crew cost."
        severity = "INFO"
    elif occupancy <= THRESHOLD_HIGH_DEMAND:
        status = STATUS_BALANCED
        action = "Standard dispatch. Vehicle capacity matches expected passenger volume comfortably."
        severity = "SUCCESS"
    elif occupancy <= 1.00:
        status = STATUS_HIGH_LOAD
        action = "High passenger load. Close promotional discounts and verify vehicle readiness."
        severity = "WARNING"
    else:
        status = STATUS_OVERFLOW_ALERT
        action = f"Demand exceeds bus capacity by {predicted_demand - capacity:.1f} passengers. Schedule overflow coach or redirect bookings to earlier/later service."
        severity = "DANGER"

    return {
        "predicted_demand": round(float(predicted_demand), 1),
        "bus_capacity": int(capacity),
        "occupancy_rate": round(float(occupancy), 3),
        "occupancy_percentage": f"{round(occupancy * 100, 1)}%",
        "status_code": status,
        "action_recommendation": action,
        "severity": severity,
        "excess_passengers": max(0.0, round(float(predicted_demand - capacity), 1)),
        "available_empty_seats": max(0, int(capacity - round(predicted_demand)))
    }
