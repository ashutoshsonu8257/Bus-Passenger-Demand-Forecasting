"""
FastAPI REST API for Bus Passenger-Demand Forecasting.
Provides production endpoints for real-time model inference, batch forecasting,
route catalog lookup, and model performance metrics.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
import json
import pandas as pd

from src.config import ROUTE_CONFIG, METRICS_SUMMARY_PATH, ROUTE_METRICS_PATH, PEAK_METRICS_PATH, FIGURES_DIR
from src.inference import get_forecaster

STATIC_DIR = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload ML artifacts on startup
    get_forecaster()
    yield

app = FastAPI(
    title="Bus Passenger-Demand Forecasting API",
    description="Production Decision-Support ML API for bus fleet capacity planning and trip demand forecasting.",
    version="1.0.0",
    lifespan=lifespan
)

if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

@app.get("/ui", include_in_schema=False)
def serve_redbus_ui():
    """Serve the RedBus-style interactive web interface."""
    return FileResponse(STATIC_DIR / "index.html")

# Pydantic Request Models
class TripForecastRequest(BaseModel):
    route_id: str = Field(..., json_schema_extra={"example": "ROUTE_101"}, description="Unique route code (e.g. ROUTE_101 to ROUTE_106)")
    departure_datetime: Optional[str] = Field(None, json_schema_extra={"example": "2026-11-15 08:00:00"}, description="Scheduled departure ISO datetime")
    departure_time: Optional[str] = Field("08:00", json_schema_extra={"example": "08:00"}, description="Scheduled departure time HH:MM")
    departure_hour: Optional[int] = Field(8, ge=0, le=23, description="Departure hour (0-23)")
    departure_minute: Optional[int] = Field(0, ge=0, le=59, description="Departure minute (0-59)")
    day_of_week: Optional[int] = Field(1, ge=0, le=6, description="0=Monday, 6=Sunday")
    month: Optional[int] = Field(11, ge=1, le=12, description="Month of departure (1-12)")
    bus_capacity: Optional[int] = Field(50, ge=10, le=100, description="Seating capacity of the assigned bus")
    advance_bookings_d1: Optional[int] = Field(35, ge=0, description="Confirmed bookings 24 hours prior to departure")
    weather_condition: Optional[str] = Field("Clear", json_schema_extra={"example": "Clear"}, description="Clear, Rain, Heavy_Rain, Snow, Fog")
    weather_impact_score: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Weather severity index (0.0 to 1.0)")
    special_event: Optional[str] = Field("None", json_schema_extra={"example": "None"}, description="None, Concert, Sports_Game, Festival, Conference")
    days_to_holiday: Optional[int] = Field(15, ge=0, description="Calendar days to nearest public holiday")

class BatchForecastRequest(BaseModel):
    trips: List[TripForecastRequest]

@app.get("/")
def root():
    return {
        "service": "Bus Passenger-Demand Forecasting API",
        "status": "online",
        "ui_url": "/ui",
        "docs_url": "/docs",
        "description": "Decision-support machine learning interface for capacity-aware bus fleet scheduling."
    }

@app.get("/health")
def healthcheck():
    forecaster = get_forecaster()
    return {
        "status": "healthy",
        "loaded_model": forecaster.champion_name,
        "test_rmse": forecaster.test_rmse,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/routes")
def list_routes():
    """Return available route configurations and operational metadata."""
    return {
        "route_count": len(ROUTE_CONFIG),
        "routes": ROUTE_CONFIG
    }

@app.get("/metrics")
def get_metrics():
    """Return held-out test set model evaluation metrics and breakdowns."""
    if not METRICS_SUMMARY_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics summary not generated yet. Run train.py.")

    with open(METRICS_SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    route_metrics = []
    if ROUTE_METRICS_PATH.exists():
        route_metrics = pd.read_csv(ROUTE_METRICS_PATH).to_dict(orient="records")

    peak_metrics = []
    if PEAK_METRICS_PATH.exists():
        peak_metrics = pd.read_csv(PEAK_METRICS_PATH).to_dict(orient="records")

    return {
        "model_comparison": summary.get("models"),
        "champion_model": summary.get("champion_model"),
        "test_improvement": summary.get("test_improvement"),
        "route_breakdown": route_metrics,
        "peak_offpeak_breakdown": peak_metrics
    }

@app.post("/predict")
def predict_demand(request: TripForecastRequest):
    """
    Generate demand forecast, baseline comparison, capacity flags, and explanation for a scheduled bus trip.
    """
    forecaster = get_forecaster()
    try:
        input_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        result = forecaster.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(request: BatchForecastRequest):
    """
    Generate batch demand forecasts for multiple scheduled departures.
    """
    forecaster = get_forecaster()
    try:
        trips_list = [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in request.trips]
        results = forecaster.predict_batch(trips_list)
        return {
            "batch_size": len(results),
            "forecasts": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")
