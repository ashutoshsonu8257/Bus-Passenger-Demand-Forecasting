"""
Exploratory Data Analysis (EDA) Script.
Profiles dataset distributions, missingness, temporal cyclicality,
route-level demand variations, and advance booking correlations.
Exports figures to reports/figures/.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RAW_DATA_PATH, FIGURES_DIR, TARGET_COLUMN

def run_eda():
    print("--- Running Exploratory Data Analysis (EDA) ---")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"])

    print(f"Dataset Shape: {df.shape}")
    print(f"Date Span: {df['departure_datetime'].min()} to {df['departure_datetime'].max()}")
    print("\nSummary Statistics of Numeric Columns:")
    print(df.describe().round(2).to_string())

    sns.set_theme(style="whitegrid")

    # 1. Demand Distribution by Route
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="route_id", y=TARGET_COLUMN, palette="Set2")
    plt.title("Passenger Demand Distribution Across Bus Routes", fontsize=13, fontweight="bold")
    plt.xlabel("Route Identifier")
    plt.ylabel("Passenger Count per Trip")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_route_distribution.png", dpi=200)
    plt.close()

    # 2. Hourly Demand Profile: Commuter vs Weekend
    plt.figure(figsize=(10, 5))
    hourly_agg = df.groupby(["departure_hour", "is_weekend"])[TARGET_COLUMN].mean().reset_index()
    hourly_agg["Day Type"] = hourly_agg["is_weekend"].map({0: "Weekday (Mon-Fri)", 1: "Weekend (Sat-Sun)"})
    sns.lineplot(data=hourly_agg, x="departure_hour", y=TARGET_COLUMN, hue="Day Type", marker="o", lw=2.5)
    plt.title("Average Hourly Ridership Profile: Weekday vs Weekend Peaks", fontsize=13, fontweight="bold")
    plt.xlabel("Departure Hour (24h Clock)")
    plt.ylabel("Average Passenger Boardings")
    plt.xticks(range(5, 23))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_hourly_profile.png", dpi=200)
    plt.close()

    # 3. Advance Bookings D-1 vs Actual Final Passenger Count Correlation
    plt.figure(figsize=(8, 6))
    corr = df["advance_bookings_d1"].corr(df[TARGET_COLUMN])
    plt.scatter(df["advance_bookings_d1"], df[TARGET_COLUMN], alpha=0.25, color="#16a085")
    plt.title(f"D-1 Advance Bookings vs Final Boarded Passengers (r = {corr:.3f})", fontsize=13, fontweight="bold")
    plt.xlabel("Confirmed Bookings at D-1 18:00 Cut-off")
    plt.ylabel("Actual Boarded Passenger Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_advance_vs_actual.png", dpi=200)
    plt.close()

    # 4. Seat Occupancy Distribution & Overflow Occurrences
    plt.figure(figsize=(8, 5))
    sns.histplot(df["seat_occupancy_rate"] * 100, bins=35, kde=True, color="#e67e22")
    plt.axvline(100, color="red", linestyle="--", lw=2, label="100% Capacity Ceiling")
    plt.axvline(85, color="orange", linestyle=":", lw=2, label="85% High Load Warning")
    plt.axvline(35, color="blue", linestyle=":", lw=2, label="35% Low Demand Floor")
    plt.title("Fleet Seat Occupancy Rate Distribution (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Seat Occupancy Rate (%)")
    plt.ylabel("Trip Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_occupancy_distribution.png", dpi=200)
    plt.close()

    print("EDA Visualizations successfully exported to reports/figures/!")

if __name__ == "__main__":
    run_eda()
