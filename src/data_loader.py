"""
Data loader and chronological train/validation/test splitter.
Ensures zero data leakage across temporal boundaries.
"""

from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Any
from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    TARGET_COLUMN
)

def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw bus trip dataset and format timestamps."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at {filepath}. Run data_generator first.")
    df = pd.read_csv(filepath)
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"])
    return df

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform rigorous data quality and sanity checks.
    """
    total_records = len(df)
    missing_counts = df.isnull().sum().to_dict()
    has_nulls = any(v > 0 for v in missing_counts.values())

    # Check target integrity
    negative_passengers = int((df[TARGET_COLUMN] < 0).sum())
    null_passengers = int(df[TARGET_COLUMN].isnull().sum())

    # Check temporal ordering
    is_sorted = df["departure_datetime"].is_monotonic_increasing

    report = {
        "total_records": total_records,
        "date_min": str(df["departure_datetime"].min()),
        "date_max": str(df["departure_datetime"].max()),
        "unique_routes": df["route_id"].nunique(),
        "routes": sorted(df["route_id"].unique().tolist()),
        "has_missing_values": has_nulls,
        "missing_summary": {k: v for k, v in missing_counts.items() if v > 0},
        "negative_target_count": negative_passengers,
        "null_target_count": null_passengers,
        "is_chronologically_sorted": is_sorted
    }
    return report

def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data strictly chronologically to prevent future information leakage.
    Train -> Validation -> Test.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    df_sorted = df.sort_values(by="departure_datetime").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    # Verify no timestamp overlap
    train_max = train_df["departure_datetime"].max()
    val_min = val_df["departure_datetime"].min()
    val_max = val_df["departure_datetime"].max()
    test_min = test_df["departure_datetime"].min()

    assert train_max <= val_min, f"Data leakage detected! Train max ({train_max}) > Val min ({val_min})"
    assert val_max <= test_min, f"Data leakage detected! Val max ({val_max}) > Test min ({test_min})"

    return train_df, val_df, test_df

def prepare_and_save_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data, validate, execute chronological split, and save processed splits.
    """
    df = load_raw_data()
    quality = validate_data_quality(df)
    print("--- Data Quality Report ---")
    for k, v in quality.items():
        print(f"{k}: {v}")

    train_df, val_df, test_df = split_chronological(df)

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print("\n--- Chronological Split Summary ---")
    print(f"Train set: {len(train_df)} trips ({train_df['departure_datetime'].min()} to {train_df['departure_datetime'].max()})")
    print(f"Val set:   {len(val_df)} trips ({val_df['departure_datetime'].min()} to {val_df['departure_datetime'].max()})")
    print(f"Test set:  {len(test_df)} trips ({test_df['departure_datetime'].min()} to {test_df['departure_datetime'].max()})")

    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_and_save_splits()
