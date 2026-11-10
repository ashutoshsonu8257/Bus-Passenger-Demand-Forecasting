"""
Model architectures and factory for Bus Passenger-Demand Forecasting.
Implements:
  1. Ridge Regression (regularized linear baseline)
  2. Random Forest Regressor (non-linear ensemble)
  3. LightGBM Regressor (gradient boosted decision tree - production candidate)
"""

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from typing import Dict, Any
from src.config import RANDOM_SEED

def get_candidate_models(seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate models initialized with reproducible seeds.
    """
    models = {
        "Ridge_Regression": Ridge(
            alpha=10.0,
            random_state=seed
        ),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=seed
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        )
    }
    return models
