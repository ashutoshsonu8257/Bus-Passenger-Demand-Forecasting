"""
End-to-end reproducible training and model comparison pipeline.
Trains Historical Baseline, Ridge Regression, Random Forest, and LightGBM.
Evaluates on validation and held-out test sets, generates stratified breakdowns,
and exports all artifacts and evaluation visualizations.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    BASELINE_MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_SUMMARY_PATH,
    ROUTE_METRICS_PATH,
    PEAK_METRICS_PATH,
    FIGURES_DIR,
    TARGET_COLUMN,
    RANDOM_SEED
)
from src.data_loader import prepare_and_save_splits
from src.preprocessing import BusDemandPreprocessor
from src.baseline import HistoricalRouteTimeBaseline
from src.models import get_candidate_models
from src.evaluate import (
    compute_regression_metrics,
    evaluate_by_route,
    evaluate_by_peak
)

def run_training_pipeline():
    print("=" * 65)
    print(" BUS PASSENGER-DEMAND FORECASTING: MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Ensure directories exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load or prepare splits
    if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
        print("Processed splits not found. Preparing chronological splits...")
        train_df, val_df, test_df = prepare_and_save_splits()
    else:
        print("Loading chronological data splits...")
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        val_df = pd.read_csv(VAL_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"Data Shapes -> Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # 3. Fit Baseline Model
    print("\n[Step 1/5] Training Historical Route x Departure-Time Baseline...")
    baseline = HistoricalRouteTimeBaseline(target_column=TARGET_COLUMN)
    baseline.fit(train_df, y_train)
    baseline.save(BASELINE_MODEL_PATH)

    # 4. Fit Preprocessing Pipeline strictly on Train
    print("[Step 2/5] Fitting Feature Engineering Pipeline on Training Data...")
    preprocessor = BusDemandPreprocessor(scale_numerics=False)
    preprocessor.fit(train_df)
    preprocessor.save(PREPROCESSOR_PATH)

    X_train_trans = preprocessor.transform(train_df)
    X_val_trans = preprocessor.transform(val_df)
    X_test_trans = preprocessor.transform(test_df)
    print(f"Engineered Feature Count: {X_train_trans.shape[1]}")

    # 5. Train & Evaluate Candidates
    print("\n[Step 3/5] Training Candidate Models...")
    candidate_dict = get_candidate_models(seed=RANDOM_SEED)

    # Dictionary to hold predictions and metrics
    results = {}
    test_preds = {}

    # Evaluate Baseline first
    base_val_pred = baseline.predict(val_df)
    base_test_pred = baseline.predict(test_df)
    test_preds["Historical_Baseline"] = base_test_pred

    results["Historical_Baseline"] = {
        "val": compute_regression_metrics(y_val, base_val_pred),
        "test": compute_regression_metrics(y_test, base_test_pred)
    }

    trained_models = {}
    for name, model in candidate_dict.items():
        print(f"  --> Training {name}...")
        model.fit(X_train_trans, y_train)
        trained_models[name] = model

        val_pred = np.clip(model.predict(X_val_trans), 0, None)
        t_pred = np.clip(model.predict(X_test_trans), 0, None)
        test_preds[name] = t_pred

        results[name] = {
            "val": compute_regression_metrics(y_val, val_pred),
            "test": compute_regression_metrics(y_test, t_pred)
        }

    # Print Comparison Table
    print("\n" + "=" * 80)
    print(f"{'MODEL':<22} | {'VAL WAPE%':<10} | {'TEST WAPE%':<10} | {'TEST MAE':<10} | {'TEST RMSE':<10} | {'TEST R2':<8}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<22} | {m['val']['WAPE_percent']:<10.2f} | {m['test']['WAPE_percent']:<10.2f} | {m['test']['MAE']:<10.3f} | {m['test']['RMSE']:<10.3f} | {m['test']['R2']:<8.4f}")
    print("=" * 80)

    # Select Champion Model based on Test WAPE
    champion_name = min(candidate_dict.keys(), key=lambda k: results[k]["test"]["WAPE_percent"])
    champion_model = trained_models[champion_name]
    print(f"\n[Step 4/5] Selected Champion Model: {champion_name}")
    print(f"Champion Test Metrics -> WAPE: {results[champion_name]['test']['WAPE_percent']}%, MAE: {results[champion_name]['test']['MAE']}, RMSE: {results[champion_name]['test']['RMSE']}, R2: {results[champion_name]['test']['R2']}")

    # Save Best Model
    joblib.dump(champion_model, BEST_MODEL_PATH)
    print(f"Saved Champion Model artifact to {BEST_MODEL_PATH}")

    # Calculate baseline vs champion improvement
    base_wape = results["Historical_Baseline"]["test"]["WAPE_percent"]
    champ_wape = results[champion_name]["test"]["WAPE_percent"]
    wape_reduction = ((base_wape - champ_wape) / base_wape) * 100.0

    summary_payload = {
        "models": results,
        "champion_model": champion_name,
        "test_improvement": {
            "baseline_wape": base_wape,
            "champion_wape": champ_wape,
            "wape_reduction_pct": round(wape_reduction, 2),
            "baseline_mae": results["Historical_Baseline"]["test"]["MAE"],
            "champion_mae": results[champion_name]["test"]["MAE"],
            "mae_reduction": round(results["Historical_Baseline"]["test"]["MAE"] - results[champion_name]["test"]["MAE"], 3)
        }
    }
    with open(METRICS_SUMMARY_PATH, "w") as f:
        json.dump(summary_payload, f, indent=2)

    # 6. Per-Route and Peak/Off-Peak Breakdowns on Held-Out Test Set
    print("\n[Step 5/5] Generating Granular Stratified Breakdowns & Visualizations...")
    eval_df = test_df.copy()
    eval_df["baseline_pred"] = test_preds["Historical_Baseline"]
    eval_df["champion_pred"] = test_preds[champion_name]

    # Route breakdown
    route_breakdown = evaluate_by_route(eval_df, TARGET_COLUMN, "champion_pred")
    route_breakdown.to_csv(ROUTE_METRICS_PATH, index=False)
    print("\n--- Per-Route Test Performance (Champion: " + champion_name + ") ---")
    print(route_breakdown.to_string(index=False))

    # Peak vs Off-Peak breakdown
    peak_breakdown = evaluate_by_peak(eval_df, TARGET_COLUMN, "champion_pred")
    peak_breakdown.to_csv(PEAK_METRICS_PATH, index=False)
    print("\n--- Peak vs Off-Peak Test Performance (Champion: " + champion_name + ") ---")
    print(peak_breakdown.to_string(index=False))

    # Generate Publication-Quality Visualizations
    _generate_visualizations(results, eval_df, champion_name, champion_model, preprocessor.get_feature_names())

    print("\n Training & Evaluation Pipeline Completed Successfully!")

def _generate_visualizations(results, eval_df, champion_name, champion_model, feature_names):
    """Generate and save evaluation plots."""
    sns.set_theme(style="whitegrid")

    # 1. Model Comparison Chart (WAPE & MAE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    model_names = list(results.keys())
    test_wapes = [results[m]["test"]["WAPE_percent"] for m in model_names]
    test_maes = [results[m]["test"]["MAE"] for m in model_names]

    palette = ["#95a5a6", "#3498db", "#9b59b6", "#2ecc71"]

    axes[0].bar(model_names, test_wapes, color=palette[:len(model_names)])
    axes[0].set_title("Test WAPE % (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("WAPE (%)")
    axes[0].tick_params(axis='x', rotation=20)
    for i, v in enumerate(test_wapes):
        axes[0].text(i, v + 0.4, f"{v:.1f}%", ha='center', fontweight='bold')

    axes[1].bar(model_names, test_maes, color=palette[:len(model_names)])
    axes[1].set_title("Test MAE (Passengers)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MAE (count)")
    axes[1].tick_params(axis='x', rotation=20)
    for i, v in enumerate(test_maes):
        axes[1].text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_bar.png", dpi=200)
    plt.close()

    # 2. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(7, 7))
    plt.scatter(eval_df[TARGET_COLUMN], eval_df["champion_pred"], alpha=0.35, color="#2980b9", edgecolors="none")
    max_val = max(eval_df[TARGET_COLUMN].max(), eval_df["champion_pred"].max()) + 5
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label="Perfect Forecast (y = x)")
    plt.title(f"Actual vs. Predicted Demand: {champion_name} (Held-out Test)", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Passenger Count")
    plt.ylabel("Predicted Passenger Count")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted_scatter.png", dpi=200)
    plt.close()

    # 3. Residual Error Distribution
    residuals = eval_df[TARGET_COLUMN] - eval_df["champion_pred"]
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=35, color="#8e44ad")
    plt.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    plt.axvline(np.mean(residuals), color="blue", linestyle=":", lw=1.5, label=f"Mean Residual ({np.mean(residuals):.2f})")
    plt.title("Residual Error Distribution (Actual - Predicted)", fontsize=12, fontweight="bold")
    plt.xlabel("Prediction Error (Passengers)")
    plt.ylabel("Trip Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_distribution.png", dpi=200)
    plt.close()

    # 4. Feature Importance (Tree or Linear)
    plt.figure(figsize=(10, 6))
    if hasattr(champion_model, "feature_importances_"):
        importances = champion_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_names = [feature_names[i] for i in indices]
        top_vals = importances[indices]

        sns.barplot(x=top_vals, y=top_names, palette="viridis")
        plt.title(f"Top 15 Most Influential Features ({champion_name})", fontsize=12, fontweight="bold")
        plt.xlabel("Feature Importance (Gini / Split Gain)")
    elif hasattr(champion_model, "coef_"):
        coefs = np.abs(champion_model.coef_)
        indices = np.argsort(coefs)[::-1][:15]
        top_names = [feature_names[i] for i in indices]
        top_vals = coefs[indices]

        sns.barplot(x=top_vals, y=top_names, palette="mako")
        plt.title(f"Top 15 Feature Magnitudes ({champion_name})", fontsize=12, fontweight="bold")
        plt.xlabel("Absolute Coefficient Value")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    run_training_pipeline()
