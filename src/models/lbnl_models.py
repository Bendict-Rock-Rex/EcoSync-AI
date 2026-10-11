"""
EcoSync-AI: LBNL Building 59 Surrogate Models & Occupancy Ablation Study
Trains and evaluates:
1. Baseline Model (WITHOUT Occupancy): Weather + Diurnal + Lag Power -> HVAC kW
2. Occupancy-Aware Model (WITH Occupancy): Weather + Diurnal + Lag Power + Occupant Count -> HVAC kW
Computes rigorous chronological test set comparison (80% train, 20% test).
Serializes models and generates ablation report.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import settings
from src.data.lbnl_loader import LBNLLoader


def run_occupancy_ablation_study():
    print("--- [EcoSync-AI LBNL Modeling] Running Occupancy Ablation Study ---")
    loader = LBNLLoader()
    df = loader.load_feature_data()
    
    target_col = "hvac_total_kw"
    
    # Common base features (No occupancy)
    base_features = [
        "air_temp_set_1",
        "solar_radiation_set_1",
        "relative_humidity_set_1",
        "hour",
        "sin_hour",
        "cos_hour",
        "is_weekend",
        "outdoor_temp_lag1",
        "outdoor_temp_lag3",
        "hvac_power_lag1"
    ]
    
    # Occupancy-aware feature set (+ Occupant count and lag)
    occ_features = base_features + ["occ_total", "occ_lag1"]
    
    # Drop rows with NaN
    sub_df = df[occ_features + [target_col]].dropna()
    
    # Chronological Split (80% train, 20% test) to prevent temporal data leakage
    split_idx = int(len(sub_df) * 0.8)
    train_df = sub_df.iloc[:split_idx]
    test_df = sub_df.iloc[split_idx:]
    
    print(f"Dataset Split: {len(train_df)} train samples, {len(test_df)} test samples.")
    print(f"Train Period: {df['DateTime'].iloc[0]} to {df['DateTime'].iloc[split_idx]}")
    print(f"Test Period: {df['DateTime'].iloc[split_idx]} to {df['DateTime'].iloc[-1]}")
    
    # 1. Train Model WITHOUT Occupancy (Baseline)
    X_train_base = train_df[base_features]
    y_train = train_df[target_col]
    X_test_base = test_df[base_features]
    y_test = test_df[target_col]
    
    model_no_occ = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    model_no_occ.fit(X_train_base, y_train)
    
    pred_no_occ = np.clip(model_no_occ.predict(X_test_base), 0, None)
    r2_no_occ = float(r2_score(y_test, pred_no_occ))
    mae_no_occ = float(mean_absolute_error(y_test, pred_no_occ))
    rmse_no_occ = float(np.sqrt(mean_squared_error(y_test, pred_no_occ)))
    
    print("\n--- Model A: WITHOUT Occupancy (Baseline) ---")
    print(f"R2: {r2_no_occ:.4f} | MAE: {mae_no_occ:.2f} kW | RMSE: {rmse_no_occ:.2f} kW")
    
    # 2. Train Model WITH Occupancy
    X_train_occ = train_df[occ_features]
    X_test_occ = test_df[occ_features]
    
    model_with_occ = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    model_with_occ.fit(X_train_occ, y_train)
    
    pred_with_occ = np.clip(model_with_occ.predict(X_test_occ), 0, None)
    r2_with_occ = float(r2_score(y_test, pred_with_occ))
    mae_with_occ = float(mean_absolute_error(y_test, pred_with_occ))
    rmse_with_occ = float(np.sqrt(mean_squared_error(y_test, pred_with_occ)))
    
    print("\n--- Model B: WITH Camera Occupant Count ---")
    print(f"R2: {r2_with_occ:.4f} | MAE: {mae_with_occ:.2f} kW | RMSE: {rmse_with_occ:.2f} kW")
    
    # 3. Calculate Performance Delta
    mae_improvement_pct = ((mae_no_occ - mae_with_occ) / mae_no_occ) * 100.0
    rmse_improvement_pct = ((rmse_no_occ - rmse_with_occ) / rmse_no_occ) * 100.0
    r2_delta = r2_with_occ - r2_no_occ
    
    print("\n--- Occupancy Value Assessment ---")
    print(f"MAE Improvement: {mae_improvement_pct:+.2f}%")
    print(f"RMSE Improvement: {rmse_improvement_pct:+.2f}%")
    print(f"R2 Delta: {r2_delta:+.4f}")
    
    # Feature importances of Occupancy-Aware Model
    occ_importances = dict(zip(occ_features, [float(x) for x in model_with_occ.feature_importances_]))
    
    # Save Models
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    save_path = settings.MODELS_DIR / "lbnl_models.joblib"
    joblib.dump({
        "model_no_occ": model_no_occ,
        "model_with_occ": model_with_occ,
        "base_features": base_features,
        "occ_features": occ_features,
        "target": target_col,
        "metrics_no_occ": {"r2": r2_no_occ, "mae_kw": mae_no_occ, "rmse_kw": rmse_no_occ},
        "metrics_with_occ": {"r2": r2_with_occ, "mae_kw": mae_with_occ, "rmse_kw": rmse_with_occ},
        "ablation_comparison": {
            "mae_reduction_pct": mae_improvement_pct,
            "rmse_reduction_pct": rmse_improvement_pct,
            "r2_delta": r2_delta,
            "occupancy_importance_pct": round(occ_importances.get("occ_total", 0.0) * 100, 2)
        }
    }, save_path)
    print(f"Saved LBNL models to {save_path}")
    
    # Write JSON summary report
    ablation_report = {
        "dataset": "LBNL_BUILDING_59",
        "target_variable": target_col,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "model_without_occupancy": {
            "features": base_features,
            "r2_score": r2_no_occ,
            "mae_kw": mae_no_occ,
            "rmse_kw": rmse_no_occ
        },
        "model_with_occupancy": {
            "features": occ_features,
            "r2_score": r2_with_occ,
            "mae_kw": mae_with_occ,
            "rmse_kw": rmse_with_occ
        },
        "ablation_findings": {
            "mae_improvement_pct": mae_improvement_pct,
            "rmse_improvement_pct": rmse_improvement_pct,
            "r2_gain": r2_delta,
            "feature_importances": occ_importances
        }
    }
    
    with open(settings.REPORTS_DIR / "lbnl_occupancy_ablation.json", "w") as f:
        json.dump(ablation_report, f, indent=2)
        
    return ablation_report


if __name__ == "__main__":
    run_occupancy_ablation_study()
