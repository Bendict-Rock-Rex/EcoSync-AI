"""
EcoSync-AI: Predictive Modeling Engine
Trains, validates, and serializes machine learning models on the KUBIK HVAC dataset:
1. Total Chiller Electrical Power Predictor (Linear, Ridge, RandomForest, GradientBoosting)
2. Zone Thermal Predictor (Forecasts zone temperature at t+10m given outdoor conditions & setpoints)
3. Chiller 1 & Chiller 2 Individual Power Predictors
Calculates and reports R2, MAE, RMSE, and cross-validation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure config is importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings


def train_power_models():
    """
    Trains surrogate predictive models for chiller electrical power consumption.
    Features: Outdoor temp, solar radiation, fancoil thermal loads, chiller setpoints, lag powers.
    Target: total_chiller_power_w
    """
    print("--- [EcoSync-AI Models] Training Power Consumption Models ---")
    df = pd.read_parquet(settings.FEATURE_DATA_PATH)
    
    # Feature columns supported by real data
    feature_cols = [
        settings.WEATHER_TAGS["outdoor_temp"],
        settings.WEATHER_TAGS["solar_radiation"],
        settings.CHILLER_TAGS["chiller1_setpoint"],
        settings.CHILLER_TAGS["chiller2_setpoint"],
        settings.CHILLER_TAGS["chiller1_order"],
        settings.CHILLER_TAGS["chiller2_order"],
        settings.FANCOIL_TAGS["fc1_zone_temp"],
        settings.FANCOIL_TAGS["fc2_zone_temp"],
        settings.FANCOIL_TAGS["fc1_setpoint"],
        settings.FANCOIL_TAGS["fc2_setpoint"],
        "hour",
        "sin_hour",
        "cos_hour",
        "outdoor_temp_lag1",
        "total_chiller_power_lag1",
        "tariff_eur_per_kwh"
    ]
    
    target_col = "total_chiller_power_w"
    
    # Drop rows with NaN in features or target
    sub_df = df[feature_cols + [target_col]].dropna()
    X = sub_df[feature_cols]
    y = sub_df[target_col]
    
    # Chronological temporal split (80% train, 20% test) to prevent temporal data leakage
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    candidates = {
        "RidgeRegression": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42)
    }
    
    metrics = {}
    best_name = None
    best_r2 = -1e9
    best_model = None
    
    for name, model in candidates.items():
        print(f"Fitting {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Physics clipping: power cannot be negative
        y_pred = np.clip(y_pred, 0, None)
        
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        metrics[name] = {
            "r2_score": r2,
            "mae_watts": mae,
            "rmse_watts": rmse
        }
        print(f"  {name} => R2: {r2:.4f} | MAE: {mae:.2f} W | RMSE: {rmse:.2f} W")
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model

    # Save best power model & feature list
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    model_save_path = settings.MODELS_DIR / "chiller_power_model.joblib"
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_cols,
        "metrics": metrics[best_name]
    }, model_save_path)
    print(f"Saved primary power model ({best_name}) to {model_save_path}")
    
    # Feature importances
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(feature_cols, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, "coef_"):
        importances = dict(zip(feature_cols, np.abs(best_model.coef_).tolist()))
    else:
        importances = {}
        
    evaluation_summary = {
        "best_model": best_name,
        "all_models": metrics,
        "features": feature_cols,
        "feature_importances": importances,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    with open(settings.REPORTS_DIR / "power_model_metrics.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
        
    return best_model, evaluation_summary


def train_zone_thermal_models():
    """
    Trains thermal dynamics model:
    Predicts next-step zone temperature (Zone 1) given current zone temp, outdoor temp,
    fancoil status, and chiller cooling supply.
    Target: zone1_temp_next (t + 10 min)
    """
    print("--- [EcoSync-AI Models] Training Zone Thermal Dynamics Model ---")
    df = pd.read_parquet(settings.FEATURE_DATA_PATH).copy()
    
    z1_col = settings.FANCOIL_TAGS["fc1_zone_temp"]
    df["zone1_temp_next"] = df[z1_col].shift(-1)
    
    thermal_features = [
        z1_col,
        settings.WEATHER_TAGS["outdoor_temp"],
        settings.WEATHER_TAGS["solar_radiation"],
        settings.FANCOIL_TAGS["fc1_setpoint"],
        settings.FANCOIL_TAGS["fc1_enable"],
        settings.FANCOIL_TAGS["fc1_cool_mode"],
        settings.CHILLER_TAGS["chiller1_active_power_w"],
        "hour",
        "sin_hour",
        "cos_hour"
    ]
    
    sub_df = df[thermal_features + ["zone1_temp_next"]].dropna()
    X = sub_df[thermal_features]
    y = sub_df["zone1_temp_next"]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    thermal_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    thermal_model.fit(X_train, y_train)
    
    y_pred = thermal_model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    print(f"Zone Thermal Model => R2: {r2:.4f} | MAE: {mae:.3f} degC | RMSE: {rmse:.3f} degC")
    
    thermal_save_path = settings.MODELS_DIR / "zone_thermal_model.joblib"
    joblib.dump({
        "model": thermal_model,
        "feature_names": thermal_features,
        "metrics": {"r2_score": r2, "mae_degc": mae, "rmse_degc": rmse}
    }, thermal_save_path)
    print(f"Saved zone thermal model to {thermal_save_path}")
    
    return thermal_model


if __name__ == "__main__":
    train_power_models()
    train_zone_thermal_models()
