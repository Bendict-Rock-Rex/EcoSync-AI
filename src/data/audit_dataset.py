"""
EcoSync-AI: Dataset Audit & Preprocessing Pipeline
Reads cooling july_october.xlsx (or processed parquet), analyzes sensor integrity,
generates data quality metrics, handles 'Null' strings and physics boundaries,
and extracts engineered features.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure config is importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings


def run_rigorous_audit(save_reports=True):
    """
    Performs comprehensive audit of the raw dataset:
    - Shape, sampling interval, missingness
    - Electrical sanity (Power Factor, Active Power, Voltage)
    - Thermal balance (Chiller COP, Thermal Power)
    - Setpoint variations & Fan coil dynamics
    """
    print("--- [EcoSync-AI Audit] Starting Rigorous Dataset Audit ---")
    
    # Check if cleaned parquet already exists for rapid reading
    if settings.PROCESSED_DATA_PATH.exists():
        print(f"Loading cached clean data from {settings.PROCESSED_DATA_PATH}...")
        df = pd.read_parquet(settings.PROCESSED_DATA_PATH)
    else:
        print(f"Loading raw workbook from {settings.RAW_DATA_PATH} (skiprows=94)...")
        df = pd.read_excel(settings.RAW_DATA_PATH, skiprows=94)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime').reset_index(drop=True)
        for c in df.columns:
            if c != 'DateTime':
                df[c] = pd.to_numeric(df[c].replace('Null', np.nan), errors='coerce')
        os.makedirs(settings.PROCESSED_DATA_PATH.parent, exist_ok=True)
        df.to_parquet(settings.PROCESSED_DATA_PATH, index=False)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # 1. Temporal integrity
    time_diffs = df['DateTime'].diff().dropna()
    median_interval_min = (time_diffs.median().total_seconds()) / 60.0
    start_dt = str(df['DateTime'].min())
    end_dt = str(df['DateTime'].max())
    total_records = len(df)
    
    # Expected records for 10-min interval: 123 days * 144 = 17712
    # Actual is 17718 rows
    
    # 2. Electrical range & PF audit
    p1 = df[settings.CHILLER_TAGS["chiller1_active_power_w"]]
    p2 = df[settings.CHILLER_TAGS["chiller2_active_power_w"]]
    pf1 = df[settings.CHILLER_TAGS["chiller1_pf"]]
    pf2 = df[settings.CHILLER_TAGS["chiller2_pf"]]
    
    # 3. Thermal & COP
    q1 = df[settings.CHILLER_TAGS["chiller1_thermal_power_w"]]
    q2 = df[settings.CHILLER_TAGS["chiller2_thermal_power_w"]]
    
    active_mask1 = (p1 > 50) & (q1 > 50)
    cop1 = q1[active_mask1] / p1[active_mask1]
    
    active_mask2 = (p2 > 50) & (q2 > 50)
    cop2 = q2[active_mask2] / p2[active_mask2]
    
    # 4. Zone comfort
    z1 = df[settings.FANCOIL_TAGS["fc1_zone_temp"]]
    z2 = df[settings.FANCOIL_TAGS["fc2_zone_temp"]]
    
    audit_results = {
        "dataset_summary": {
            "total_records": int(total_records),
            "total_sensor_channels": int(len(df.columns)),
            "start_timestamp": start_dt,
            "end_timestamp": end_dt,
            "nominal_sampling_interval_min": float(median_interval_min),
        },
        "chiller1_statistics": {
            "active_power_max_w": float(p1.max()),
            "active_power_mean_w": float(p1.mean()),
            "active_power_p95_w": float(p1.quantile(0.95)),
            "thermal_power_mean_w": float(q1.mean()),
            "empirical_cop_median": float(cop1.median()) if len(cop1) > 0 else 0.0,
            "empirical_cop_mean": float(cop1.mean()) if len(cop1) > 0 else 0.0,
            "runtime_fraction": float((p1 > 50).mean()),
            "pf_min": float(pf1.min()),
            "pf_max": float(pf1.max()),
        },
        "chiller2_statistics": {
            "active_power_max_w": float(p2.max()),
            "active_power_mean_w": float(p2.mean()),
            "active_power_p95_w": float(p2.quantile(0.95)),
            "thermal_power_mean_w": float(q2.mean()),
            "empirical_cop_median": float(cop2.median()) if len(cop2) > 0 else 0.0,
            "empirical_cop_mean": float(cop2.mean()) if len(cop2) > 0 else 0.0,
            "runtime_fraction": float((p2 > 50).mean()),
            "pf_min": float(pf2.min()),
            "pf_max": float(pf2.max()),
        },
        "zone_temperatures": {
            "zone1_temp_min": float(z1.min()),
            "zone1_temp_mean": float(z1.mean()),
            "zone1_temp_max": float(z1.max()),
            "zone2_temp_min": float(z2.min()),
            "zone2_temp_mean": float(z2.mean()),
            "zone2_temp_max": float(z2.max()),
            "zone1_comfort_compliance_pct": float(((z1 >= 21.0) & (z1 <= 26.0)).mean() * 100),
            "zone2_comfort_compliance_pct": float(((z2 >= 21.0) & (z2 <= 26.0)).mean() * 100),
        },
        "null_statistics": {
            "total_null_entries": int(df.isnull().sum().sum()),
            "top_null_columns": {c: int(v) for c, v in df.isnull().sum().nlargest(5).items()}
        }
    }
    
    if save_reports:
        os.makedirs(settings.REPORTS_DIR, exist_ok=True)
        report_path = settings.REPORTS_DIR / "dataset_audit_report.json"
        with open(report_path, "w") as f:
            json.dump(audit_results, f, indent=2)
        print(f"Saved audit report to {report_path}")
        
    print("--- [EcoSync-AI Audit] Audit Completed Successfully ---")
    return audit_results, df


def build_feature_pipeline(df=None, save_parquet=True):
    """
    Cleans raw data, imputes sensor dropouts, constructs TOU tariff markers,
    temporal features, physics-informed lags, and saves engineered dataset.
    """
    print("--- [EcoSync-AI Feature Pipeline] Building Features ---")
    if df is None:
        if settings.PROCESSED_DATA_PATH.exists():
            df = pd.read_parquet(settings.PROCESSED_DATA_PATH)
        else:
            _, df = run_rigorous_audit()
            
    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    # 1. Forward-fill short null dropouts (up to 3 steps = 30 mins), then backward-fill
    df = df.ffill(limit=3).bfill(limit=3)
    
    # 2. Temporal cyclical features
    dt = df['DateTime']
    hour = dt.dt.hour + dt.dt.minute / 60.0
    day_of_week = dt.dt.dayofweek
    is_weekend = (day_of_week >= 5).astype(int)
    
    df['hour'] = hour
    df['day_of_week'] = day_of_week
    df['is_weekend'] = is_weekend
    df['sin_hour'] = np.sin(2 * np.pi * hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * hour / 24.0)
    df['sin_day'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['cos_day'] = np.cos(2 * np.pi * day_of_week / 7.0)
    
    # 3. Dynamic TOU Pricing Signal (€/kWh)
    # Peak (10-14, 18-22 on weekdays): €0.24
    # Valley (00-08 weekdays, all weekend): €0.08
    # Flat: €0.15
    def compute_tariff(row):
        if row['is_weekend'] == 1:
            return settings.TARIFF_VALLEY_EUR_PER_KWH
        h = row['hour']
        if (10.0 <= h < 14.0) or (18.0 <= h < 22.0):
            return settings.TARIFF_PEAK_EUR_PER_KWH
        elif 0.0 <= h < 8.0:
            return settings.TARIFF_VALLEY_EUR_PER_KWH
        else:
            return settings.TARIFF_FLAT_EUR_PER_KWH

    df['tariff_eur_per_kwh'] = df.apply(compute_tariff, axis=1)
    
    # 4. Total Plant Electrical Power & Total Plant Thermal Power
    p1 = df[settings.CHILLER_TAGS["chiller1_active_power_w"]]
    p2 = df[settings.CHILLER_TAGS["chiller2_active_power_w"]]
    q1 = df[settings.CHILLER_TAGS["chiller1_thermal_power_w"]]
    q2 = df[settings.CHILLER_TAGS["chiller2_thermal_power_w"]]
    fc1_q = df[settings.FANCOIL_TAGS["fc1_thermal_power_w"]]
    fc2_q = df[settings.FANCOIL_TAGS["fc2_thermal_power_w"]]
    
    df['total_chiller_power_w'] = p1 + p2
    df['total_chiller_power_kw'] = df['total_chiller_power_w'] / 1000.0
    df['total_thermal_power_w'] = q1 + q2
    df['total_fancoil_thermal_w'] = fc1_q + fc2_q
    
    # 5. Physics & System Lag Features (10-min, 30-min, 60-min lags)
    tout = df[settings.WEATHER_TAGS["outdoor_temp"]]
    solar = df[settings.WEATHER_TAGS["solar_radiation"]]
    z1 = df[settings.FANCOIL_TAGS["fc1_zone_temp"]]
    z2 = df[settings.FANCOIL_TAGS["fc2_zone_temp"]]
    
    df['outdoor_temp_lag1'] = tout.shift(1).bfill()
    df['outdoor_temp_lag3'] = tout.shift(3).bfill()
    df['solar_rad_lag1'] = solar.shift(1).bfill()
    df['zone1_temp_lag1'] = z1.shift(1).bfill()
    df['zone2_temp_lag1'] = z2.shift(1).bfill()
    df['total_chiller_power_lag1'] = df['total_chiller_power_w'].shift(1).bfill()
    df['total_chiller_power_lag3'] = df['total_chiller_power_w'].shift(3).bfill()
    
    # 6. Thermal COP (instantaneous with physical clipping)
    chiller_active = df['total_chiller_power_w'] > 100.0
    instant_cop = np.where(chiller_active, df['total_thermal_power_w'] / df['total_chiller_power_w'], 0.0)
    df['system_cop'] = np.clip(instant_cop, 0.0, 10.0)
    
    # 7. Energy Cost per 10-min interval: Power(kW) * (10/60 h) * Price(€/kWh)
    df['cost_eur_10min'] = df['total_chiller_power_kw'] * (10.0 / 60.0) * df['tariff_eur_per_kwh']
    df['co2_kg_10min'] = df['total_chiller_power_kw'] * (10.0 / 60.0) * settings.GRID_EMISSION_FACTOR_KG_CO2_PER_KWH

    if save_parquet:
        os.makedirs(settings.FEATURE_DATA_PATH.parent, exist_ok=True)
        df.to_parquet(settings.FEATURE_DATA_PATH, index=False)
        print(f"Saved feature dataset with shape {df.shape} to {settings.FEATURE_DATA_PATH}")

    print("--- [EcoSync-AI Feature Pipeline] Completed Successfully ---")
    return df


if __name__ == "__main__":
    audit, raw_df = run_rigorous_audit()
    feat_df = build_feature_pipeline(raw_df)
    print(f"Features created: {len(feat_df.columns)} columns across {len(feat_df)} rows.")
