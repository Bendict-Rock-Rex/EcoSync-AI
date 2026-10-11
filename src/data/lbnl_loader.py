"""
EcoSync-AI: LBNL Building 59 Data Loader & Preprocessing Pipeline
Reads Building_59.zip streams (ele.csv, occ.csv, site_weather.csv, wifi.csv),
performs resampling, temporal alignment, occupancy aggregation, and caches clean Parquet files.
Maintains absolute data isolation from KUBIK.
"""

import os
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import settings


class LBNLLoader:
    def __init__(self):
        self.dataset_id = "LBNL_BUILDING_59"
        self.zip_path = settings.LBNL_ZIP_PATH
        self.processed_path = settings.LBNL_PROCESSED_PATH
        self.feature_path = settings.LBNL_FEATURE_PATH

    def process_and_cache(self, force_recompute=False) -> pd.DataFrame:
        """
        Extracts and aligns 15-minute electrical, occupancy, and weather records from Building_59.zip.
        Window: 2018-05-22 to 2019-02-21 (Camera Occupancy Active Window).
        """
        if self.feature_path.exists() and not force_recompute:
            print(f"Loading cached LBNL feature dataset from {self.feature_path}...")
            df = pd.read_parquet(self.feature_path)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            return df

        if not self.zip_path.exists():
            raise FileNotFoundError(f"LBNL zip archive not found at {self.zip_path}")

        print("--- [LBNL Loader] Extracting and processing Building 59 streams ---")
        with zipfile.ZipFile(self.zip_path, "r") as z:
            # 1. Electrical Sub-Meters (15-min cadence)
            print("Reading ele.csv...")
            with z.open("Bldg59_clean data/ele.csv") as f:
                ele = pd.read_csv(f)
                ele = ele.loc[:, ~ele.columns.str.contains("^Unnamed")]
                ele["DateTime"] = pd.to_datetime(ele["date"])
                ele = ele.drop(columns=["date"]).sort_values("DateTime").set_index("DateTime")

            # 2. Camera Occupancy (1-min cadence -> Resample to 15-min mean and max)
            print("Reading occ.csv...")
            with z.open("Bldg59_clean data/occ.csv") as f:
                occ = pd.read_csv(f)
                occ["DateTime"] = pd.to_datetime(occ["date"])
                occ = occ.drop(columns=["date"]).sort_values("DateTime").set_index("DateTime")
                occ_15m = occ.resample("15min").agg({
                    "occ_third_south": "mean",
                    "occ_fourth_south": "mean"
                })

            # 3. Weather Telemetry (15-min cadence)
            print("Reading site_weather.csv...")
            with z.open("Bldg59_clean data/site_weather.csv") as f:
                weather = pd.read_csv(f)
                weather["DateTime"] = pd.to_datetime(weather["date"])
                weather = weather.drop(columns=["date"]).sort_values("DateTime").set_index("DateTime")

            # 4. Wi-Fi Device Counts (10-min cadence -> Resample to 15-min mean)
            print("Reading wifi.csv...")
            with z.open("Bldg59_clean data/wifi.csv") as f:
                wifi = pd.read_csv(f)
                wifi["DateTime"] = pd.to_datetime(wifi["date"])
                wifi = wifi.drop(columns=["date"]).sort_values("DateTime").set_index("DateTime")
                wifi_15m = wifi.resample("15min").mean()

        # Join across overlapping interval
        print("Merging aligned streams...")
        merged = ele.join(occ_15m, how="inner").join(weather, how="inner").join(wifi_15m, how="left")
        
        # Reset index to column
        df = merged.reset_index()
        
        # Clean missing values using forward fill up to 30 min (2 steps), then backward fill
        df = df.ffill(limit=2).bfill(limit=2)
        
        # Compute aggregate power & occupancy variables
        df["hvac_total_kw"] = df["hvac_N"].fillna(0) + df["hvac_S"].fillna(0)
        df["mels_total_kw"] = df["mels_N"].fillna(0) + df["mels_S"].fillna(0)
        df["lig_total_kw"] = df["lig_S"].fillna(0)
        df["whole_building_kw"] = df["hvac_total_kw"] + df["mels_total_kw"] + df["lig_total_kw"]
        
        df["occ_total"] = df["occ_third_south"].fillna(0) + df["occ_fourth_south"].fillna(0)
        df["wifi_total"] = (
            df["wifi_first_south"].fillna(0) +
            df["wifi_second_south"].fillna(0) +
            df["wifi_third_south"].fillna(0) +
            df["wifi_fourth_south"].fillna(0)
        )

        # Feature Engineering: Temporal Cyclical Features
        dt = df["DateTime"]
        hour = dt.dt.hour + dt.dt.minute / 60.0
        day_of_week = dt.dt.dayofweek
        df["hour"] = hour
        df["day_of_week"] = day_of_week
        df["is_weekend"] = (day_of_week >= 5).astype(int)
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
        df["sin_day"] = np.sin(2 * np.pi * day_of_week / 7.0)
        df["cos_day"] = np.cos(2 * np.pi * day_of_week / 7.0)

        # Lags for physics and thermal inertia (15m, 45m lags)
        df["outdoor_temp_lag1"] = df["air_temp_set_1"].shift(1).bfill()
        df["outdoor_temp_lag3"] = df["air_temp_set_1"].shift(3).bfill()
        df["solar_rad_lag1"] = df["solar_radiation_set_1"].shift(1).bfill()
        df["hvac_power_lag1"] = df["hvac_total_kw"].shift(1).bfill()
        df["occ_lag1"] = df["occ_total"].shift(1).bfill()

        # Save clean and feature parquets
        os.makedirs(self.processed_path.parent, exist_ok=True)
        df.to_parquet(self.processed_path, index=False)
        df.to_parquet(self.feature_path, index=False)
        print(f"--- [LBNL Loader] Saved {len(df)} records to {self.feature_path} ---")
        return df

    def load_feature_data(self) -> pd.DataFrame:
        if self.feature_path.exists():
            df = pd.read_parquet(self.feature_path)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            return df
        if self.zip_path.exists():
            return self.process_and_cache()
        raise FileNotFoundError(
            f"LBNL feature dataset not found at {self.feature_path} and raw archive {self.zip_path} is absent. "
            "Ensure data/processed/lbnl_features.parquet is committed to the repository."
        )


if __name__ == "__main__":
    loader = LBNLLoader()
    df = loader.process_and_cache(force_recompute=True)
    print("Columns:", df.columns.tolist()[:15])
    print("Shape:", df.shape)
    print("Date min/max:", df["DateTime"].min(), "to", df["DateTime"].max())
