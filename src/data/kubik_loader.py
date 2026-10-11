"""
EcoSync-AI: KUBIK Data Loader
Dedicated, modular loader for KUBIK/HOLISDER experimental HVAC telemetry.
Preserves existing functionality and returns standardized pandas DataFrames.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import settings
from src.data.audit_dataset import run_rigorous_audit, build_feature_pipeline


class KubikLoader:
    def __init__(self):
        self.dataset_id = "KUBIK_HOLISDER"

    def load_clean_data(self) -> pd.DataFrame:
        if settings.PROCESSED_DATA_PATH.exists():
            df = pd.read_parquet(settings.PROCESSED_DATA_PATH)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            return df
        if settings.RAW_DATA_PATH.exists():
            run_rigorous_audit()
            df = pd.read_parquet(settings.PROCESSED_DATA_PATH)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            return df
        raise FileNotFoundError(
            f"KUBIK dataset not found at {settings.PROCESSED_DATA_PATH} or {settings.RAW_DATA_PATH}. "
            "Ensure data/processed/kubik_features.parquet is committed to the repository."
        )

    def load_feature_data(self) -> pd.DataFrame:
        if settings.FEATURE_DATA_PATH.exists():
            df = pd.read_parquet(settings.FEATURE_DATA_PATH)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            return df
        if settings.RAW_DATA_PATH.exists():
            build_feature_pipeline()
            df = pd.read_parquet(settings.FEATURE_DATA_PATH)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            return df
        raise FileNotFoundError(
            f"KUBIK feature dataset not found at {settings.FEATURE_DATA_PATH} or {settings.RAW_DATA_PATH}. "
            "Ensure data/processed/kubik_features.parquet is committed to the repository."
        )


if __name__ == "__main__":
    loader = KubikLoader()
    df = loader.load_feature_data()
    print(f"KubikLoader loaded {len(df)} rows across {len(df.columns)} columns.")
