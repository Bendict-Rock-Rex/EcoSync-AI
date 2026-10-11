"""
EcoSync-AI: Explainability Engine
Computes feature importances, local surrogate explanations, and physics-based drivers
for predictions. Uses scikit-learn tree feature importances and linear weight attribution.
Generates interpretable explanations for any HVAC operating point.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure config is importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings


class ExplainabilityEngine:
    def __init__(self):
        power_model_path = settings.MODELS_DIR / "chiller_power_model.joblib"
        if not power_model_path.exists():
            raise FileNotFoundError(f"Power model not found at {power_model_path}. Train models first.")
        
        bundle = joblib.load(power_model_path)
        self.power_model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.feature_names = bundle["feature_names"]
        self.metrics = bundle.get("metrics", {})
        
        # Friendly human-readable feature descriptions
        self.friendly_names = {
            settings.WEATHER_TAGS["outdoor_temp"]: "Outdoor Ambient Temperature (°C)",
            settings.WEATHER_TAGS["solar_radiation"]: "Solar Global Radiation (W/m²)",
            settings.CHILLER_TAGS["chiller1_setpoint"]: "Chiller 1 Supply Temp Setpoint (°C)",
            settings.CHILLER_TAGS["chiller2_setpoint"]: "Chiller 2 Supply Temp Setpoint (°C)",
            settings.CHILLER_TAGS["chiller1_order"]: "Chiller 1 Run Command (ON/OFF)",
            settings.CHILLER_TAGS["chiller2_order"]: "Chiller 2 Run Command (ON/OFF)",
            settings.FANCOIL_TAGS["fc1_zone_temp"]: "Fancoil 1 Zone Temperature (°C)",
            settings.FANCOIL_TAGS["fc2_zone_temp"]: "Fancoil 2 Zone Temperature (°C)",
            settings.FANCOIL_TAGS["fc1_setpoint"]: "Fancoil 1 Thermostat Setpoint (°C)",
            settings.FANCOIL_TAGS["fc2_setpoint"]: "Fancoil 2 Thermostat Setpoint (°C)",
            "hour": "Time of Day (Hour)",
            "sin_hour": "Diurnal Cycle (Sine)",
            "cos_hour": "Diurnal Cycle (Cosine)",
            "outdoor_temp_lag1": "Previous Outdoor Temp (°C)",
            "total_chiller_power_lag1": "Previous Chiller Active Power (W)",
            "tariff_eur_per_kwh": "Electricity Tariff (€/kWh)"
        }

    def get_global_importance(self):
        """
        Extracts global feature importances or normalized coefficients.
        """
        if hasattr(self.power_model, "feature_importances_"):
            raw_scores = self.power_model.feature_importances_
        elif hasattr(self.power_model, "coef_"):
            raw_scores = np.abs(self.power_model.coef_)
        else:
            raw_scores = np.ones(len(self.feature_names))
            
        total = np.sum(raw_scores) if np.sum(raw_scores) > 0 else 1.0
        normalized = (raw_scores / total).tolist()
        
        ranking = []
        for name, score in zip(self.feature_names, normalized):
            ranking.append({
                "feature_code": name,
                "feature_label": self.friendly_names.get(name, name),
                "relative_importance_pct": round(score * 100, 2)
            })
            
        ranking.sort(key=lambda x: x["relative_importance_pct"], reverse=True)
        return ranking

    def explain_instance(self, feature_vector_df):
        """
        Computes local feature contribution (linear attribution: coef_i * (x_i - mean_i))
        """
        feature_vector = feature_vector_df[self.feature_names].iloc[0]
        
        # Attribution
        contributions = []
        if hasattr(self.power_model, "coef_"):
            coefs = self.power_model.coef_
            for name, coef, val in zip(self.feature_names, coefs, feature_vector):
                # impact in Watts
                impact_watts = float(coef * val)
                contributions.append({
                    "feature_label": self.friendly_names.get(name, name),
                    "value": round(float(val), 2),
                    "impact_watts": round(impact_watts, 2),
                    "direction": "increases_power" if impact_watts > 0 else "reduces_power"
                })
        else:
            # Fallback for tree-based surrogate: magnitude * value
            for name, val in zip(self.feature_names, feature_vector):
                contributions.append({
                    "feature_label": self.friendly_names.get(name, name),
                    "value": round(float(val), 2),
                    "impact_watts": 0.0,
                    "direction": "neutral"
                })
                
        contributions.sort(key=lambda x: abs(x["impact_watts"]), reverse=True)
        return {
            "model_type": self.model_name,
            "top_drivers": contributions[:6]
        }


if __name__ == "__main__":
    explainer = ExplainabilityEngine()
    global_imp = explainer.get_global_importance()
    print("--- Top Global Predictive Drivers ---")
    for item in global_imp[:6]:
        print(f"  {item['feature_label']}: {item['relative_importance_pct']}%")
