"""
EcoSync-AI: Unified Feature Mapping & Schema Bridge
Provides safe harmonization between KUBIK and LBNL representations
WITHOUT ever combining their physical identities or fabricating absent variables.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class CanonicalFeatureSchema:
    """
    Standardized naming for common dimensions:
    - timestamp
    - ambient_temp_degc
    - solar_radiation_w_m2
    - hvac_electric_power_kw
    - occupancy_count (Optional, None if absent)
    - power_factor (Optional, None if absent)
    """

    @staticmethod
    def map_kubik_record(record: Dict[str, Any]) -> Dict[str, Any]:
        p1 = float(record.get("HVAC_ENF1_CVM.W", 0.0))
        p2 = float(record.get("HVAC_ENF2_CVM.W", 0.0))
        pf1 = record.get("HVAC_ENF1_CVM.PF_III", None)
        
        return {
            "dataset_id": "KUBIK_HOLISDER",
            "building_name": "KUBIK Experimental Facility",
            "timestamp": str(record.get("DateTime", "")),
            "ambient_temp_degc": float(record.get("METEO_GardenWS_AT.PV", np.nan)),
            "solar_radiation_w_m2": float(record.get("METEO_RoofSR_Global.PV", np.nan)),
            "hvac_electric_power_kw": (p1 + p2) / 1000.0,
            "occupancy_count": None,  # Explicitly None - NEVER imputed or fabricated
            "power_factor": abs(float(pf1)) if pf1 is not None and not np.isnan(float(pf1)) else None,
            "chilled_water_temp_sp_degc": float(record.get("HVAC_Enfriadora1.SP", np.nan)),
            "data_provenance": "KUBIK 10-min cooling experimental dataset (Derio, Spain)"
        }

    @staticmethod
    def map_lbnl_record(record: Dict[str, Any]) -> Dict[str, Any]:
        hvac_kw = float(record.get("hvac_total_kw", float(record.get("hvac_N", 0.0)) + float(record.get("hvac_S", 0.0))))
        occ = record.get("occ_total", None)
        
        return {
            "dataset_id": "LBNL_BUILDING_59",
            "building_name": "LBNL Building 59 Office",
            "timestamp": str(record.get("DateTime", "")),
            "ambient_temp_degc": float(record.get("air_temp_set_1", np.nan)),
            "solar_radiation_w_m2": float(record.get("solar_radiation_set_1", np.nan)),
            "hvac_electric_power_kw": hvac_kw,
            "occupancy_count": float(occ) if occ is not None and not np.isnan(float(occ)) else None,
            "power_factor": None,  # Explicitly None - LBNL has NO PF sensors
            "chilled_water_temp_sp_degc": None,  # Building 59 uses RTU air VAV, not water chiller loop
            "data_provenance": "LBNL Building 59 15-min clean operational dataset (Berkeley, CA)"
        }
