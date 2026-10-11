"""
EcoSync-AI System Configuration
All thresholds, tag definitions, safety limits, and hyperparameters.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "cooling july_october.xlsx"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "kubik_10min_cleaned.parquet"
FEATURE_DATA_PATH = BASE_DIR / "data" / "processed" / "kubik_features.parquet"

# LBNL Building 59 Paths
LBNL_RAW_DIR = BASE_DIR / "doi_10_7941_D1N33Q__v20220202"
LBNL_ZIP_PATH = LBNL_RAW_DIR / "Building_59.zip"
LBNL_PROCESSED_PATH = BASE_DIR / "data" / "processed" / "lbnl_15min_cleaned.parquet"
LBNL_FEATURE_PATH = BASE_DIR / "data" / "processed" / "lbnl_features.parquet"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Core Sensor Tags in KUBIK Dataset
TIMESTAMP_COL = "DateTime"

WEATHER_TAGS = {
    "solar_radiation": "METEO_RoofSR_Global.PV",      # W/m2 Global horizontal radiation
    "outdoor_temp": "METEO_GardenWS_AT.PV",          # degC Outdoor dry bulb temperature
}

FANCOIL_TAGS = {
    "fc1_enable": "F0N2_FCM1.Enable",
    "fc1_zone_temp": "F0N2_FCM1.PV",
    "fc1_setpoint": "F0N2_FCM1.SP",
    "fc1_cool_mode": "F0N2_FCM1.ST_COOL_Mode",
    "fc1_thermal_power_w": "F0N2_FCM1_W5.W",
    "fc1_flow": "F0N2_FCM1_C5.PV",
    "fc1_inlet_t": "F0N2_FCM1_ST1.PV",
    "fc1_outlet_t": "F0N2_FCM1_ST2.PV",

    "fc2_enable": "F0N2_FCM2.Enable",
    "fc2_zone_temp": "F0N2_FCM2.PV",
    "fc2_setpoint": "F0N2_FCM2.SP",
    "fc2_cool_mode": "F0N2_FCM2.ST_COOL_Mode",
    "fc2_thermal_power_w": "F0N2_FCM2_W6.W",
    "fc2_flow": "F0N2_FCM2_C6.PV",
    "fc2_inlet_t": "F0N2_FCM2_ST1.PV",
    "fc2_outlet_t": "F0N2_FCM2_ST2.PV",
}

CHILLER_TAGS = {
    "chiller1_active_power_w": "HVAC_ENF1_CVM.W",
    "chiller1_order": "HVAC_ENF1_m.OUT",
    "chiller1_setpoint": "HVAC_Enfriadora1.SP",
    "chiller1_pf": "HVAC_ENF1_CVM.PF_III",
    "chiller1_apparent_va": "HVAC_ENF1_CVM.VA",
    "chiller1_reactive_var": "HVAC_ENF1_CVM.VAr",
    "chiller1_thermal_power_w": "HVAC_WP301.W",
    "chiller1_water_outlet": "HVAC_SP301.PV",
    "chiller1_water_inlet": "HVAC_SP302.PV",
    "chiller1_flow": "HVAC_CP301.PV",
    "chiller1_pump_order": "HVAC_BP301.OUT",

    "chiller2_active_power_w": "HVAC_ENF2_CVM.W",
    "chiller2_order": "HVAC_ENF2_m.OUT",
    "chiller2_setpoint": "HVAC_Enfriadora2.SP",
    "chiller2_pf": "HVAC_ENF2_CVM.PF_III",
    "chiller2_apparent_va": "HVAC_ENF2_CVM.VA",
    "chiller2_reactive_var": "HVAC_ENF2_CVM.VAr",
    "chiller2_thermal_power_w": "HVAC_WP401.W",
    "chiller2_water_outlet": "HVAC_SP401.PV",
    "chiller2_water_inlet": "HVAC_SP402.PV",
    "chiller2_flow": "HVAC_CP401.PV",
    "chiller2_pump_order": "HVAC_BP401.OUT",
}

BASEMENT_TEMP_TAGS = [
    "HVAC_STA.PV",
    "HVAC_STB.PV",
    "HVAC_STC.PV",
    "HVAC_STD.PV"
]

# Physical and Comfort Safety Thresholds
SAFETY_BOUNDS = {
    "zone_temp_min": 21.0,           # deg C lower comfort bound
    "zone_temp_max": 26.0,           # deg C upper comfort bound
    "chiller_sp_min": 7.0,           # deg C minimum chilled water supply setpoint
    "chiller_sp_max": 14.0,          # deg C maximum chilled water supply setpoint
    "chiller_power_max_w": 3500.0,   # W maximum allowable active power draw
    "pf_warning_min": 0.70,          # Power factor warning limit
    "min_short_cycle_steps": 3,      # Minimum consecutive steps (30 min) to stay ON/OFF
}

# Tariffs & Emissions (Standard Spanish Industrial TOU Rates for summer 2020)
# Peak: 10:00 - 14:00 and 18:00 - 22:00 weekdays (€0.24/kWh)
# Flat: 08:00 - 10:00, 14:00 - 18:00, 22:00 - 24:00 weekdays (€0.15/kWh)
# Valley: 00:00 - 08:00 weekdays & all weekend (€0.08/kWh)
TARIFF_PEAK_EUR_PER_KWH = 0.24
TARIFF_FLAT_EUR_PER_KWH = 0.15
TARIFF_VALLEY_EUR_PER_KWH = 0.08
GRID_EMISSION_FACTOR_KG_CO2_PER_KWH = 0.190  # Spain REE average factor
