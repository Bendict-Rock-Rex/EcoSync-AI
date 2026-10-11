"""
EcoSync-AI: Dataset Registry & Common Capability Specification
Enforces distinct dataset identities, capabilities, feature mappings,
and prevents illegal cross-dataset contamination.
"""

from typing import Dict, List, Any
from pathlib import Path


class DatasetMetadata:
    def __init__(
        self,
        dataset_id: str,
        building_name: str,
        facility_type: str,
        location: str,
        climate_zone: str,
        period_str: str,
        cadence_min: float,
        supported_capabilities: List[str],
        unsupported_capabilities: Dict[str, str],
        primary_power_target: str,
        power_unit: str,
        has_occupancy: bool,
        has_power_quality: bool,
    ):
        self.dataset_id = dataset_id
        self.building_name = building_name
        self.facility_type = facility_type
        self.location = location
        self.climate_zone = climate_zone
        self.period_str = period_str
        self.cadence_min = cadence_min
        self.supported_capabilities = supported_capabilities
        self.unsupported_capabilities = unsupported_capabilities
        self.primary_power_target = primary_power_target
        self.power_unit = power_unit
        self.has_occupancy = has_occupancy
        self.has_power_quality = has_power_quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "building_name": self.building_name,
            "facility_type": self.facility_type,
            "location": self.location,
            "climate_zone": self.climate_zone,
            "period": self.period_str,
            "cadence_min": self.cadence_min,
            "supported_capabilities": self.supported_capabilities,
            "unsupported_capabilities": self.unsupported_capabilities,
            "primary_power_target": self.primary_power_target,
            "power_unit": self.power_unit,
            "has_occupancy": self.has_occupancy,
            "has_power_quality": self.has_power_quality,
        }


DATASET_REGISTRY = {
    "KUBIK_HOLISDER": DatasetMetadata(
        dataset_id="KUBIK_HOLISDER",
        building_name="KUBIK Experimental Building (Tecnalia)",
        facility_type="Full-Scale HVAC Experimental Test Facility",
        location="Derio, Basque Country, Spain",
        climate_zone="Cfb (Marine West Coast)",
        period_str="2020-07-01 to 2020-10-31 (4 Months)",
        cadence_min=10.0,
        supported_capabilities=[
            "Chiller Electrical Active Power Prediction",
            "Coupled Three-Phase Power Quality Verification (PF, VRS, VST, VTR)",
            "Chilled Water Circuit COP & Staging Diagnostics",
            "Supply Water Temperature Setpoint What-If Simulation",
            "Evidence-Gated Safety Checks & Asset Protection",
            "Time-of-Use Tariff Arbitrage & Cost Optimization (€/kWh)",
            "Scope 2 Carbon Footprint Tracking (kg CO2)",
        ],
        unsupported_capabilities={
            "Occupancy-Aware Demand Prediction": "KUBIK is an unoccupied experimental facility with zero camera or PIR occupant sensors.",
            "Whole-Building Non-HVAC Load Analysis": "Dataset captures only the HVAC chiller and fan-coil cooling circuit.",
            "Zone CO2 Dynamic Tracking": "CO2 telemetry is not included in the summer cooling workbook.",
        },
        primary_power_target="total_chiller_power_w",
        power_unit="Watts (W)",
        has_occupancy=False,
        has_power_quality=True,
    ),
    "LBNL_BUILDING_59": DatasetMetadata(
        dataset_id="LBNL_BUILDING_59",
        building_name="LBNL Building 59 (Lawrence Berkeley National Laboratory)",
        facility_type="4-Story Commercial Office Building (111,904 sq ft)",
        location="Berkeley, California, USA",
        climate_zone="3C (Warm-Summer Mediterranean)",
        period_str="2018-05-22 to 2019-02-21 (275 Days Overlap)",
        cadence_min=15.0,
        supported_capabilities=[
            "Camera-Based Occupancy Headcount Analytics (Floors 3 & 4)",
            "Occupancy-Aware HVAC Electrical Load Prediction",
            "Independent Model Architecture Validation (Ablation: With vs Without Occupancy)",
            "Whole-Building Load Breakdown (HVAC vs Plug Loads vs Lighting)",
            "Zone Comfort Temperature & Airflow Monitoring (UFAD / VAV)",
            "Weather Sensitivity Analysis (Solar Radiation, Ambient Temp, Humidity)",
        ],
        unsupported_capabilities={
            "Three-Phase Power Quality Verification (PF, Harmonics)": "LBNL electrical sub-meters record only active power (kW) and energy; no PF, phase angles, or voltage waveforms.",
            "Chilled Water Supply Setpoint Actuation": "Building 59 uses rooftop packaged air conditioners (RTUs) with supply air VFD modulation rather than a dual-chiller water loop.",
            "Real-Time Equipment Control Dispatch": "Dataset is retrospective passive telemetry; no physical actuation link.",
            "Labelled Equipment Fault Detection": "No ground-truth physical fault injection labels exist in the dataset.",
        },
        primary_power_target="hvac_total_kw",
        power_unit="Kilowatts (kW)",
        has_occupancy=True,
        has_power_quality=False,
    ),
}


def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_id: {dataset_id}. Must be one of {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_id]


def list_available_datasets() -> List[str]:
    return list(DATASET_REGISTRY.keys())
