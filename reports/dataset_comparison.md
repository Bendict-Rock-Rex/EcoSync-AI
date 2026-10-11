# Dataset Comparison: KUBIK/HOLISDER vs. LBNL Building 59

## 1. High-Level Comparison Matrix

| Property | KUBIK / HOLISDER | LBNL Building 59 |
| :--- | :--- | :--- |
| **Physical Setting** | Dedicated HVAC Experimental Test-Bed (Tecnalia) | Occupied 4-Storey Commercial Office Building |
| **Geographic Location** | Derio, Basque Country, Spain | Berkeley, California, USA |
| **Climate Zone** | Cfb (Marine West Coast) | 3C (Warm-Summer Mediterranean) |
| **Primary System Type** | Dual Water-Cooled Chillers & Fan Coil Loops | Packaged Rooftop AC Units (RTUs) & VAV / UFAD |
| **Measurement Period** | July 1, 2020 – October 31, 2020 (4 Months) | May 22, 2018 – February 21, 2019 (275 Days Overlap) |
| **Sampling Interval** | 10 minutes | 15 minutes (Clean aligned master) |
| **Total Record Count** | 17,718 timesteps | 24,290 timesteps |
| **Primary Energy Metric** | Chiller Electrical Active Power (Watts, W) | HVAC Sub-Metered Electrical Load (Kilowatts, kW) |
| **Occupant Tracking** | **None** (Controlled experimental facility) | **Camera Headcount (TRAF-SYS)** & Wi-Fi Client Counts |
| **Three-Phase Electrical Diagnostics** | **Comprehensive** (PF, VRS, VST, VTR, IR, IS, IT) | **Absent** (Active power kW and kWh only) |
| **Thermal COP Telemetry** | **Directly Measured** ($Q_{\text{cold}} / P_{\text{elec}}$) | Limited (ASHP heat meter covers Aug-Dec 2020 only) |
| **Supply Setpoint Interventions**| **Discrete Staging & Setpoint Control** (10.5°C, 12°C) | Continuous VFD Fan Speed Modulation |

---

## 2. Why the Datasets Must Remain Strictly Separate

1. **Physical Dissimilarity**: KUBIK is a hydronic chilled-water system supplying fan coils with discrete thermal staging; Building 59 is an air-side variable-air-volume (VAV) system with rooftop DX compressors and underfloor plenums.
2. **Telemetry Incompatibilities**: LBNL does not possess three-phase power factor or voltage sensors. Imputing zero or copying values from KUBIK would corrupt power quality monitoring.
3. **Occupancy Authenticity**: KUBIK was operated without internal human occupancy schedules. Copying LBNL occupancy into KUBIK would hallucinate false correlations between non-existent occupants and chiller loads.

---

## 3. What Each Dataset Validates in EcoSync-AI

- **KUBIK**:
  - Primary driver for **coupled HVAC-electrical power quality**.
  - Validates **Power Factor degradation warnings**, three-phase voltage balance ($V_{RS}, V_{ST}, V_{TR}$), and chiller electrical surge limits.
  - Powers the **Evidence-Gated Action Optimizer** for chilled water float and TOU tariff price arbitrage.
- **LBNL Building 59**:
  - Provides **independent real-world validation** of surrogate ML modeling on a multi-floor commercial building.
  - Investigates the **empirical value of camera-based occupancy counts** for HVAC energy forecasting.
  - Establishes whole-building energy breakdown between HVAC (~70%), lighting (~5%), and plug loads (~25%).
