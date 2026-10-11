# Comprehensive Audit Report: LBNL Building 59 Dataset
**Dataset Provenance:** `doi_10_7941_D1N33Q__v20220202`  
**Authors:** Tianzhen Hong, Na Luo, David Blum, Norman Bourassa, Jeffrey Broughton, Chris Weyandt, Mary Ann Piette (Lawrence Berkeley National Laboratory)  
**Location:** 1 Cyclotron Road, Berkeley, CA 94720, USA (Climate Zone 3C)  
**Building Profile:** 4-storey commercial office building, 111,904 sq ft conditioned area, Packaged Rooftop Air Conditioners (RTUs) with Variable Air Volume (VAV) underfloor air distribution (UFAD), air-source heat pump (ASHP), EnergyStar plug loads, LED lighting.

---

## 1. Executive Summary & Audit Findings

The LBNL Building 59 dataset contains real-world operational measurements collected continuously from **2018-01-01 to 2020-12-31** across 27 time-series CSV streams. Unlike the controlled experimental facility at KUBIK, Building 59 is an actively occupied commercial facility featuring camera-based occupant counters and Wi-Fi device tracking.

### Key Discoveries & Data Realities:
1. **Occupancy Measurement Scope:**
   - Camera-based occupant count (`occ.csv`) was actively logged from **2018-05-22 07:00:00 to 2019-02-21 10:12:00** (275 days, 396,193 1-minute observations) for the south halves of Floor 3 and Floor 4.
   - Mean occupant count across active periods was 6.59 persons (Floor 3 South) and 4.85 persons (Floor 4 South), reaching peak values of 74 and 60 persons during all-hands events.
   - Wi-Fi connection tracking (`wifi.csv`) spans the entire 3-year record at 10-minute intervals.
2. **Electrical Energy Streams (`ele.csv`):**
   - Metered at 15-minute intervals across 5 sub-meters: `mels_S` (Miscellaneous Electric Loads South), `mels_N` (North), `lig_S` (Lighting South), `hvac_N` (HVAC North), `hvac_S` (HVAC South).
   - HVAC electric load accounts for ~70% of total building electric demand, ranging from 0 kW up to 79.8 kW.
3. **Absence of Power Quality & Electrical Diagnostics in LBNL:**
   - **Crucial Negative Finding:** The LBNL Building 59 dataset measures *active electric power (kW)* and energy only. It **does NOT contain three-phase Power Factor (PF), voltages ($V_{RS}, V_{ST}, V_{TR}$), phase currents ($I_R, I_S, I_T$), reactive power (VAr), or Total Harmonic Distortion (THD)**.
   - Therefore, KUBIK remains the **exclusive, irreplaceable primary dataset** in EcoSync-AI for coupled electrical power factor degradation, phase voltage balance, and electrical risk verification.
4. **Absence of Injected Fault Labels:**
   - Neither dataset contains supervised, ground-truth physical fault injection labels. Fault detection must remain evidence-gated, unsupervised, and based on physical safety boundary invariants.
5. **Usable Joint Overlap Window for Occupancy-Aware Modeling:**
   - Merging `ele.csv` (15-min), `occ.csv` (aggregated from 1-min to 15-min), and `site_weather.csv` (15-min) yields a continuous, high-quality overlapping dataset of **24,290 15-minute records** spanning **May 22, 2018 to February 21, 2019** with less than 0.05% missing values.

---

## 2. File Inventory & Storage Metrics

| File Name | Size (Uncompressed) | Cadence | Measurement Domain |
| :--- | :--- | :--- | :--- |
| `ele.csv` | 5.88 MB | 15 min | Sub-metered electrical power: HVAC, MELS, Lighting (kW) |
| `occ.csv` | 11.12 MB | 1 min | Camera occupant counts: Floors 3 & 4 South wings |
| `site_weather.csv` | 5.07 MB | 15 min | On-site ambient temp (°C), RH (%), Solar Radiation (W/m²), Dew point (°C) |
| `wifi.csv` | 2.61 MB | 10 min | Wi-Fi connected client count across 4 floors |
| `zone_co2.csv` | 37.22 MB | 1 min | Indoor CO2 concentration by zone (ppm) |
| `zone_temp_exterior.csv` | 401.40 MB | 1 min | Exterior zone room air temperatures (°F) |
| `zone_temp_interior.csv` | 31.13 MB | 10 min | Interior zone temperatures (°F) |
| `zone_temp_sp_c.csv` | 38.85 MB | 5 min | Zone cooling setpoints (°F) |
| `rtu_sa_t.csv` / `rtu_sa_fr.csv` | ~150 MB | 1 min | Rooftop unit supply air temperature and CFM flow rates |
| `ashp_meter.csv` | 1.43 MB | 5 min | Air-source heat pump energy meter (MBtu/h) (Aug-Dec 2020 only) |

---

## 3. Sensor Unit Differences & Conversions

| Physical Dimension | KUBIK Unit | LBNL Unit | Mathematical Conversion |
| :--- | :--- | :--- | :--- |
| HVAC Power | Watts (W) | Kilowatts (kW) | $P_{\text{kW}} = P_{\text{W}} / 1000$ |
| Ambient Temp | Celsius (°C) | Celsius (°C) | Direct match |
| Indoor Temp | Celsius (°C) | Fahrenheit (°F) | $T_{^\circ\text{C}} = (T_{^\circ\text{F}} - 32) \times \frac{5}{9}$ |
| Solar Radiation | $\text{W/m}^2$ | $\text{W/m}^2$ | Direct match |
| Airflow | $\text{m}^3/\text{h}$ | CFM | $1\text{ CFM} = 1.699\text{ m}^3/\text{h}$ |

---

## 4. Distinct Role of Each Building in EcoSync-AI

```text
┌────────────────────────────────────────────────────────┐
│                   ECOSYNC-AI PLATFORM                  │
├──────────────────────────┬─────────────────────────────┤
│   KUBIK / HOLISDER       │      LBNL BUILDING 59       │
│ (Experimental Facility)  │   (Real Occupied Building)  │
├──────────────────────────┼─────────────────────────────┤
│ • Primary HVAC-Electrical│ • Real-World Occupancy      │
│   Coupling               │   Analytics (Headcount)     │
│ • Three-Phase Power      │ • Occupancy-Aware Energy    │
│   Quality (PF, V, I)     │   Surrogate Modeling        │
│ • Chilled Water Circuit  │ • Independent Architecture  │
│   COP & Staging          │   Validation                │
│ • Real Tariff Arbitrage  │ • Macro Weather & Building  │
│   & Physical Actuation   │   Baseline Comparison       │
└──────────────────────────┴─────────────────────────────┘
```
**Strict Data Integrity Boundary:** KUBIK and LBNL data streams are maintained in separate namespaces (`kubik_loader.py` and `lbnl_loader.py`) with zero cross-contamination.
