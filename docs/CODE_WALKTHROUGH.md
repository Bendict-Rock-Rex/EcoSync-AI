# EcoSync-AI Code Walkthrough

A beginner-friendly guide explaining what each key file does and how data moves through the application.

---

## 1. Data Ingestion & Registry Layer (`src/data/`)

### `dataset_registry.py`
- **What it does:** Acts as the building directory. It stores the metadata, physical facility types, and supported versus unsupported capabilities for both `KUBIK_HOLISDER` and `LBNL_BUILDING_59`.
- **Why it matters:** It ensures the application never assumes that a capability available on one building automatically works on another.

### `kubik_loader.py`
- **What it does:** Reads the processed KUBIK 10-minute dataset (`kubik_features.parquet`).
- **Data returned:** A clean pandas DataFrame containing 17,718 rows with chiller powers, water setpoints, 3-phase power factor, voltages, and Spanish TOU tariff signals.

### `lbnl_loader.py`
- **What it does:** Opens the official `Building_59.zip` archive and merges electrical sub-meters (`ele.csv`), camera occupant counts (`occ.csv`), on-site weather (`site_weather.csv`), and Wi-Fi devices (`wifi.csv`).
- **Resampling:** Aggregates 1-minute occupancy counts into 15-minute means to align with the electrical meters over the overlapping 275-day active period (24,290 records).
- **Caching:** Saves output as `data/processed/lbnl_features.parquet` for instant reloading.

### `feature_mapping.py`
- **What it does:** Maps both buildings into a standardized format (`ambient_temp_degc`, `solar_radiation_w_m2`, `hvac_electric_power_kw`) while leaving absent variables as `None` (e.g. occupancy count is `None` for KUBIK; power factor is `None` for LBNL).

---

## 2. Machine Learning & Explainability (`src/models/`)

### `train_surrogates.py`
- **What it does:** Trains regression models (Ridge, Random Forest, Gradient Boosting) on KUBIK data to predict chiller active power ($W$) and zone temperature response ($^\circ\text{C}$).
- **Evaluation:** Uses chronological train/test splits (first 80% train, last 20% test) to prevent temporal data leakage.

### `lbnl_models.py`
- **What it does:** Trains two separate models on LBNL Building 59:
  1. *Baseline Model*: Uses weather and previous power lags only.
  2. *Occupancy-Aware Model*: Adds camera occupant headcount.
- **Ablation Assessment:** Quantifies whether adding occupant counts actually improves forecasting accuracy on a real building.

### `explainability.py`
- **What it does:** Computes global feature importance rankings and explains individual predictions by calculating the exact watt impact of each sensor variable.

---

## 3. Decision Engine & Safety Verification (`src/optimization/` & `src/verification/`)

### `action_optimizer.py`
- **What it does:** Generates 5 discrete operational setpoint candidates (baseline, moderate supply float, max eco, dual chiller, off-peak pre-cool).
- **Evidence Gating:** Gated by physical safety constraints: room temperatures must stay between 21°C and 26°C, setpoints between 7°C and 14°C, and active power below 3,500 W.

### `monitor.py`
- **What it does:** Inspects real-time telemetry for equipment degradation, Power Factor violations ($|\text{PF}| < 0.65$), and three-phase voltage balance deviations ($>5\%$).

---

## 4. User Interface (`app/dashboard.py`)

- **What it does:** Provides an industrial operations console following Schneider Electric design standards with an active facility selector (`KUBIK Experimental Facility` or `LBNL Building 59`).
- **Standardized 9-Stage Workflow:**
  1. `Building Operating State` (live physical telemetry and environmental metrics)
  2. `Demand Forecast` (short-horizon demand forecasts vs. ground truth)
  3. `Risk Assessment` (local and global factor attribution)
  4. `Recommended Actions` (physics-informed setpoint candidates)
  5. `Scenario Evaluation` (what-if load, cost, and carbon predictions)
  6. `Evidence Validation` (data provenance and quality checking)
  7. `Operational Constraint Checks` (comfort bounds, equipment limits, safety caps)
  8. `Recommended Operating Strategy` (Pareto-optimized action recommendation)
  9. `Performance Verification` (continuous telemetry and invariant monitoring)
- **Consoles:**
  1. End-to-End Workflow View
  2. Demand Forecast Tracking
  3. Risk Assessment & Attribution
  4. Scenario Evaluation & Decision Engine
  5. Asset Health & Telemetry Verification
  6. Facility Dataset Specifications
