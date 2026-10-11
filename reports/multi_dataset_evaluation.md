# Multi-Dataset Evaluation & Occupancy Ablation Findings

**Evaluation Date:** October 2026  
**Pipeline:** EcoSync-AI Multi-Building Validation Engine  
**Validation Protocols:** Chronological 80/20 train/test splits (Zero data leakage)

---

## 1. Primary Model Performance by Dataset

### KUBIK / HOLISDER Chiller Power Surrogate
- **Target Variable:** `total_chiller_power_w` (Watts)
- **Features:** Outdoor temperature, Solar radiation, Chiller supply setpoints, Chiller staging orders, Fan-coil return temperatures, Diurnal cycle, Power lags.
- **Evaluation Split:** Chronological split (14,174 train rows / 3,544 test rows).
- **Test Set Metrics:**
  - $R^2$ Score: **0.3546**
  - Mean Absolute Error (MAE): **305.16 W**
  - Root Mean Squared Error (RMSE): **386.58 W**
- **Physical Interpretation:** Captures baseline cycling and temperature lift dependencies across wide ambient thermal changes.

---

## 2. LBNL Building 59 Occupancy Ablation Study

Does measured occupant headcount improve HVAC load forecasting in a real commercial building?

### Experimental Design:
- **Dataset:** LBNL Building 59 Overlap Period (24,290 15-minute observations).
- **Target Variable:** `hvac_total_kw` (Kilowatts).
- **Training Period:** 2018-05-22 to 2018-12-29 (19,432 samples).
- **Chronological Test Period:** 2018-12-29 to 2019-02-21 (4,858 samples).
- **Algorithms:** Gradient Boosting Regressors ($N=100$, Depth=5, $\text{LR}=0.08$).

### Quantitative Ablation Results:

| Model Variant | Feature Set | Test $R^2$ | Test MAE (kW) | Test RMSE (kW) |
| :--- | :--- | :--- | :--- | :--- |
| **Model A (Baseline - No Occupancy)** | Weather + Diurnal + Temp Lags + Power Lag | **0.6648** | **7.26 kW** | **9.49 kW** |
| **Model B (Occupancy-Aware)** | Model A + Camera Headcount (`occ_total`) + Occ Lag | **0.6643** | **7.25 kW** | **9.50 kW** |
| **Performance Difference** | — | **-0.0005** | **+0.17% (Better)** | **-0.07% (Slightly worse)** |

### Scientific Insights & Engineering Rationale:
1. **Marginal Direct Gain:** Adding camera-measured occupant headcount provided a minuscule **+0.17% reduction in MAE** (7.26 kW down to 7.25 kW).
2. **Dominance of Thermal Inertia & Baselines:** In large commercial office buildings, rooftop HVAC fans and minimum outdoor air ventilation dampers run continuously on fixed schedule baselines during business hours regardless of whether 5 or 50 people occupy a wing.
3. **Weather & Schedule Priority:** Feature importance analysis revealed that previous power lag (`hvac_power_lag1`, 82.2%) and outdoor air temperature (`air_temp_set_1`, 7.6%) account for nearly 90% of model predictive power. Occupant headcount accounted for only **1.60%** relative importance.
4. **Takeaway for Practice:** Camera-based occupancy tracking provides situational awareness for ventilation trimming, but coarse weather forecasting and past power load remain the dominant predictive drivers for short-term electric demand.
