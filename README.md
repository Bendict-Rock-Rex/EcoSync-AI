# EcoSync-AI — HVAC and Electrical Asset Intelligence Platform
### Schneider Electric Hackathon 2026

EcoSync-AI is a physics-informed HVAC and electrical asset intelligence platform delivering end-to-end decision support, demand forecasting, explainability, operational constraint gating, and continuous performance verification across real-world commercial and experimental building facilities.

---

## 🏛️ Facilities & Multi-Dataset Architecture

| Facility | Location | System Architecture | Primary Analytical Scope |
| :--- | :--- | :--- | :--- |
| **KUBIK / HOLISDER** | Derio, Basque Country, Spain | Dual water-cooled chillers & hydronic fan coil loops | Chiller active power prediction, 3-phase power factor (|PF| ≥ 0.65), phase voltage balance, thermal COP, TOU tariff arbitrage (€/kWh) |
| **LBNL Building 59** | Berkeley, California, USA | 4-storey commercial office with rooftop VAV RTUs | Whole-building load disaggregation, camera-based occupant headcount analytics, occupancy-aware vs non-occupancy demand forecasting |

---

## ⚡ Standardized 9-Stage Engineering Workflow

```text
[Building Operating State]
           ↓
    [Demand Forecast]
           ↓
    [Risk Assessment]
           ↓
  [Recommended Actions]
           ↓
   [Scenario Evaluation]
           ↓
   [Evidence Validation]
           ↓
[Operational Constraint Checks]
           ↓
[Recommended Operating Strategy]
           ↓
[Performance Verification]
```

1. **Building Operating State**: Ingestion and validation of electrical, thermal, environmental, and occupancy telemetry.
2. **Demand Forecast**: Short-horizon regression surrogates evaluated on strict chronological holdout test sets (80% train / 20% test).
3. **Risk Assessment**: Transparent local and global factor attribution quantifying individual variable watt contributions.
4. **Recommended Actions**: Discrete physical setpoint candidates (supply water reset, economizer free-cooling, staging, ventilation trim).
5. **Scenario Evaluation**: Multi-objective what-if projections balancing active power, energy expenditures, thermal comfort, and carbon emissions.
6. **Evidence Validation**: Data-quality checks ensuring all required sensor dimensions are present before action approval.
7. **Operational Constraint Checks**: Hard physical safety gating (21.0°C–26.0°C zone comfort, 7.0°C–14.0°C supply setpoints, 3,500 W chiller capacity cap).
8. **Recommended Operating Strategy**: Multi-objective Pareto selection with explicit constraint compliance tagging (`Simulated scenario`).
9. **Performance Verification**: Real-time physical invariant auditing, phase voltage balance, and empirical COP sanity bounds.

---

## 📁 Repository Directory Structure

```text
EcoSync-AI/
├── app/
│   └── dashboard.py          # Industrial Streamlit Control Room Console
├── config/
│   └── settings.py           # Project paths, safety limits, tariff parameters
├── data/
│   └── processed/
│       ├── kubik_features.parquet      # Processed KUBIK 10-min feature dataset (9.1 MB)
│       └── lbnl_features.parquet       # Processed LBNL 15-min feature dataset (1.8 MB)
├── models/
│   ├── chiller_power_model.joblib      # KUBIK Ridge electrical surrogate model
│   ├── zone_thermal_model.joblib       # KUBIK Gradient Boosting thermal dynamics model
│   └── lbnl_models.joblib              # LBNL Baseline & Occupancy-aware model bundle
├── reports/
│   ├── dataset_audit_report.json       # KUBIK statistical audit and empirical COP report
│   ├── dataset_comparison.md           # Formal cross-building comparison matrix
│   ├── lbnl_dataset_audit.md           # LBNL Building 59 provenance and sensor audit
│   ├── lbnl_data_dictionary.csv        # Documented variable descriptions and sampling rates
│   ├── lbnl_feature_availability.csv   # Feature-by-feature sensor cross-reference
│   ├── lbnl_occupancy_ablation.json    # Quantitative occupancy ablation findings
│   ├── multi_dataset_evaluation.md     # Chronological holdout evaluation summary
│   ├── power_model_metrics.json        # Evaluation metrics and global importances
│   └── tag_metadata.json               # Descriptions and subsystems for 88 KUBIK channels
├── src/
│   ├── data/
│   │   ├── audit_dataset.py            # Ingestion, cleaning, and feature engineering
│   │   ├── dataset_registry.py         # Multi-dataset registry and capability gating
│   │   ├── feature_mapping.py          # Canonical common schema bridge
│   │   ├── kubik_loader.py             # Modular loader for KUBIK
│   │   └── lbnl_loader.py              # Modular loader for LBNL Building 59
│   ├── models/
│   │   ├── explainability.py           # Feature importance and local attribution engine
│   │   ├── lbnl_models.py              # LBNL model training and occupancy ablation study
│   │   └── train_surrogates.py         # KUBIK surrogate training routines
│   ├── optimization/
│   │   └── action_optimizer.py         # Candidate generation, what-if evaluator, safety gating
│   └── verification/
│       └── monitor.py                  # Physical invariants, telemetry verification, anomaly alarms
├── tests/
│   └── test_ecosync.py                 # Automated unit test suite (9 tests)
├── .gitignore                          # Excludes raw 250MB zip archive, caches, and secrets
├── requirements.txt                    # Python runtime dependencies
├── run_dashboard.bat                   # 1-Click launcher for local dashboard
├── run_tests.bat                       # 1-Click test runner
└── README.md                           # Documentation and deployment manual
```

---

## 🚀 Local Execution & Testing

### 1. Run the Dashboard
```bash
python -m streamlit run app/dashboard.py
```
*(Or double-click `run_dashboard.bat` on Windows).*

### 2. Run the Automated Test Suite
```bash
python tests/test_ecosync.py
```
*(All 9 automated unit tests pass in under 4 seconds).*

---

## ☁️ Deployment on Streamlit Community Cloud

This repository is pre-configured for direct zero-configuration deployment to **Streamlit Community Cloud**:
1. **Repository**: Push the prepared files to a GitHub repository named `EcoSync-AI`.
2. **App Entry Point**: `app/dashboard.py`.
3. **Dependencies**: Handled automatically via `requirements.txt`.
4. **Data Handling**: High-efficiency pre-processed Parquet feature files (`data/processed/`) are included in the repository (~11 MB total). Bulky raw archives (`Building_59.zip` at 251 MB) are excluded via `.gitignore` to comply with GitHub's 100 MB file limit while ensuring the deployed web application has full access to verified operational telemetry.
