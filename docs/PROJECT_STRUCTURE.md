# EcoSync-AI Project Structure & File Map

```text
EcoSync-AI/
├── app/
│   └── dashboard.py          # Unified Multi-Building Streamlit Operations Console (7 Views)
├── config/
│   └── settings.py           # Paths, dataset locations, safety bounds, tariff schedules
├── data/
│   └── processed/
│       ├── kubik_10min_cleaned.parquet # Cleaned KUBIK experimental time-series
│       ├── kubik_features.parquet      # Engineered KUBIK features, lags, and tariffs
│       ├── lbnl_15min_cleaned.parquet  # Cleaned LBNL Building 59 aligned time-series
│       └── lbnl_features.parquet       # LBNL features with camera occupancy & lags
├── doi_10_7941_D1N33Q__v20220202/      # Original LBNL Building 59 dataset folder (Unmodified)
│   ├── Building_59.zip                 # Complete 27 CSV time-series package
│   ├── data_description_table_3year_clean_data.xlsx # Official LBNL variable metadata
│   ├── metadata_Dryad_Bldg59.docx      # Dataset background document
│   └── README_Dryad_Bldg59.txt         # Dataset provenance and author notes
├── docs/
│   ├── architecture.md       # High-level pipeline flow and safety gating logic
│   ├── CODE_WALKTHROUGH.md   # Beginner-friendly tour of every critical file
│   ├── IMPLEMENTATION_LOG.md # Detailed record of changes, findings, and remaining work
│   └── PROJECT_STRUCTURE.md  # Repository directory layout and file responsibilities
├── models/
│   ├── chiller_power_model.joblib      # Trained KUBIK chiller electrical power surrogate
│   ├── zone_thermal_model.joblib       # Trained KUBIK zone temperature dynamics model
│   └── lbnl_models.joblib              # Trained LBNL models (Baseline & Occupancy-Aware)
├── reports/
│   ├── dataset_audit_report.json       # KUBIK statistical audit and COP metrics
│   ├── dataset_comparison.md           # KUBIK vs LBNL comprehensive matrix
│   ├── dataset_numerical_summary.csv   # KUBIK raw column statistics
│   ├── lbnl_data_dictionary.csv        # Complete variable table for LBNL Building 59
│   ├── lbnl_dataset_audit.md           # Audit report for LBNL Building 59
│   ├── lbnl_feature_availability.csv   # Feature-by-feature cross-building comparison
│   ├── lbnl_occupancy_ablation.json    # Quantitative ablation study results
│   ├── multi_dataset_evaluation.md     # Cross-building evaluation and ablation findings
│   ├── power_model_metrics.json        # KUBIK model performance metrics and importances
│   └── tag_metadata.json               # Descriptions and subsystems for 88 KUBIK tags
├── src/
│   ├── data/
│   │   ├── audit_dataset.py            # KUBIK audit and preprocessing pipeline
│   │   ├── dataset_registry.py         # Multi-dataset registry and capability gating
│   │   ├── feature_mapping.py          # Standardized common schema bridge
│   │   ├── kubik_loader.py             # Modular loader for KUBIK
│   │   └── lbnl_loader.py              # Modular loader for LBNL Building 59
│   ├── models/
│   │   ├── explainability.py           # Global and local feature attribution engine
│   │   ├── lbnl_models.py              # LBNL model training and occupancy ablation study
│   │   └── train_surrogates.py         # KUBIK surrogate model training routines
│   ├── optimization/
│   │   └── action_optimizer.py         # Candidate generation, what-if evaluator, safety gating
│   └── verification/
│       └── monitor.py                  # Physical invariants, telemetry verification, anomaly alarms
├── tests/
│   └── test_ecosync.py                 # Automated unit tests covering both datasets and engines
├── cooling july_october.xlsx           # Original unchanged KUBIK experimental workbook
├── run_dashboard.bat                   # 1-Click launcher for Streamlit operations dashboard
├── run_tests.bat                       # 1-Click test runner
├── requirements.txt                    # Python library dependencies
└── README.md                           # Main user manual and getting-started guide
```
