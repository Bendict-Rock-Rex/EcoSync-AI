# EcoSync-AI Architecture & System Reference

## 1. System Architecture

EcoSync-AI is structured into a modular pipeline where each component has a single, testable responsibility:

```
[cooling july_october.xlsx] (Raw Workbook, 17,718 rows x 89 channels)
            │
            ▼
[src/data/audit_dataset.py]
  ├── Rigorous audit (missingness, range, COP, power factor)
  ├── Metadata parsing (row 94 headers, tags mapped to subsystems)
  ├── Imputation & Parquet cache (kubik_10min_cleaned.parquet)
  └── Feature pipeline (kubik_features.parquet with TOU tariffs & lags)
            │
            ▼
[src/models/train_surrogates.py]
  ├── Total Chiller Power Predictor (Ridge / RF / Gradient Boosting)
  └── Zone Thermal Dynamics Predictor (Forecasts t+10m zone temperature)
            │
            ├─────────────────────────────────┐
            ▼                                 ▼
[src/models/explainability.py]    [src/optimization/action_optimizer.py]
  ├── Global Feature Importance     ├── Discrete Candidate Actions (Float, Staging, Pre-cool)
  └── Local Step Attribution        ├── What-If Physical Simulation
                                    ├── Evidence-Gated Safety Checks (21-26°C, 7-14°C, 3.5kW)
                                    └── Pareto Multi-Objective Selection
            │                                 │
            └────────────────┬────────────────┘
                             ▼
              [src/verification/monitor.py]
                ├── Physical Invariants Verification
                ├── Three-Phase Balance & PF Diagnostics
                └── Windowed Energy & Carbon Accounting
                             │
                             ▼
                    [app/dashboard.py]
            (Streamlit 6-Console Operations Center)
```

## 2. Ingestion & Audit Realities

- **Row Offset**: The raw Excel file has descriptive metadata in rows 1 to 93. True time-series records begin at row 95 (0-indexed `skiprows=94`).
- **Null Tokens**: Missing sensor records were encoded as the string literal `"Null"` in multiple numeric channels. The ingestion pipeline safely translates these into `np.nan` and applies a forward-fill / backward-fill window of up to 30 minutes.
- **Empirical COP**: Calculated from thermal output over electrical power. Observed median empirical COP was ~0.21 for Chiller 1 and ~0.32 for Chiller 2, reflecting low-load operation and cycling during shoulder seasons.
- **Power Factor**: The Circutor electrical power analyzers reported power factor with negative sign convention (-0.91 to 0.0), indicating capacitive/inductive phase shifts that are audited and verified.

## 3. Evidence-Gated Safety Protocol

When evaluating candidate actions, the system enforces non-negotiable physical constraints:
1. **Thermal Comfort**: Zone temperature must remain within 21.0°C and 26.0°C.
2. **Setpoint Range**: Chilled water supply setpoints are bounded between 7.0°C and 14.0°C.
3. **Maximum Power**: Electrical draw cannot exceed 3,500 W to protect local sub-panels.
4. **Anti-Cycling**: Candidates requiring contradictory chiller starts within 3 consecutive steps (30 min) are gated out.
