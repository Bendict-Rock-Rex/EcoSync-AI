"""
EcoSync-AI Simulation & Digital Twin Engine
Bridges Python decision pipeline with the calibrated Building Digital Twin (ODE & Simulink).
Executes baseline and counterfactual scenario runs, records full trajectories,
computes comfort violations, power spikes, energy consumption, and closed-loop verification metrics.
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from config import settings


class BuildingDigitalTwin:
    """
    Reduced-order physical digital twin calibrated on KUBIK experimental cooling data.
    Provides:
    1. Python-native vectorized ODE solver (fast, deterministic, zero-dependency cloud fallback)
    2. Direct MATLAB / Simulink batch runner (when local MATLAB installation is detected)
    Both implementations adhere to the exact same 1st-order lumped thermal-electrical dynamics:
        C_th * dT_zone/dt = Q_envelope + Q_solar + Q_internal - Q_cooling
        COP = COP_nominal * (1 + beta_chw * (T_chw - 10.5) - beta_amb * (T_amb - 25.0))
        P_elec = (Q_cooling / COP) + P_idle
    """

    def __init__(self, mat_engine_path: Optional[str] = "E:\\MATLAB\\Matlab_2025\\bin\\matlab.exe"):
        # Calibrated parameters from KUBIK July-October 2020 dataset
        self.C_th = 4.2e6         # J/K effective thermal capacitance
        self.UA = 125.0           # W/K building envelope conductance
        self.solar_aperture = 1.8 # m2 effective solar aperture area
        self.q_internal = 350.0   # W baseline continuous internal gain
        self.UA_fc = 620.0        # W/K fan-coil heat exchanger capacity
        self.COP_nominal = 3.20   # Nominal COP at 10.5°C supply and 25°C ambient
        self.beta_chw = 0.032     # +3.2% COP per +1°C chiller supply float
        self.beta_amb = 0.018     # -1.8% COP per +1°C outdoor ambient rise
        self.P_chiller_idle = 180.0 # W standby power
        self.P_chiller_max = 3500.0 # W electrical safety bound
        self.pf_nominal = 0.88    # Running power factor
        self.pf_idle = 0.22       # Magnetizing standby power factor
        
        self.matlab_bin = mat_engine_path if Path(mat_engine_path).exists() else None

    def is_matlab_available(self) -> bool:
        """Check if local MATLAB executable exists and is runnable."""
        return self.matlab_bin is not None and Path(self.matlab_bin).exists()

    def simulate_scenario_python(
        self,
        time_steps_sec: np.ndarray,
        T_amb: np.ndarray,
        solar_rad: np.ndarray,
        chiller_on: np.ndarray,
        T_chw_sp: np.ndarray,
        T_zone_sp: np.ndarray,
        initial_T_zone: float = 24.2
    ) -> Dict[str, Any]:
        """
        Pure Python vectorized numerical integration of the calibrated building model.
        Guarantees 100% reproducibility in cloud environments (Streamlit Cloud).
        """
        N = len(time_steps_sec)
        T_zone = np.zeros(N)
        Q_cooling = np.zeros(N)
        P_elec = np.zeros(N)
        COP_eff = np.zeros(N)
        PF_eff = np.zeros(N)
        comfort_violation = np.zeros(N, dtype=bool)

        T_curr = float(initial_T_zone)
        dt = 600.0 # default 10 minutes (matching KUBIK cadence)

        for k in range(N):
            if k > 0:
                dt = float(time_steps_sec[k] - time_steps_sec[k-1])
                if dt <= 0:
                    dt = 600.0

            # 1. Thermal loads
            q_env = self.UA * (T_amb[k] - T_curr)
            q_sol = self.solar_aperture * solar_rad[k]
            q_int = self.q_internal
            q_gain = q_env + q_sol + q_int

            # 2. HVAC Cooling Extraction
            if chiller_on[k] > 0.5 and T_curr > T_chw_sp[k]:
                q_avail = self.UA_fc * max(0.0, T_curr - T_chw_sp[k])
                if T_curr > T_zone_sp[k]:
                    gain_prop = 1200.0 # W/K
                    q_demand = gain_prop * (T_curr - T_zone_sp[k])
                    q_cool = min(q_avail, max(0.0, q_demand))
                else:
                    q_cool = 0.0
            else:
                q_cool = 0.0

            # 3. Discrete thermal update
            dT = ((q_gain - q_cool) / self.C_th) * dt
            T_curr += dT

            # 4. Chiller electrical power & COP
            if chiller_on[k] > 0.5:
                cop = self.COP_nominal * (
                    1.0 + self.beta_chw * (T_chw_sp[k] - 10.5) - self.beta_amb * (T_amb[k] - 25.0)
                )
                cop = max(1.2, min(5.5, cop))
                if q_cool > 50.0:
                    p_active = (q_cool / cop) + self.P_chiller_idle
                    pf = self.pf_nominal
                else:
                    p_active = self.P_chiller_idle
                    pf = self.pf_idle
                p_active = min(self.P_chiller_max, p_active)
            else:
                cop = 0.0
                p_active = 0.0
                pf = 1.0

            T_zone[k] = T_curr
            Q_cooling[k] = q_cool
            P_elec[k] = p_active
            COP_eff[k] = cop
            PF_eff[k] = pf
            comfort_violation[k] = (T_curr > settings.SAFETY_BOUNDS["zone_temp_max"]) or (
                T_curr < settings.SAFETY_BOUNDS["zone_temp_min"]
            )

        total_kwh = float(np.sum(P_elec * (dt / 3600.0)) / 1000.0)

        return {
            "engine": "Python_Calibrated_ODE",
            "time_sec": time_steps_sec.tolist(),
            "T_zone_degc": np.round(T_zone, 3).tolist(),
            "chiller_power_w": np.round(P_elec, 1).tolist(),
            "cooling_thermal_w": np.round(Q_cooling, 1).tolist(),
            "cop": np.round(COP_eff, 3).tolist(),
            "power_factor": np.round(PF_eff, 3).tolist(),
            "comfort_violation": comfort_violation.tolist(),
            "total_energy_kwh": round(total_kwh, 3),
            "peak_power_w": round(float(np.max(P_elec)), 1),
            "mean_zone_temp": round(float(np.mean(T_zone)), 2),
            "max_zone_temp": round(float(np.max(T_zone)), 2),
            "comfort_violation_count": int(np.sum(comfort_violation)),
        }

    def simulate_scenario_matlab(
        self,
        time_steps_sec: np.ndarray,
        T_amb: np.ndarray,
        solar_rad: np.ndarray,
        chiller_on: np.ndarray,
        T_chw_sp: np.ndarray,
        T_zone_sp: np.ndarray,
        initial_T_zone: float = 24.2
    ) -> Dict[str, Any]:
        """
        Runs the simulation via the batch MATLAB engine and extracts JSON results.
        Falls back seamlessly to Python simulation if MATLAB call fails or is unavailable.
        """
        if not self.is_matlab_available():
            res = self.simulate_scenario_python(
                time_steps_sec, T_amb, solar_rad, chiller_on, T_chw_sp, T_zone_sp, initial_T_zone
            )
            res["engine"] = "Python_Fallback (MATLAB not installed or unaccessible)"
            return res

        # Prepare inputs payload
        payload = {
            "time_sec": time_steps_sec.tolist(),
            "T_amb": T_amb.tolist(),
            "solar_rad": solar_rad.tolist(),
            "chiller_on": chiller_on.tolist(),
            "T_chw_sp": T_chw_sp.tolist(),
            "T_zone_sp": T_zone_sp.tolist(),
            "initial_T_zone": float(initial_T_zone),
        }

        in_file = BASE_DIR / "matlab" / "simulation_inputs.json"
        out_file = BASE_DIR / "matlab" / "results" / "simulation_results.json"
        in_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(in_file, "w") as f:
            json.dump(payload, f)

        matlab_cmd = (
            f"addpath('matlab'); run_building_simulation('{in_file.as_posix()}', '{out_file.as_posix()}'); exit;"
        )

        try:
            cmd = [str(self.matlab_bin), "-batch", matlab_cmd]
            subprocess.run(cmd, cwd=str(BASE_DIR), timeout=45, capture_output=True, check=True)

            if out_file.exists():
                with open(out_file, "r") as f:
                    res = json.load(f)
                res["engine"] = "MATLAB_R2025a_Batch_Simulation"
                return res
        except Exception:
            pass

        # Python fallback if MATLAB command times out or errors
        res = self.simulate_scenario_python(
            time_steps_sec, T_amb, solar_rad, chiller_on, T_chw_sp, T_zone_sp, initial_T_zone
        )
        res["engine"] = "Python_Calibrated_ODE (MATLAB Fallback)"
        return res
