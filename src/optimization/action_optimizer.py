"""
EcoSync-AI: Multi-Building Optimization & What-If Engine
Supports:
1. KUBIK: Chilled water setpoint float, dual-chiller staging, comfort-gated precooling
2. LBNL Building 59: Occupancy-aware ventilation reset, static pressure trim, economizer free-cooling
Every candidate evaluates what-if physical impacts, enforces hard safety constraints,
computes Pareto multi-objective trade-offs, and provides explicit reasons for rejecting unsafe actions.
All simulated outputs are explicitly flagged as SIMULATION ONLY.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure config is importable
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from config import settings


class ActionOptimizer:
    def __init__(self):
        # KUBIK models
        power_model_path = settings.MODELS_DIR / "chiller_power_model.joblib"
        thermal_model_path = settings.MODELS_DIR / "zone_thermal_model.joblib"
        
        self.kubik_power_bundle = joblib.load(power_model_path) if power_model_path.exists() else None
        self.kubik_thermal_bundle = joblib.load(thermal_model_path) if thermal_model_path.exists() else None
        
        # LBNL models
        lbnl_model_path = settings.MODELS_DIR / "lbnl_models.joblib"
        self.lbnl_bundle = joblib.load(lbnl_model_path) if lbnl_model_path.exists() else None

    def generate_candidate_actions(self, current_state, dataset_id="KUBIK_HOLISDER"):
        """Backward-compatible alias for KUBIK candidate generation."""
        if dataset_id == "KUBIK_HOLISDER":
            return self.generate_kubik_candidates(current_state)
        return self.generate_lbnl_candidates(current_state)

    def evaluate_what_if(self, candidate, current_state, dataset_id="KUBIK_HOLISDER"):
        """Backward-compatible alias for what-if evaluation."""
        if dataset_id == "KUBIK_HOLISDER":
            return self.evaluate_kubik_what_if(candidate, current_state)
        return self.evaluate_lbnl_what_if(candidate, current_state)

    # =========================================================
    # KUBIK CANDIDATE ACTIONS & EVALUATION
    # =========================================================
    def generate_kubik_candidates(self, current_state):
        curr_sp1 = float(current_state.get(settings.CHILLER_TAGS["chiller1_setpoint"], 10.5))
        curr_sp2 = float(current_state.get(settings.CHILLER_TAGS["chiller2_setpoint"], 10.3))
        curr_fc1_sp = float(current_state.get(settings.FANCOIL_TAGS["fc1_setpoint"], 23.0))
        curr_fc2_sp = float(current_state.get(settings.FANCOIL_TAGS["fc2_setpoint"], 23.0))

        if np.isnan(curr_sp1) or curr_sp1 < 5.0: curr_sp1 = 10.5
        if np.isnan(curr_sp2) or curr_sp2 < 5.0: curr_sp2 = 10.3
        if np.isnan(curr_fc1_sp) or curr_fc1_sp < 15.0: curr_fc1_sp = 23.0
        if np.isnan(curr_fc2_sp) or curr_fc2_sp < 15.0: curr_fc2_sp = 23.0

        return [
            {
                "action_id": "KUBIK_ACT_0_BASELINE",
                "name": "Current Operational Baseline",
                "description": "Maintain existing chiller staging and thermostat setpoints.",
                "intervention_type": "No Change",
                "chiller1_order": 1,
                "chiller2_order": 0,
                "chiller1_sp": curr_sp1,
                "chiller2_sp": curr_sp2,
                "fc1_sp": curr_fc1_sp,
                "fc2_sp": curr_fc2_sp,
            },
            {
                "action_id": "KUBIK_ACT_1_ECO_FLOAT_MODERATE",
                "name": "Moderate Supply Temp Float (+1.5°C)",
                "description": "Float chilled water setpoints by +1.5°C to reduce compressor lift and active power.",
                "intervention_type": "Chilled Water Reset",
                "chiller1_order": 1,
                "chiller2_order": 0,
                "chiller1_sp": min(curr_sp1 + 1.5, settings.SAFETY_BOUNDS["chiller_sp_max"]),
                "chiller2_sp": min(curr_sp2 + 1.5, settings.SAFETY_BOUNDS["chiller_sp_max"]),
                "fc1_sp": curr_fc1_sp,
                "fc2_sp": curr_fc2_sp,
            },
            {
                "action_id": "KUBIK_ACT_2_ECO_MAX_EFFICIENCY",
                "name": "Maximum Energy Conservation (+2.5°C float, 25.0°C zone)",
                "description": "Aggressive supply float with thermostat trim to upper comfort boundary (25.0°C).",
                "intervention_type": "Supply Float + Thermostat Trim",
                "chiller1_order": 1,
                "chiller2_order": 0,
                "chiller1_sp": min(curr_sp1 + 2.5, settings.SAFETY_BOUNDS["chiller_sp_max"]),
                "chiller2_sp": min(curr_sp2 + 2.5, settings.SAFETY_BOUNDS["chiller_sp_max"]),
                "fc1_sp": 25.0,
                "fc2_sp": 25.0,
            },
            {
                "action_id": "KUBIK_ACT_3_DUAL_CHILLER_PEAK_COOL",
                "name": "Dual-Chiller High Capacity Staging",
                "description": "Engage both Chillers 1 and 2 for rapid pull-down during extreme ambient heat.",
                "intervention_type": "Compressor Staging",
                "chiller1_order": 1,
                "chiller2_order": 1,
                "chiller1_sp": 10.0,
                "chiller2_sp": 10.0,
                "fc1_sp": 22.5,
                "fc2_sp": 22.5,
            },
            {
                "action_id": "KUBIK_ACT_4_PRECOOL_STORAGE",
                "name": "Off-Peak Pre-Cooling (Thermal Storage)",
                "description": "Drop chilled water to 8.5°C to charge building thermal inertia during cheap valley tariff.",
                "intervention_type": "Thermal Pre-Cooling",
                "chiller1_order": 1,
                "chiller2_order": 0,
                "chiller1_sp": 8.5,
                "chiller2_sp": 8.5,
                "fc1_sp": 21.5,
                "fc2_sp": 21.5,
            }
        ]

    def evaluate_kubik_what_if(self, candidate, current_state):
        power_model = self.kubik_power_bundle["model"]
        power_features = self.kubik_power_bundle["feature_names"]
        thermal_model = self.kubik_thermal_bundle["model"]
        thermal_features = self.kubik_thermal_bundle["feature_names"]

        # Feature vector for power
        vec = {}
        for f in power_features:
            if f == settings.CHILLER_TAGS["chiller1_setpoint"]:
                vec[f] = candidate["chiller1_sp"]
            elif f == settings.CHILLER_TAGS["chiller2_setpoint"]:
                vec[f] = candidate["chiller2_sp"]
            elif f == settings.CHILLER_TAGS["chiller1_order"]:
                vec[f] = candidate["chiller1_order"]
            elif f == settings.CHILLER_TAGS["chiller2_order"]:
                vec[f] = candidate["chiller2_order"]
            elif f == settings.FANCOIL_TAGS["fc1_setpoint"]:
                vec[f] = candidate["fc1_sp"]
            elif f == settings.FANCOIL_TAGS["fc2_setpoint"]:
                vec[f] = candidate["fc2_sp"]
            else:
                vec[f] = float(current_state.get(f, 0.0))

        # Evidence validation: Ensure all features are finite
        has_valid_evidence = True
        evidence_violations = []
        for f in power_features:
            val = vec.get(f)
            if val is None or not np.isfinite(val):
                has_valid_evidence = False
                evidence_violations.append(f"Invalid telemetry: {f} is non-finite")

        if not has_valid_evidence:
            return {
                "candidate": candidate,
                "predicted_power_kw": 0.0,
                "predicted_power_w": 0.0,
                "predicted_zone_temp_degc": 24.0,
                "predicted_cost_interval": 0.0,
                "predicted_co2_interval": 0.0,
                "is_safe": False,
                "has_valid_evidence": False,
                "safety_violations": evidence_violations,
                "comfort_penalty": 99.0,
                "rejection_reason": "; ".join(evidence_violations)
            }

        feat_df = pd.DataFrame([vec])[power_features]
        raw_pred_power_w = float(power_model.predict(feat_df)[0])
        
        # Physical bounds: Running equipment cannot draw 0.0 W
        if candidate["chiller1_order"] == 1 and candidate["chiller2_order"] == 1:
            min_operating_power_w = 400.0
        elif candidate["chiller1_order"] == 1 or candidate["chiller2_order"] == 1:
            min_operating_power_w = 180.0
        else:
            min_operating_power_w = 0.0

        if raw_pred_power_w < min_operating_power_w and min_operating_power_w > 0:
            pred_power_w = min_operating_power_w
        else:
            pred_power_w = max(0.0, raw_pred_power_w)

        # Feature vector for zone thermal
        t_vec = {}
        for f in thermal_features:
            if f == settings.FANCOIL_TAGS["fc1_setpoint"]:
                t_vec[f] = candidate["fc1_sp"]
            elif f == settings.CHILLER_TAGS["chiller1_active_power_w"]:
                t_vec[f] = pred_power_w * (1.0 if candidate["chiller2_order"] == 0 else 0.6)
            else:
                t_vec[f] = float(current_state.get(f, 24.0))

        # Check thermal features evidence
        for f, val in t_vec.items():
            if val is None or not np.isfinite(val):
                has_valid_evidence = False
                evidence_violations.append(f"Invalid thermal telemetry: {f}")

        t_feat_df = pd.DataFrame([t_vec])[thermal_features]
        pred_zone_temp = float(thermal_model.predict(t_feat_df)[0])

        tariff = float(current_state.get("tariff_eur_per_kwh", settings.TARIFF_FLAT_EUR_PER_KWH))
        cost_10min = (pred_power_w / 1000.0) * (10.0 / 60.0) * tariff
        co2_10min = (pred_power_w / 1000.0) * (10.0 / 60.0) * settings.GRID_EMISSION_FACTOR_KG_CO2_PER_KWH

        # Evidence-Gated Safety Checks
        safety_violations = []
        is_safe = True

        if not has_valid_evidence:
            safety_violations.extend(evidence_violations)
            is_safe = False

        if pred_zone_temp > settings.SAFETY_BOUNDS["zone_temp_max"]:
            safety_violations.append(f"Comfort upper breach: Zone predicted at {pred_zone_temp:.1f}°C (> {settings.SAFETY_BOUNDS['zone_temp_max']}°C)")
            is_safe = False
        elif pred_zone_temp < settings.SAFETY_BOUNDS["zone_temp_min"]:
            safety_violations.append(f"Overcooling breach: Zone predicted at {pred_zone_temp:.1f}°C (< {settings.SAFETY_BOUNDS['zone_temp_min']}°C)")
            is_safe = False

        for sp_val, sp_name in [(candidate["chiller1_sp"], "Chiller 1 SP"), (candidate["chiller2_sp"], "Chiller 2 SP")]:
            if sp_val < settings.SAFETY_BOUNDS["chiller_sp_min"] or sp_val > settings.SAFETY_BOUNDS["chiller_sp_max"]:
                safety_violations.append(f"Chilled water limit violated for {sp_name}: {sp_val:.1f}°C outside [{settings.SAFETY_BOUNDS['chiller_sp_min']}, {settings.SAFETY_BOUNDS['chiller_sp_max']}]")
                is_safe = False

        if pred_power_w > settings.SAFETY_BOUNDS["chiller_power_max_w"]:
            safety_violations.append(f"Power overload risk: {pred_power_w:.0f} W exceeds limit {settings.SAFETY_BOUNDS['chiller_power_max_w']} W")
            is_safe = False

        comfort_penalty = abs(pred_zone_temp - 23.5)

        return {
            "candidate": candidate,
            "predicted_power_kw": round(pred_power_w / 1000.0, 3),
            "predicted_power_w": round(pred_power_w, 1),
            "predicted_zone_temp_degc": round(pred_zone_temp, 2),
            "predicted_cost_interval": round(cost_10min, 4),
            "predicted_co2_interval": round(co2_10min, 4),
            "is_safe": is_safe,
            "has_valid_evidence": has_valid_evidence,
            "safety_violations": safety_violations,
            "comfort_penalty": comfort_penalty,
            "rejection_reason": "; ".join(safety_violations) if not is_safe else "Passed all mandatory constraints"
        }

    # =========================================================
    # LBNL BUILDING 59 CANDIDATE ACTIONS & EVALUATION
    # =========================================================
    def generate_lbnl_candidates(self, current_state):
        occ_total = float(current_state.get("occ_total", 0.0))
        tout = float(current_state.get("air_temp_set_1", 18.0))
        hvac_lag = float(current_state.get("hvac_power_lag1", 45.0))

        return [
            {
                "action_id": "LBNL_ACT_0_BASELINE",
                "name": "Current Fixed Schedule Baseline",
                "description": "Standard constant static pressure and scheduled ventilation baseline.",
                "intervention_type": "No Change",
                "ventilation_mode": "Scheduled Constant",
                "occ_offset": 0.0,
                "power_multiplier": 1.0,
                "expected_comfort_risk": "Low"
            },
            {
                "action_id": "LBNL_ACT_1_DCV_OCCUPANCY_RESET",
                "name": "Demand-Controlled Ventilation (DCV Reset)",
                "description": "Dynamically modulate outdoor air ventilation based on camera occupant headcount.",
                "intervention_type": "Occupancy Ventilation Reset",
                "ventilation_mode": "Demand Controlled",
                "occ_offset": 0.0,
                # In low occupancy (<10 people), DCV reduces RTU fan/compressor load by ~12-18%
                "power_multiplier": 0.85 if occ_total < 10 else 0.95,
                "expected_comfort_risk": "None" if occ_total < 50 else "Moderate"
            },
            {
                "action_id": "LBNL_ACT_2_ECONOMIZER_FREE_COOLING",
                "name": "Air-Side Economizer Free-Cooling Optimization",
                "description": "Maximize 100% outdoor air intake when ambient temperature (Tout < 16°C) permits free cooling.",
                "intervention_type": "Economizer Damper Reset",
                "ventilation_mode": "Full Economizer",
                "occ_offset": 0.0,
                # Economizer saves compressor power when Tout is cool
                "power_multiplier": 0.78 if tout < 16.0 else 0.98,
                "expected_comfort_risk": "None" if tout < 16.0 else "High (Warm Outdoor Air)"
            },
            {
                "action_id": "LBNL_ACT_3_AGGRESSIVE_FAN_TRIM",
                "name": "Aggressive Static Pressure Trim (-25% Fan Power)",
                "description": "Deep setback of supply air duct static pressure to minimize fan energy.",
                "intervention_type": "Static Pressure Setback",
                "ventilation_mode": "Deep Trim",
                "occ_offset": 0.0,
                "power_multiplier": 0.75,
                "expected_comfort_risk": "High if Occupancy > 25"
            }
        ]

    def evaluate_lbnl_what_if(self, candidate, current_state):
        model_with_occ = self.lbnl_bundle["model_with_occ"]
        occ_features = self.lbnl_bundle["occ_features"]

        vec = {f: float(current_state.get(f, 0.0)) for f in occ_features}

        # Evidence validation: Ensure features are finite
        has_valid_evidence = True
        evidence_violations = []
        for f in occ_features:
            val = vec.get(f)
            if val is None or not np.isfinite(val):
                has_valid_evidence = False
                evidence_violations.append(f"Invalid telemetry: {f} is non-finite")

        if not has_valid_evidence:
            return {
                "candidate": candidate,
                "predicted_power_kw": 0.0,
                "predicted_power_w": 0.0,
                "predicted_zone_temp_degc": 20.0,
                "predicted_cost_interval": 0.0,
                "predicted_co2_interval": 0.0,
                "is_safe": False,
                "has_valid_evidence": False,
                "safety_violations": evidence_violations,
                "comfort_penalty": 99.0,
                "rejection_reason": "; ".join(evidence_violations)
            }

        feat_df = pd.DataFrame([vec])[occ_features]
        base_pred_kw = float(model_with_occ.predict(feat_df)[0])
        
        # Apply physics-informed intervention multiplier
        pred_power_kw = max(10.0, base_pred_kw * candidate["power_multiplier"])

        # Cost and CO2 calculation for California commercial rate ($0.22/kWh) and CA grid factor (0.210 kg CO2/kWh)
        tariff_usd = 0.22
        cost_15min = pred_power_kw * (15.0 / 60.0) * tariff_usd
        co2_15min = pred_power_kw * (15.0 / 60.0) * 0.210

        # Safety & Comfort Constraints
        safety_violations = []
        is_safe = True

        if not has_valid_evidence:
            safety_violations.extend(evidence_violations)
            is_safe = False

        occ_total = float(current_state.get("occ_total", 0.0))
        tout = float(current_state.get("air_temp_set_1", 18.0))

        # Check economizer suitability
        if candidate["action_id"] == "LBNL_ACT_2_ECONOMIZER_FREE_COOLING" and tout > 18.5:
            safety_violations.append(f"Economizer lockout violated: Outdoor temp ({tout:.1f}°C) exceeds 18.5°C threshold. Risk of overheating.")
            is_safe = False

        # Check aggressive fan trim under high occupancy
        if candidate["action_id"] == "LBNL_ACT_3_AGGRESSIVE_FAN_TRIM" and occ_total > 20:
            safety_violations.append(f"Ventilation adequacy risk: Deep fan trim rejected during high occupancy ({occ_total:.0f} occupants > 20 limit). Risk of high CO2.")
            is_safe = False

        # Hard electrical power limit for Building 59 RTU bus (140 kW)
        if pred_power_kw > 140.0:
            safety_violations.append(f"Sub-station demand limit exceeded: {pred_power_kw:.1f} kW > 140.0 kW limit.")
            is_safe = False

        comfort_penalty = 1.0 if not is_safe else (0.2 if occ_total > 15 else 0.0)

        return {
            "candidate": candidate,
            "predicted_power_kw": round(pred_power_kw, 2),
            "predicted_power_w": round(pred_power_kw * 1000.0, 1),
            "predicted_zone_temp_degc": round(tout * 0.3 + 16.0, 2),  # Estimated representative interior equilibrium
            "predicted_cost_interval": round(cost_15min, 4),
            "predicted_co2_interval": round(co2_15min, 4),
            "is_safe": is_safe,
            "has_valid_evidence": has_valid_evidence,
            "safety_violations": safety_violations,
            "comfort_penalty": comfort_penalty,
            "rejection_reason": "; ".join(safety_violations) if not is_safe else "Passed all mandatory constraints"
        }

    # =========================================================
    # UNIFIED SELECTION INTERFACE
    # =========================================================
    def select_best_action(self, dataset_id_or_state, current_state=None, w_cost=0.5, w_comfort=0.3, w_carbon=0.2):
        if current_state is None and isinstance(dataset_id_or_state, dict):
            # Called as select_best_action(current_state)
            dataset_id = "KUBIK_HOLISDER"
            state = dataset_id_or_state
        else:
            dataset_id = dataset_id_or_state
            state = current_state

        if dataset_id == "KUBIK_HOLISDER":
            candidates = self.generate_kubik_candidates(state)
            evaluations = [self.evaluate_kubik_what_if(c, state) for c in candidates]
            power_unit = "W"
        else:
            candidates = self.generate_lbnl_candidates(state)
            evaluations = [self.evaluate_lbnl_what_if(c, state) for c in candidates]
            power_unit = "kW"

        # Baseline evaluation reference
        baseline_eval = next((e for e in evaluations if "BASELINE" in e["candidate"]["action_id"]), evaluations[0])
        baseline_power = baseline_eval["predicted_power_kw"]
        baseline_cost = baseline_eval["predicted_cost_interval"]
        baseline_co2 = baseline_eval["predicted_co2_interval"]

        # Calculate scores and savings for ALL candidates
        for item in evaluations:
            cost = item["predicted_cost_interval"]
            comfort = item["comfort_penalty"]
            carbon = item["predicted_co2_interval"]

            score = (w_cost * cost * 100.0) + (w_comfort * comfort * 10.0) + (w_carbon * carbon * 10.0)
            item["selection_score"] = round(score, 4)

            delta_power_kw = baseline_power - item["predicted_power_kw"]
            delta_cost_pct = ((baseline_cost - cost) / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
            delta_co2_pct = ((baseline_co2 - carbon) / baseline_co2 * 100.0) if baseline_co2 > 0 else 0.0

            item["savings"] = {
                "delta_power_kw": round(delta_power_kw, 3),
                "delta_power_display": f"{round(delta_power_kw * (1000.0 if power_unit == 'W' else 1.0), 1)} {power_unit}",
                "delta_cost_pct": round(delta_cost_pct, 2),
                "delta_co2_pct": round(delta_co2_pct, 2)
            }

        # Lowest energy candidate regardless of feasibility (useful evidence)
        lowest_energy_cand = min(evaluations, key=lambda x: x["predicted_power_kw"])

        # Filter strictly for feasible candidates (safe AND valid evidence)
        safe_evals = [e for e in evaluations if e["is_safe"] and e.get("has_valid_evidence", True)]

        if len(safe_evals) > 0:
            # Pick lowest multi-objective score among strictly safe/feasible candidates
            best_feasible = min(safe_evals, key=lambda x: x["selection_score"])
            selected = best_feasible
            recommendation_status = "FEASIBLE_ACTION_RECOMMENDED"
            status_message = f"Feasible recommendation selected: {selected['candidate']['name']}"
        else:
            best_feasible = None
            selected = None
            recommendation_status = "NO_FEASIBLE_ACTION"
            status_message = (
                "NO FEASIBLE ENERGY-SAVING ACTION. All evaluated candidates violate one or more "
                "mandatory operating constraints. Maintain the appropriate safe operating state and request further assessment."
            )

        return {
            "dataset_id": dataset_id,
            "power_unit": power_unit,
            "selected_action": selected,
            "best_feasible_candidate": best_feasible,
            "lowest_energy_candidate": lowest_energy_cand,
            "recommendation_status": recommendation_status,
            "status_message": status_message,
            "baseline": baseline_eval,
            "all_evaluated_candidates": evaluations,
            "is_simulated": True,
            "num_candidates_evaluated": len(evaluations),
            "num_candidates_passed": len(safe_evals),
            "num_candidates_rejected": len(evaluations) - len(safe_evals)
        }


if __name__ == "__main__":
    optimizer = ActionOptimizer()
    
    # Test KUBIK
    k_df = pd.read_parquet(settings.FEATURE_DATA_PATH)
    k_res = optimizer.select_best_action("KUBIK_HOLISDER", k_df.iloc[100].to_dict())
    print("KUBIK Selected:", k_res["selected_action"]["candidate"]["name"])
    
    # Test LBNL
    l_df = pd.read_parquet(settings.LBNL_FEATURE_PATH)
    l_res = optimizer.select_best_action("LBNL_BUILDING_59", l_df.iloc[100].to_dict())
    print("LBNL Selected:", l_res["selected_action"]["candidate"]["name"])
