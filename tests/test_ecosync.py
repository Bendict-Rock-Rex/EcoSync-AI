"""
EcoSync-AI Automated Test Suite (using Python standard library unittest)
Verifies:
1. KUBIK dataset ingestion, parquet integrity, and shape
2. LBNL dataset ingestion, alignment, and camera occupancy variables
3. Dataset registry isolation and distinct capability gating
4. Canonical feature mapping and null handling
5. KUBIK models, explainability, action optimization, and safety gating
6. LBNL surrogate models and occupancy ablation metrics
7. Telemetry physical verification and alert flags
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Project root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import settings
from src.data.dataset_registry import (
    DATASET_REGISTRY,
    get_dataset_metadata,
    list_available_datasets,
    load_feature_availability_matrix,
    FEATURE_AVAILABILITY_COLUMNS
)
from src.data.kubik_loader import KubikLoader
from src.data.lbnl_loader import LBNLLoader
from src.data.feature_mapping import CanonicalFeatureSchema
from src.models.explainability import ExplainabilityEngine
from src.optimization.action_optimizer import ActionOptimizer
from src.verification.monitor import VerificationMonitor
import joblib


class TestEcoSyncAI(unittest.TestCase):

    def test_kubik_dataset_exists_and_valid(self):
        """Verify KUBIK preprocessed parquet exists with expected rows/columns."""
        loader = KubikLoader()
        df = loader.load_feature_data()
        self.assertEqual(len(df), 17718, f"Expected 17,718 records, found {len(df)}")
        self.assertIn("total_chiller_power_w", df.columns)
        self.assertIn("cost_eur_10min", df.columns)
        self.assertIn("co2_kg_10min", df.columns)
        self.assertIn("system_cop", df.columns)

    def test_lbnl_dataset_exists_and_valid(self):
        """Verify LBNL loader extracts Building 59 streams and aligns occupancy."""
        loader = LBNLLoader()
        df = loader.load_feature_data()
        self.assertGreater(len(df), 20000, f"Expected >20,000 aligned records, found {len(df)}")
        self.assertIn("hvac_total_kw", df.columns)
        self.assertIn("occ_total", df.columns)
        self.assertIn("air_temp_set_1", df.columns)
        self.assertIn("solar_radiation_set_1", df.columns)
        self.assertGreater(df["occ_total"].mean(), 0, "Mean occupancy count should be positive")

    def test_dataset_registry_isolation(self):
        """Verify dataset identities remain strictly separate and distinct."""
        datasets = list_available_datasets()
        self.assertIn("KUBIK_HOLISDER", datasets)
        self.assertIn("LBNL_BUILDING_59", datasets)
        
        kubik_meta = get_dataset_metadata("KUBIK_HOLISDER")
        lbnl_meta = get_dataset_metadata("LBNL_BUILDING_59")
        
        self.assertTrue(kubik_meta.has_power_quality)
        self.assertFalse(kubik_meta.has_occupancy)
        
        self.assertFalse(lbnl_meta.has_power_quality)
        self.assertTrue(lbnl_meta.has_occupancy)

    def test_canonical_feature_mapping(self):
        """Verify common schema mapping preserves true values and reports None for absent features."""
        kubik_loader = KubikLoader()
        k_df = kubik_loader.load_feature_data()
        k_mapped = CanonicalFeatureSchema.map_kubik_record(k_df.iloc[100].to_dict())
        
        self.assertEqual(k_mapped["dataset_id"], "KUBIK_HOLISDER")
        self.assertIsNone(k_mapped["occupancy_count"], "KUBIK must have None for occupancy")
        self.assertIsNotNone(k_mapped["power_factor"], "KUBIK must report valid power factor")
        
        lbnl_loader = LBNLLoader()
        l_df = lbnl_loader.load_feature_data()
        l_mapped = CanonicalFeatureSchema.map_lbnl_record(l_df.iloc[100].to_dict())
        
        self.assertEqual(l_mapped["dataset_id"], "LBNL_BUILDING_59")
        self.assertIsNotNone(l_mapped["occupancy_count"], "LBNL must report valid occupancy")
        self.assertIsNone(l_mapped["power_factor"], "LBNL must report None for absent PF")

    def test_safety_bounds_configuration(self):
        """Verify physical and comfort safety bounds."""
        bounds = settings.SAFETY_BOUNDS
        self.assertLess(bounds["zone_temp_min"], bounds["zone_temp_max"])
        self.assertLess(bounds["chiller_sp_min"], bounds["chiller_sp_max"])
        self.assertGreater(bounds["chiller_power_max_w"], 0)
        self.assertEqual(bounds["zone_temp_min"], 21.0)
        self.assertEqual(bounds["zone_temp_max"], 26.0)

    def test_kubik_models_and_explainability(self):
        """Verify KUBIK predictive models and feature importance generation."""
        explainer = ExplainabilityEngine()
        global_imp = explainer.get_global_importance()
        self.assertGreater(len(global_imp), 0)
        top_feature = global_imp[0]
        self.assertIn("relative_importance_pct", top_feature)
        self.assertGreater(top_feature["relative_importance_pct"], 0)

    def test_lbnl_ablation_models(self):
        """Verify LBNL surrogate model bundle exists and contains both baseline and occupancy models."""
        model_path = settings.MODELS_DIR / "lbnl_models.joblib"
        self.assertTrue(model_path.exists(), "LBNL model bundle must exist")
        bundle = joblib.load(model_path)
        self.assertIn("model_no_occ", bundle)
        self.assertIn("model_with_occ", bundle)
        self.assertIn("ablation_comparison", bundle)
        self.assertIn("mae_reduction_pct", bundle["ablation_comparison"])

    def test_action_optimizer_and_safety_gating(self):
        """Verify candidate generation, what-if predictions, and safety checks on KUBIK."""
        optimizer = ActionOptimizer()
        kubik_loader = KubikLoader()
        df = kubik_loader.load_feature_data()
        sample_state = df.iloc[500].to_dict()
        
        candidates = optimizer.generate_candidate_actions(sample_state)
        self.assertGreaterEqual(len(candidates), 4)
        
        res = optimizer.select_best_action(sample_state)
        self.assertIn("selected_action", res)
        self.assertIn("baseline", res)
        
        # Test safety gating: force unsafe setpoint below minimum 7.0°C
        unsafe_cand = {
            "action_id": "TEST_UNSAFE",
            "name": "Unsafe Setpoint Test",
            "chiller1_order": 1,
            "chiller2_order": 0,
            "chiller1_sp": 3.0,
            "chiller2_sp": 3.0,
            "fc1_sp": 23.0,
            "fc2_sp": 23.0
        }
        eval_unsafe = optimizer.evaluate_what_if(unsafe_cand, sample_state)
        self.assertFalse(eval_unsafe["is_safe"], "Candidate with 3.0°C setpoint must fail safety check")

    def test_no_feasible_action_when_all_rejected(self):
        """Verify that when all candidates violate constraints, no action is recommended."""
        optimizer = ActionOptimizer()
        kubik_loader = KubikLoader()
        df = kubik_loader.load_feature_data()
        
        # Snapshot in heatwave where zone temp is high (~36.8°C), exceeding 26.0°C upper limit
        heatwave_state = df.iloc[5280].to_dict()
        res = optimizer.select_best_action("KUBIK_HOLISDER", heatwave_state)
        
        self.assertIsNone(res["selected_action"], "Selected action must be None when all candidates are rejected")
        self.assertEqual(res["recommendation_status"], "NO_FEASIBLE_ACTION")
        self.assertIn("NO FEASIBLE ENERGY-SAVING ACTION", res["status_message"])
        self.assertIsNotNone(res["lowest_energy_candidate"], "Lowest energy candidate should still be tracked for auditing")
        self.assertFalse(res["lowest_energy_candidate"]["is_safe"], "Lowest energy candidate must be marked unsafe")

    def test_lowest_energy_rejected_feasible_selected(self):
        """Verify that a rejected lowest-energy candidate is never recommended over a feasible candidate."""
        optimizer = ActionOptimizer()
        kubik_loader = KubikLoader()
        df = kubik_loader.load_feature_data()
        state = df.iloc[1000].to_dict()
        
        res = optimizer.select_best_action("KUBIK_HOLISDER", state)
        if res["selected_action"] is not None:
            self.assertTrue(res["selected_action"]["is_safe"], "Recommended action must pass all mandatory safety checks")
            self.assertEqual(res["selected_action"]["rejection_reason"], "Passed all mandatory constraints")

    def test_missing_and_nonfinite_telemetry_handling(self):
        """Verify that non-finite or missing sensor values are caught and rejected by evidence validation."""
        optimizer = ActionOptimizer()
        kubik_loader = KubikLoader()
        df = kubik_loader.load_feature_data()
        corrupted_state = df.iloc[1000].to_dict()
        corrupted_state[settings.WEATHER_TAGS["outdoor_temp"]] = np.nan
        
        cand = optimizer.generate_kubik_candidates(corrupted_state)[0]
        eval_res = optimizer.evaluate_kubik_what_if(cand, corrupted_state)
        
        self.assertFalse(eval_res["is_safe"], "Evaluation must fail when input telemetry contains NaN")
        self.assertFalse(eval_res["has_valid_evidence"], "Evidence validity flag must be False")
        self.assertIn("Invalid telemetry", eval_res["rejection_reason"])

    def test_power_and_energy_unit_conversions(self):
        """Verify electrical power and interval cost unit math."""
        # 1000 W for 10 min at 0.24 EUR/kWh = 1 kW * (1/6 h) * 0.24 = 0.04 EUR
        p_w = 1000.0
        dur_h = 10.0 / 60.0
        tariff = 0.24
        cost = (p_w / 1000.0) * dur_h * tariff
        self.assertAlmostEqual(cost, 0.04, places=4)
        
        # Zero baseline edge case: delta cost pct must be 0.0 without ZeroDivisionError
        optimizer = ActionOptimizer()
        zero_state = {f: 0.0 for f in optimizer.kubik_power_bundle["feature_names"]}
        zero_state["tariff_eur_per_kwh"] = 0.15
        res = optimizer.select_best_action("KUBIK_HOLISDER", zero_state)
        self.assertIn("savings", res["all_evaluated_candidates"][0])
        self.assertEqual(res["all_evaluated_candidates"][0]["savings"]["delta_cost_pct"], 0.0)

    def test_feature_availability_valid_schema(self):
        """Verify that lbnl_feature_availability.csv parses cleanly with expected schema, no nulls, and no duplicates."""
        df, err = load_feature_availability_matrix()
        self.assertIsNone(err, f"Feature availability loader returned error: {err}")
        self.assertIsNotNone(df, "Feature availability dataframe must not be None")
        self.assertEqual(len(df), 16, f"Expected 16 rows, got {len(df)}")
        self.assertEqual(list(df.columns), FEATURE_AVAILABILITY_COLUMNS)
        self.assertEqual(df.isnull().sum().sum(), 0, "Feature availability matrix must contain zero null values")
        self.assertFalse(df["Specific_Measurement"].duplicated().any(), "Specific_Measurement must not have duplicates")

    def test_feature_availability_missing_file_handling(self):
        """Verify that a missing report file returns an error message gracefully without raising an exception."""
        non_existent = settings.REPORTS_DIR / "non_existent_report.csv"
        df, err = load_feature_availability_matrix(non_existent)
        self.assertIsNone(df, "DataFrame must be None when file does not exist")
        self.assertIsNotNone(err, "Error message must be returned when file does not exist")
        self.assertIn("not found", err)

    def test_feature_availability_malformed_csv_handling(self):
        """Verify that a malformed CSV with inconsistent commas returns an error message without crashing."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("Col1,Col2\nVal1\nVal1,Val2,Val3,Extra\n")
            tmp_path = Path(tmp.name)
        try:
            df, err = load_feature_availability_matrix(tmp_path)
            self.assertIsNone(df, "DataFrame must be None on malformed CSV")
            self.assertIsNotNone(err, "Error message must describe parsing or schema failure")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_feature_availability_schema_mismatch_handling(self):
        """Verify that a CSV with incorrect columns is detected as a schema mismatch."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("WrongCol1,WrongCol2\nA,B\n")
            tmp_path = Path(tmp.name)
        try:
            df, err = load_feature_availability_matrix(tmp_path)
            self.assertIsNone(df, "DataFrame must be None on schema mismatch")
            self.assertIsNotNone(err, "Error message must report schema mismatch")
            self.assertIn("Schema mismatch", err)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_power_factor_standby_vs_load_diagnostics(self):
        """Verify that PF near 0.13 in standby does not trigger a false utility penalty warning."""
        monitor = VerificationMonitor()
        
        # Standby state (power 120 W, PF 0.13)
        standby_record = {
            settings.CHILLER_TAGS["chiller1_active_power_w"]: 120.0,
            settings.CHILLER_TAGS["chiller1_pf"]: -0.13,
            "HVAC_ENF1_CVM.VRS": 400.0,
            "HVAC_ENF1_CVM.VST": 400.0,
            "HVAC_ENF1_CVM.VTR": 400.0,
            settings.FANCOIL_TAGS["fc1_zone_temp"]: 23.0,
            settings.FANCOIL_TAGS["fc2_zone_temp"]: 23.0
        }
        snap_standby = monitor.verify_telemetry(standby_record)
        # Should NOT be WARNING (marked INFO for idling inductive magnetizing current)
        self.assertNotEqual(snap_standby["status"], "WARNING")
        
        # Active load state (power 1500 W, degraded PF 0.50)
        load_record = standby_record.copy()
        load_record[settings.CHILLER_TAGS["chiller1_active_power_w"]] = 1500.0
        load_record[settings.CHILLER_TAGS["chiller1_pf"]] = -0.50
        snap_load = monitor.verify_telemetry(load_record)
        self.assertEqual(snap_load["status"], "WARNING", "Active load with PF 0.50 must raise WARNING")


    def test_digital_twin_python_simulation_physics(self):
        """Verify that BuildingDigitalTwin Python ODE produces physically realistic response."""
        from src.simulation.building_twin import BuildingDigitalTwin
        twin = BuildingDigitalTwin()
        N = 12
        time_sec = np.arange(N) * 600.0
        T_amb = np.full(N, 30.0)
        solar = np.full(N, 400.0)
        chiller_on = np.ones(N)
        T_chw = np.full(N, 10.5)
        T_zone_sp = np.full(N, 23.5)

        res = twin.simulate_scenario_python(time_sec, T_amb, solar, chiller_on, T_chw, T_zone_sp, initial_T_zone=23.5)
        self.assertEqual(len(res["T_zone_degc"]), N)
        self.assertGreater(res["total_energy_kwh"], 0.0)
        self.assertGreater(res["peak_power_w"], 0.0)
        self.assertLess(res["peak_power_w"], settings.SAFETY_BOUNDS["chiller_power_max_w"])
        # Temperatures must stay finite and physically bounded
        for t in res["T_zone_degc"]:
            self.assertTrue(15.0 <= t <= 40.0, f"Zone temp {t} out of physical bounds")

    def test_simulink_model_file_exists(self):
        """Verify that the generated Simulink model artifact exists."""
        slx_path = BASE_DIR / "matlab" / "simulink" / "kubik_building_model.slx"
        self.assertTrue(slx_path.exists(), "Simulink model .slx file must exist")
        self.assertGreater(slx_path.stat().st_size, 1000, "Simulink model file must not be empty")

    def test_matlab_digital_twin_simulation_reproducibility(self):
        """Verify that MATLAB digital twin simulation runs successfully or falls back cleanly."""
        from src.simulation.building_twin import BuildingDigitalTwin
        twin = BuildingDigitalTwin()
        N = 6
        time_sec = np.arange(N) * 600.0
        T_amb = np.full(N, 28.0)
        solar = np.full(N, 350.0)
        chiller_on = np.ones(N)
        T_chw = np.full(N, 10.5)
        T_zone_sp = np.full(N, 23.5)

        res = twin.simulate_scenario_matlab(time_sec, T_amb, solar, chiller_on, T_chw, T_zone_sp, initial_T_zone=23.5)
        self.assertIn("total_energy_kwh", res)
        self.assertGreater(res["total_energy_kwh"], 0.0)

    def test_closed_loop_autonomous_controller_verification(self):
        """Verify closed-loop sequence: optimization, plant intervention, verification, and recovery."""
        from src.simulation.closed_loop_controller import ClosedLoopController
        loader = KubikLoader()
        df = loader.load_feature_data()
        rec = df.iloc[100].to_dict()
        rec[settings.FANCOIL_TAGS["fc1_zone_temp"]] = 23.5

        ctrl = ClosedLoopController(use_matlab=False)
        # 1. Normal run with successful verification
        out_norm = ctrl.run_simulation_horizon(rec, scenario_name="Standard_Cooling_Day", horizon_steps=12)
        self.assertEqual(out_norm["mode"], "AUTONOMOUS_CLOSED_LOOP")
        self.assertIn(out_norm["verification_status"], ["VERIFIED_SUCCESS", "VERIFIED_FAILURE"])
        self.assertEqual(len(out_norm["events_log"]), 11, "Must contain all 11 architecture steps in log")

        # 2. Failed action triggers recovery
        out_fail = ctrl.run_simulation_horizon(rec, scenario_name="Standard_Cooling_Day", horizon_steps=12, simulate_action_failure=True)
        self.assertEqual(out_fail["verification_status"], "VERIFIED_FAILURE")
        recovery_events = [e for e in out_fail["events_log"] if "RECOVERY" in e.get("status", "") or "RECOVERY" in e.get("detail", "")]
    def test_illegal_6c_candidate_rejection_at_safety_gate(self):
        """Verify that an illegal 6.0°C chilled water setpoint is rejected by the safety gate and never selected."""
        optimizer = ActionOptimizer()
        loader = KubikLoader()
        df = loader.load_feature_data()
        rec = df.iloc[100].to_dict()

        cand_6c = {
            "action_id": "TEST_ILLEGAL_6C",
            "name": "Illegal Overcooling 6C",
            "description": "Setpoint below safety boundary [7.0, 14.0]",
            "intervention_type": "Chilled Water Reset",
            "chiller1_order": 1, "chiller2_order": 0,
            "chiller1_sp": 6.0, "chiller2_sp": 6.0,
            "fc1_sp": 23.0, "fc2_sp": 23.0
        }
        eval_res = optimizer.evaluate_kubik_what_if(cand_6c, rec)
        self.assertFalse(eval_res["is_safe"], "6.0°C candidate must be flagged unsafe")
        self.assertIn("Chilled water limit violated", eval_res["rejection_reason"])

    def test_actuator_fault_injection_and_recovery(self):
        """Verify that a supervisor-bypassing hardware actuator fault is caught by verification and triggers recovery."""
        from src.simulation.closed_loop_controller import ClosedLoopController
        loader = KubikLoader()
        df = loader.load_feature_data()
        rec = df.iloc[100].to_dict()
        rec[settings.FANCOIL_TAGS["fc1_zone_temp"]] = 23.5

        ctrl = ClosedLoopController(use_matlab=False)
        out_fault = ctrl.run_simulation_horizon(rec, scenario_name="Standard_Cooling_Day", horizon_steps=12, simulate_action_failure=True)
        self.assertEqual(out_fault["verification_status"], "VERIFIED_FAILURE")
        # Ensure fault injection was logged explicitly
        fault_logs = [e for e in out_fault["events_log"] if "FAULT" in e.get("phase", "") or "FAULT" in e.get("status", "")]
        self.assertGreater(len(fault_logs), 0, "Must record explicit fault injection event in event log")
        # Ensure recovery was applied
        recovery_logs = [e for e in out_fault["events_log"] if "RECOVERY" in e.get("status", "")]
        self.assertGreater(len(recovery_logs), 0, "Must record recovery event in event log")

    def test_severe_heatwave_unmet_comfort_reporting(self):
        """Verify that severe heatwave (42°C ambient, 28.5°C zone) reports unmet comfort constraint honestly."""
        from src.simulation.closed_loop_controller import ClosedLoopController
        loader = KubikLoader()
        df = loader.load_feature_data()
        rec = df.iloc[100].to_dict()

        ctrl = ClosedLoopController(use_matlab=False)
        out_heat = ctrl.run_simulation_horizon(rec, scenario_name="Severe_Peak_Heatwave", horizon_steps=12, simulate_no_feasible=True)
        self.assertEqual(out_heat["mode"], "NO_FEASIBLE_ACTION_HANDLED")
        self.assertEqual(out_heat["energy_saved_kwh"], 0.0, "Zero false energy savings")
        self.assertEqual(out_heat["savings_pct"], 0.0)
        # Must report unmet comfort constraint honestly, not safe comfort compliance
        self.assertEqual(out_heat["verification_status"], "SAFETY_HELD_UNMET_COMFORT")
        self.assertGreater(out_heat["baseline_results"]["comfort_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()


