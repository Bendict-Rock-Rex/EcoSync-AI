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
from src.data.dataset_registry import DATASET_REGISTRY, get_dataset_metadata, list_available_datasets
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

    def test_verification_monitor(self):
        """Verify telemetry verification and continuous monitoring."""
        monitor = VerificationMonitor()
        kubik_loader = KubikLoader()
        df = kubik_loader.load_feature_data()
        
        snap = monitor.verify_telemetry(df.iloc[1000].to_dict())
        self.assertIn("status", snap)
        self.assertIn("metrics", snap)


if __name__ == "__main__":
    unittest.main()
