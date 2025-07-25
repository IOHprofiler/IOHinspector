import unittest
import polars as pl
import numpy as np
from iohinspector.data_processing.aocc import get_aocc

class TestGetAOCC(unittest.TestCase):
    def setUp(self):
        # Simple dataset with two groups and two data_ids
        self.df = pl.DataFrame({
            "data_id": [1, 1, 1, 2, 2, 2],
            "function_name": ["f1", "f1", "f1", "f1", "f1", "f1"],
            "algorithm_name": ["alg1", "alg1", "alg1", "alg1", "alg1", "alg1"],
            "evaluations": [0, 5, 10, 0, 5, 10],
            "eaf": [10.0, 7.0, 4.0, 12.0, 9.0, 6.0],
        })

    def test_basic_aocc(self):
        # AOCC should be computed for the group
        result = get_aocc(self.df, max_budget=10)
        self.assertIn("AOCC", result.columns)
        aocc_val = result["AOCC"][0]
        self.assertTrue(aocc_val == 6.5)

    def test_multiple_groups(self):
        # Add a second group
        df = self.df.with_columns([
            pl.Series("function_name", ["f1", "f1", "f1", "f2", "f2", "f2"])
        ])
        result = get_aocc(df, max_budget=10)
        self.assertIn("AOCC", result.columns)
        aocc_f1_val = result.filter(pl.col("function_name") == "f1")["AOCC"][0]
        aocc_f2_val = result.filter(pl.col("function_name") == "f2")["AOCC"][0]
        self.assertTrue(aocc_f1_val == 5.5)
        self.assertTrue(aocc_f2_val == 7.5)

    def test_custom_fval_col(self):
        # Use a different column for fval_col
        df = self.df.rename({"eaf": "custom_col"})
        result = get_aocc(df, max_budget=10, fval_col="custom_col")
        self.assertIn("AOCC", result.columns)
        aocc_val = result["AOCC"][0]
        self.assertTrue(aocc_val == 6.5)

    def test_custom_group_cols(self):
        # Use only algorithm_name as group col
        result = get_aocc(self.df, max_budget=10, group_cols=["algorithm_name"])
        aocc_val = result["AOCC"][0]
        self.assertTrue(aocc_val == 6.5)

    def test_aocc_with_missing_evaluations(self):
        # Remove some evaluation steps to test fill_null
        df = pl.DataFrame({
            "data_id": [1, 1, 1, 2, 2],
            "function_name": ["f1", "f1", "f1", "f1", "f1"],
            "algorithm_name": ["alg1", "alg1", "alg1", "alg1", "alg1"],
            "evaluations": [0, 5, 10, 0, 10],
            "eaf": [10.0, 8.0, 4.0, 12.0, 6.0],
        })
        result = get_aocc(df, max_budget=10)
        
        self.assertIn("AOCC", result.columns)
        aocc_val = result["AOCC"][0]
        self.assertTrue(aocc_val == 6)

    def test_aocc_zero_budget(self):
        # Test with max_budget=0 (should handle gracefully)
        df = self.df
        result = get_aocc(df, max_budget=0)
        self.assertIn("AOCC", result.columns)
        # AOCC should be nan or 0
        aocc_val = result["AOCC"][0]
        self.assertTrue(np.isnan(aocc_val) or aocc_val == 0)

if __name__ == "__main__":
    unittest.main()