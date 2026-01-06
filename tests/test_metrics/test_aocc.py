import unittest
import polars as pl
import pandas as pd
import numpy as np
from iohinspector.metrics import get_aocc

class TestAOCC(unittest.TestCase):
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
        result = get_aocc(self.df, eval_max=10)
        self.assertIsInstance(result, pd.DataFrame)

        self.assertIn("AOCC", result.columns)
        aocc_val = result["AOCC"][0]
        self.assertTrue(aocc_val == 6.5)


        result = get_aocc(self.df, eval_max=10, return_as_pandas=False)
        self.assertIsInstance(result, pl.DataFrame)

    def test_multiple_groups(self):
        # Add a second group
        df = self.df.with_columns([
            pl.Series("function_name", ["f1", "f1", "f1", "f2", "f2", "f2"])
        ])
        
        result = get_aocc(df, eval_max=10)
        self.assertIn("AOCC", result.columns)
        aocc_f1_val = result[result["function_name"] == "f1"]["AOCC"].iloc[0]
        aocc_f2_val = result[result["function_name"] == "f2"]["AOCC"].iloc[0]
        self.assertTrue(aocc_f1_val == 5.5)
        self.assertTrue(aocc_f2_val == 7.5)

    def test_custom_fval_col(self):
        # Use a different column for fval_var
        df = self.df.rename({"eaf": "custom_col"})
        result = get_aocc(df, eval_max=10, fval_var="custom_col")
        self.assertIn("AOCC", result.columns)
        aocc_val = result["AOCC"][0]
        self.assertTrue(aocc_val == 6.5)

    def test_custom_free_vars(self):
        # Use only algorithm_name as free var
        result = get_aocc(self.df, eval_max=10, free_vars=["algorithm_name"])
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
        result = get_aocc(df, eval_max=10)
        
        self.assertIn("AOCC", result.columns)
        aocc_val = result["AOCC"][0]
        self.assertTrue(aocc_val == 6)

    def test_aocc_zero_budget(self):
        # Test with max_budget=0 (should handle gracefully)
        df = self.df
        result = get_aocc(df, eval_max=0)
        self.assertIn("AOCC", result.columns)
        # AOCC should be nan or 0
        aocc_val = result["AOCC"][0]
        self.assertTrue(np.isnan(aocc_val) or aocc_val == 0)


    def test_aocc_log(self):
        self.df = pl.DataFrame({
            "data_id": [1, 1, 1, 2, 2, 2],
            "function_name": ["f1", "f1", "f1", "f1", "f1", "f1"],
            "algorithm_name": ["alg1", "alg1", "alg1", "alg1", "alg1", "alg1"],
            "evaluations": [1, 10, 100, 1, 10, 100],
            "eaf": [10.0, 7.0, 4.0, 12.0, 9.0, 6.0],
        })
        result = get_aocc(self.df, eval_max=100, scale_eval_log=True)
        aocc_val = result["AOCC"][0]
        self.assertTrue(aocc_val == 6.5)

     


if __name__ == "__main__":
    unittest.main()