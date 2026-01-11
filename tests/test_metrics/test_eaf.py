import unittest
import polars as pl
import pandas as pd
import numpy as np
from iohinspector.metrics.eaf import (
    get_discritized_eaf_single_objective,
    get_eaf_data,
    get_eaf_pareto_data,
    get_eaf_diff_data
)

class TestGetDiscritizedEAF(unittest.TestCase):
    def setUp(self):
        # Create a simple polars DataFrame for testing
        self.data = pl.DataFrame({
            "evaluations": [1, 10, 100, 1000],
            "raw_y": [1.0, 0.1, 0.01, 0.001],
            "data_id": [1, 1, 1, 1]
        })

        self.multi_data = pl.DataFrame({
            "evaluations": [1, 10, 100, 1000, 1, 10, 100, 1000],
            "raw_y": [1.0, 0.1, 0.01, 0.001, 1.5, 0.15, 0.015, 0.0015],
            "data_id": [1, 1, 1, 1, 2, 2, 2, 2]
        })

    def test_basic_single_data_id(self):
        result = get_discritized_eaf_single_objective(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        
        self.assertIn('eaf_target', result.index.names)
        self.assertTrue(len(result.columns) == 10) # default x_targets
        self.assertEqual(result.shape[0], 101)  # default y_targets
        # Assert all values are 1 or 0
        self.assertTrue(result[self.data["evaluations"].to_list()].map(lambda x: x in [1, 0]).all().all())
        self.assertEqual(result[1].tolist()[-1], 0)
        self.assertEqual(result[1000].tolist()[0], 1)

        result = get_discritized_eaf_single_objective(self.data, return_as_pandas=False)
        self.assertIsInstance(result, pl.DataFrame)

    def test_basic_multi_data_id(self):
        result = get_discritized_eaf_single_objective(self.multi_data)
        self.assertIn('eaf_target', result.index.names)
        self.assertTrue(len(result.columns) == 10) # default x_targets
        self.assertEqual(result.shape[0], 101)  # default y_targets
        # Assert all values are 1, 0.5 or 0
        self.assertTrue(result[self.multi_data["evaluations"].to_list()].map(lambda x: x in [1, 0.5, 0]).all().all())
        self.assertEqual(result[1].tolist()[-1], 0)
        self.assertEqual(result[1000].tolist()[0], 1)

    def test_custom_eval_values(self):
        eval_values = [1, 3, 5]
        result = get_discritized_eaf_single_objective(self.data, eval_values=eval_values)
        self.assertTrue(all(x in result.columns for x in eval_values))

    def test_custom_eval_min_max(self):
        result = get_discritized_eaf_single_objective(self.data, eval_min=2, eval_max=4, eval_targets=2)
        self.assertTrue(all(x in result.columns for x in [2, 4]))

    def test_custom_f_min_max_targets(self):
        result = get_discritized_eaf_single_objective(self.data, f_min=1e-12, f_max=1.0, f_targets=5)
        self.assertEqual(result.shape[0], 5)
        self.assertAlmostEqual(result.index.min(), 0.0)
        self.assertAlmostEqual(result.index.max(), 1.0)

    def test_scale_eval_log_and_f_log(self):
        result = get_discritized_eaf_single_objective(self.data, scale_f_log=False, scale_eval_log=False)
        # Check that all values except the last row are 1, and the last row is 0
        values = result.values
        self.assertTrue(np.all(values[:-1] == 1))
        self.assertTrue(np.all(values[-1] == 0))
        
        self.budgets = result.columns.to_list()
        np.testing.assert_allclose(self.budgets, np.linspace(1, 1000, 10))


class TestGetEAFData(unittest.TestCase):
    def setUp(self):
        # Simple predictable data: constant improvement
        self.simple_data = pl.DataFrame({
            "evaluations": [1, 2, 3, 4, 5],
            "raw_y": [5.0, 4.0, 3.0, 2.0, 1.0],  # Decreasing linearly
            "data_id": [1, 1, 1, 1, 1]
        })
        
        # Two identical runs for predictable EAF
        self.dual_data = pl.DataFrame({
            "evaluations": [1, 2, 3, 1, 2, 3],
            "raw_y": [10.0, 5.0, 1.0, 10.0, 5.0, 1.0],  # Same values for both runs
            "data_id": [1, 1, 1, 2, 2, 2]
        })


    def test_basic_with_simple_data(self):
        """Test with simple predictable data"""
        result = get_eaf_data(self.simple_data, eval_min=1, eval_max=5, scale_eval_log=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("evaluations", result.columns)
        self.assertIn("raw_y", result.columns)
        self.assertIn("data_id", result.columns)
        
        # Check that we have the expected number of rows (should be same as input)
        self.assertEqual(len(result), len(self.simple_data))
        
        # Check that data_id is preserved
        self.assertEqual(result["data_id"].unique().tolist(), [1])
        
        # Check evaluation values are within expected range
        self.assertTrue((result["evaluations"] >= 1).all())
        self.assertTrue((result["evaluations"] <= 5).all())
        
    def test_dual_runs_predictable_eaf(self):
        result = get_eaf_data(self.dual_data, eval_min=1, eval_max=3, scale_eval_log=False)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("evaluations", result.columns)
        self.assertIn("raw_y", result.columns)
        self.assertIn("data_id", result.columns)
        
        # Should have both data_ids
        self.assertEqual(len(result["data_id"].unique()), 2)
        self.assertEqual(set(result["data_id"].unique()), {1, 2})
        
        # Should have expected number of rows
        self.assertEqual(len(result), len(self.dual_data))
        
    def test_return_types(self):
        """Test different return types"""
        # Pandas return
        result_pd = get_eaf_data(self.simple_data, return_as_pandas=True)
        self.assertIsInstance(result_pd, pd.DataFrame)
        
        # Polars return
        result_pl = get_eaf_data(self.simple_data, return_as_pandas=False)
        self.assertIsInstance(result_pl, pl.DataFrame)


class TestGetEAFParetoData(unittest.TestCase):
    def setUp(self):
        # Simple predictable Pareto front data
        # Run 1: Points that clearly dominate each other
        # Run 2: Same structure but slightly worse
        self.simple_mo_data = pl.DataFrame({
            "obj1": [3.0, 2.0, 1.0, 3.5, 2.5, 0.5],  # Minimization objective
            "obj2": [1.0, 2.0, 3.0, 1.5, 2.5, 3.0],  # Minimization objective
            "data_id": [1, 1, 1, 2, 2, 2]
        })
        
        self.simple_results = pl.DataFrame({
            "obj1": [3.0, 2.0, 1.0, 3.5, 2.5, 0.5],
            "obj2": [1.0, 2.0, 3.0, 1.5, 2.5, 3.0],
            "eaf": [0.5, 0.5, 1.0, 1.0, 1.0, 0.5]
        })

        # Identical runs for predictable EAF = 50%
        self.identical_mo_data = pl.DataFrame({
            "obj1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "obj2": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            "data_id": [1, 1, 1, 2, 2, 2]
        })

    def test_simple_pareto_fronts(self):
        """Test with simple, predictable Pareto front data"""
        result = get_eaf_pareto_data(self.simple_mo_data, "obj1", "obj2")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("eaf", result.columns)
        self.assertIn("obj1", result.columns)
        self.assertIn("obj2", result.columns)
        # Should have some data points
        self.assertGreater(len(result), 0)

        for row in result.itertuples():
            obj1 = row.obj1
            obj2 = row.obj2
            eaf_value = row.eaf
            expected_eaf = self.simple_results.filter(
                (pl.col("obj1") == obj1) & (pl.col("obj2") == obj2)
            )["eaf"].to_list()[0]
        
            self.assertAlmostEqual(eaf_value, expected_eaf)

        
        
    def test_identical_runs_eaf(self):
        result = get_eaf_pareto_data(self.identical_mo_data, "obj1", "obj2")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("eaf", result.columns)
        self.assertIn("obj1", result.columns)
        self.assertIn("obj2", result.columns)
        
        # Should have some data points
        self.assertGreater(len(result), 0)
        
        max_per_pair = result.groupby(["obj1", "obj2"])["eaf"].max()
        self.assertTrue(np.allclose(max_per_pair.values, 1.0))
        
    def test_return_types(self):
        """Test different return types"""
        # Pandas return (default)
        result_pd = get_eaf_pareto_data(self.simple_mo_data, "obj1", "obj2")
        self.assertIsInstance(result_pd, pd.DataFrame)
        
        # Polars return
        result_pl = get_eaf_pareto_data(self.simple_mo_data, "obj1", "obj2", return_as_pandas=False)
        self.assertIsInstance(result_pl, pl.DataFrame)


class TestGetEAFDiffData(unittest.TestCase):
    def setUp(self):
        # Dataset 1: Better performance (lower values for minimization)
        self.better_data = pl.DataFrame({
            "obj1": [1.0, 2.0, 3.0],
            "obj2": [3.0, 2.0, 1.0],
            "data_id": [1, 1, 1]
        })
        
        # Dataset 2: Worse performance (higher values)
        self.worse_data = pl.DataFrame({
            "obj1": [2.0, 3.0, 4.0],
            "obj2": [4.0, 3.0, 2.0],
            "data_id": [1, 1, 1]
        })
        
        # Identical datasets for predictable diff = 0
        self.identical_data1 = pl.DataFrame({
            "obj1": [1.0, 2.0],
            "obj2": [2.0, 1.0],
            "data_id": [1, 1]
        })
        
        self.identical_data2 = pl.DataFrame({
            "obj1": [2.0, 1.0],
            "obj2": [1.0, 2.0],
            "data_id": [1, 1]
        })

    def test_clear_performance_difference(self):
        """Test with clearly better vs worse datasets"""
        result = get_eaf_diff_data(self.better_data, self.worse_data, "obj1", "obj2")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("eaf_diff", result.columns)
        self.assertIn("x_min", result.columns)
        self.assertIn("y_min", result.columns)
        self.assertIn("x_max", result.columns)
        self.assertIn("y_max", result.columns)
        
        # Should have some rectangles with differences
        self.assertGreater(len(result), 0)
        
        # Check that rectangle coordinates are valid
        self.assertTrue((result["x_min"] <= result["x_max"]).all())
        self.assertTrue((result["y_min"] <= result["y_max"]).all())
        
        # Check for no NaN values
        self.assertFalse(result.isna().any().any())
        
        # Since better_data dominates worse_data, should have positive differences
        self.assertGreater(result["eaf_diff"].max(), 0)

    def test_identical_datasets_zero_diff(self):
        """Test with identical datasets - should get minimal or no differences"""
        result = get_eaf_diff_data(self.identical_data1, self.identical_data2, "obj1", "obj2")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("eaf_diff", result.columns)
        
        # Result should be either empty or contain only very small differences
        self.assertTrue(len(result) == 0 or abs(result["eaf_diff"]).max() < 0.1)

    def test_return_types(self):
        """Test different return types"""
        # Pandas return (default)
        result_pd = get_eaf_diff_data(self.better_data, self.worse_data, "obj1", "obj2")
        self.assertIsInstance(result_pd, pd.DataFrame)
        
        # Polars return
        result_pl = get_eaf_diff_data(self.better_data, self.worse_data, "obj1", "obj2", return_as_pandas=False)
        self.assertIsInstance(result_pl, pl.DataFrame)



if __name__ == "__main__":
    unittest.main()