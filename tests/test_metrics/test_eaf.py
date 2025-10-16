from turtle import pd
import unittest
import polars as pl
import numpy as np
import matplotlib
from pathlib import Path
from iohinspector.metrics import get_discritized_eaf_single_objective
matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

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
        self.assertIn('eaf_target', result.index.names)
        self.assertTrue(len(result.columns) == 10) # default x_targets
        self.assertEqual(result.shape[0], 101)  # default y_targets
        # Assert all values are 1 or 0
        self.assertTrue(result[self.data["evaluations"].to_list()].applymap(lambda x: x in [1, 0]).all().all())
        self.assertEqual(result[1].tolist()[-1], 0)
        self.assertEqual(result[1000].tolist()[0], 1)

    def test_basic_multi_data_id(self):
        result = get_discritized_eaf_single_objective(self.multi_data)
        self.assertIn('eaf_target', result.index.names)
        self.assertTrue(len(result.columns) == 10) # default x_targets
        self.assertEqual(result.shape[0], 101)  # default y_targets
        # Assert all values are 1, 0.5 or 0
        self.assertTrue(result[self.multi_data["evaluations"].to_list()].applymap(lambda x: x in [1, 0.5, 0]).all().all())
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
        result = get_discritized_eaf_single_objective(self.data, f_min=0.0, f_max=1.0, f_targets=5)
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



if __name__ == "__main__":
    unittest.main()