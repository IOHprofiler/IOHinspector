import unittest
import polars as pl
import numpy as np
from iohinspector.metrics import get_trajectory

class TestGetTrajectory(unittest.TestCase):
    def setUp(self):
        # Example data with two algorithms, two data_ids, and three evaluations each
        self.data = pl.DataFrame({
            "data_id": [1, 1, 1, 2, 2, 2],
            "algorithm_name": ["A", "A", "A", "B", "B", "B"],
            "evaluations": [1, 10, 20, 1, 10, 20],
            "raw_y": [0.5, 0.4, 0.3, 1.0, 0.9, 0.7]
        })

    def test_basic_trajectory(self):
        result = get_trajectory(self.data, return_as_pandas=False)
        self.assertIsInstance(result, pl.DataFrame)
        # Should have as many rows as input (since all evaluations present)
        self.assertEqual(result.shape[0], 40) # 2 algorithms * 20 evaluations
        self.assertIn("evaluations", result.columns)
        self.assertIn("raw_y", result.columns)
        # Check that all evaluation points are present
        for algo in self.data["algorithm_name"].unique():
            evals = result.filter(pl.col("algorithm_name") == algo)["evaluations"].to_list()
            self.assertEqual(set(evals), set(range(1, 21)))
        # Check that raw_y is non-increasing for each algorithm
        for algo in self.data["algorithm_name"].unique():
            raw_y_values = result.filter(pl.col("algorithm_name") == algo).sort("evaluations")["raw_y"].to_list()
            self.assertTrue(all(x >= y for x, y in zip(raw_y_values, raw_y_values[1:])))

    def test_traj_length(self):
        # Only first two evaluations should be present
        result = get_trajectory(self.data, traj_length=1, return_as_pandas=False)
        for algo in self.data["algorithm_name"].unique():
            evals = result.filter(pl.col("algorithm_name") == algo)["evaluations"].to_list()
            self.assertEqual(set(evals), set(range(1, 3)))
        
        result = get_trajectory(self.data, traj_length=10, return_as_pandas=False)
        for algo in self.data["algorithm_name"].unique():
            evals = result.filter(pl.col("algorithm_name") == algo)["evaluations"].to_list()
            self.assertEqual(set(evals), set(range(1, 12)))

    def test_min_fevals(self):
        # Start from evaluation 2
        result = get_trajectory(self.data, min_fevals=2, return_as_pandas=False)
        for algo in self.data["algorithm_name"].unique():
            evals = result.filter(pl.col("algorithm_name") == algo)["evaluations"].to_list()
            self.assertEqual(set(evals), set(range(2, 21)))
        

    def test_custom_free_variables(self):
        # Use only data_id as free variable
        result = get_trajectory(self.data, free_variables=[], return_as_pandas=False)
        self.assertIn("data_id", result.columns)
        self.assertIn("raw_y", result.columns)

    def test_maximization(self):
        result = get_trajectory(self.data, maximization=True, return_as_pandas=False)

        for algo in self.data["algorithm_name"].unique():
            raw_y_values = result.filter(pl.col("algorithm_name") == algo).sort("evaluations")["raw_y"].to_list()
            self.assertTrue(all(x <= y for x, y in zip(raw_y_values, raw_y_values[1:])))

if __name__ == "__main__":
    unittest.main()