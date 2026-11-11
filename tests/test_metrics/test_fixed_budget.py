import unittest
import polars as pl
import numpy as np
from typing import Callable
from iohinspector.metrics.fixed_budget import aggregate_convergence


class TestFixedBudget(unittest.TestCase):
    def setUp(self):
        # Create a simple test DataFrame
        self.df = pl.DataFrame(
            {
                "evaluations": [1, 2, 3, 1, 2, 3, 1, 3, 1, 3],
                "raw_y": [30, 20, 10, 35, 25, 15, 40, 30, 20, 10],
                "algorithm_name": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B"],
                "run_id": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
            }
        )

    def test_basic_aggregation(self):
        result = aggregate_convergence(self.df)
        # Should contain columns for mean, min, max, median, std, geometric_mean
        for col in ["mean", "min", "max", "median", "std", "geometric_mean"]:
            self.assertIn(col, result.columns)
        # Should have 6 rows (3 evals x 2 algs)
        self.assertEqual(len(result), 6)
        # Check that means are correct for one group
        mean_a = result.filter(pl.col("algorithm_name") == "A")["mean"].to_numpy()
        np.testing.assert_allclose(mean_a, [32.5, 22.5, 12.5])
        mean_b = result.filter(pl.col("algorithm_name") == "B")["mean"].to_numpy()
        np.testing.assert_allclose(mean_b, [30, 30, 20])

    def test_custom_op(self):
        def custom_sum(s):
            return s.sum()  # Sum the Series and return as float

        result = aggregate_convergence(self.df, custom_op=custom_sum)
        self.assertIn("custom_sum", result.columns)

        # Check that custom_sum is correct for one group
        sum_a = result.filter(pl.col("algorithm_name") == "A")["custom_sum"].to_numpy()
        np.testing.assert_allclose(sum_a, [65, 45, 25])
        sum_a = result.filter(pl.col("algorithm_name") == "B")["custom_sum"].to_numpy()
        np.testing.assert_allclose(sum_a, [60, 60, 40])

    def test_maximization(self):
        # Should not affect aggregation, but test for code path
        result = aggregate_convergence(self.df, maximization=True)
        self.assertIn("mean", result.columns)

    def test_eval_min_eval_max(self):
        # Limit to a subset of evaluations
        result = aggregate_convergence(self.df, eval_min=2, eval_max=3)
        self.assertTrue((result["evaluations"] >= 2).all())
        self.assertTrue((result["evaluations"] <= 3).all())

    def test_return_polars(self):
        result = aggregate_convergence(self.df)
        self.assertIsInstance(result, pl.DataFrame)

    def test_free_variables(self):
        # Use a different free variable
        df = self.df.with_columns(pl.lit("foo").alias("other_var"))
        result = aggregate_convergence(df, by=["other_var"])
        self.assertIn("other_var", result.columns)

    def test_empty_data(self):
        empty_df = self.df.filter(pl.col("evaluations") > 100)
        with self.assertRaises(ValueError):
            aggregate_convergence(empty_df)

    def test_single_row(self):
        single_df = pl.DataFrame(
            {"evaluations": [1], "raw_y": [42], "algorithm_name": ["A"], "run_id": [0]}
        )
        result = aggregate_convergence(single_df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result["mean"][0], 42.0)


if __name__ == "__main__":
    unittest.main()
