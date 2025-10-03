import unittest
import polars as pl
import math
from iohinspector.metrics.fixed_target import aggregate_running_time

class TestFixedTarget(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            "evaluations": [1, 10, 20, 1, 15, 26],
            "raw_y": [1.0, 0.7, 0.1, 0.9, 0.3, 0.2],
            "algorithm_name": ["A", "A", "A", "B", "B", "B"],
            "data_id": [1, 1, 1, 2, 2, 2]
        })


    def test_basic_aggregation(self):
        result = aggregate_running_time(self.df, return_as_pandas=False)
        self.assertIn("mean", result.columns)
        self.assertIn("ERT", result.columns)
        self.assertIn("PAR-10", result.columns)
        self.assertTrue(result.height > 0)

        # Assert the value of success_count for A is 1 and for B is 0
        # You can use filter as shown, or use row indexing with .row or .to_dicts()
        a_success_count = result.filter(
            (pl.col("algorithm_name") == "A") & (pl.col("raw_y") == 0.1)
        )["success_count"].to_list()[0]
        b_success_count = result.filter(
            (pl.col("algorithm_name") == "B") & (pl.col("raw_y") == 0.1)
        )["success_count"].to_list()[0]

        
        self.assertEqual(a_success_count, 1)
        self.assertEqual(b_success_count, 0)

    def test_return_as_pandas(self):
        result = aggregate_running_time(self.df, return_as_pandas=True)
        self.assertTrue(hasattr(result, "to_numpy"))  # pandas DataFrame

    def test_custom_op(self):
        def my_sum(s):
            return float(s.sum())
        result = aggregate_running_time(self.df, custom_op=my_sum, return_as_pandas=False)
        self.assertIn("my_sum", result.columns)

    def test_maximization(self):
        # Should not raise error
        df = pl.DataFrame({
            "evaluations": [1, 10, 20, 1, 15, 26],
            "raw_y": [0.1, 0.7, 1.0, 0.2, 0.3, 0.9],
            "algorithm_name": ["A", "A", "A", "B", "B", "B"],
            "data_id": [1, 1, 1, 2, 2, 2]
        })

        result = aggregate_running_time(df, maximization=True, return_as_pandas=False)
        self.assertTrue(result.height > 0)

        a_success_count = result.filter(
            (pl.col("algorithm_name") == "A") & (pl.col("raw_y") >= 0.9)
        )["success_count"].to_list()[0]
        b_success_count = result.filter(
            (pl.col("algorithm_name") == "B") & (pl.col("raw_y") >= 0.9)
        )["success_count"].to_list()[0]

        
        self.assertEqual(a_success_count, 1)
        self.assertEqual(b_success_count, 0)

    def test_with_f_min_f_max(self):
        result = aggregate_running_time(self.df, f_min=0.2, f_max=0.5, return_as_pandas=False)
        self.assertTrue(result["raw_y"].min() >= 0.2)
        self.assertTrue(result["raw_y"].max() <= 0.5)

    def test_with_different_free_variables(self):
        result = aggregate_running_time(self.df, free_variables=["algorithm_name", "data_id"], return_as_pandas=False)
        self.assertIn("algorithm_name", result.columns)
        self.assertIn("data_id", result.columns)


    def test_success_ratio_and_count(self):
        # Add a non-finite value
        df = self.df.with_columns([
            pl.when(pl.col("evaluations") == 3).then(float("nan")).otherwise(pl.col("evaluations")).alias("evaluations")
        ])
        result = aggregate_running_time(df, return_as_pandas=False)
        self.assertIn("success_ratio", result.columns)
        self.assertIn("success_count", result.columns)
        self.assertTrue((result["success_ratio"] <= 1).all())

if __name__ == "__main__":
    unittest.main()