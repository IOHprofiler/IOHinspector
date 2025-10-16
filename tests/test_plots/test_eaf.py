import unittest
import polars as pl
import numpy as np
import matplotlib
from pathlib import Path
from iohinspector.plots import plot_eaf_single_objective, plot_eaf_pareto, plot_eaf_diffs

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt


class TestPlotEAFSingleObjective(unittest.TestCase):
    def test_basic_call_returns_dataframe(self):
        df = pl.DataFrame({
            "raw_y": [10, 8, 6, 20, 18, 16],
            "evaluations": [1, 2, 5, 1, 4, 5],
            "data_id": [1, 1, 1, 2, 2, 2]
        })
        dt = plot_eaf_single_objective(df)
        self.assertTrue(hasattr(dt, "columns"))
        self.assertIn("evaluations", dt.columns)
        self.assertIn("raw_y", dt.columns)

class TestPlotEAFPareto(unittest.TestCase):
    def test_basic_call_returns_dataframe(self):
        df = pl.DataFrame({
            "x": [1, 2, 3, 1, 2, 3],
            "y": [10, 8, 6, 20, 18, 16],
            "data_id": [1, 1, 1, 2, 2, 2]
        })
        plot_eaf_pareto(df, x_column="x", y_column="y")
        

class TestPlotEAFDiffs(unittest.TestCase):
    def test_basic_call_returns_dataframe(self):
        df1 = pl.DataFrame({
            "x": [1, 2, 3],
            "y": [10, 8, 6],
            "data_id": [1, 1, 1]
        })
        df2 = pl.DataFrame({
            "x": [1, 2, 3],
            "y": [9, 7, 5],
            "data_id": [2, 2, 2]
        })
        plot_eaf_diffs(df1, df2, x_column="x", y_column="y")


if __name__ == "__main__":
    unittest.main()