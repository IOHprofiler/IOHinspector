import unittest
import polars as pl
import numpy as np
import matplotlib
from pathlib import Path
from iohinspector.plots import plot_eaf_single_objective, plot_eaf_pareto, plot_eaf_diffs

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt


class TestPlotEAFSingleObjective(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            "raw_y": [10, 8, 6, 20, 18, 16],
            "evaluations": [1, 2, 5, 1, 4, 5],
            "data_id": [1, 1, 1, 2, 2, 2]
        })

    def test_basic_call_returns_axes_and_data(self):
        ax, data = plot_eaf_single_objective(self.data)
        self.assertIsNotNone(ax)
        self.assertIsNotNone(data)
        

class TestPlotEAFPareto(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            "x": [1, 2, 3, 1, 2, 3],
            "y": [10, 8, 6, 20, 18, 16],
            "data_id": [1, 1, 1, 2, 2, 2]
        })

    def test_basic_call_returns_axes_and_data(self):
        ax, data = plot_eaf_pareto(self.data, obj1_var="x", obj2_var="y")
        self.assertIsNotNone(ax)
        self.assertIsNotNone(data)
        

class TestPlotEAFDiffs(unittest.TestCase):
    def setUp(self):
        self.data1 = pl.DataFrame({
            "x": [1, 2, 3],
            "y": [10, 8, 6],
            "data_id": [1, 1, 1]
        })
        self.data2 = pl.DataFrame({
            "x": [1, 2, 3],
            "y": [9, 7, 5],
            "data_id": [2, 2, 2]
        })

    def test_basic_call_returns_axes_and_data(self):
        ax, data = plot_eaf_diffs(self.data1, self.data2, obj1_var="x", obj2_var="y")
        self.assertIsNotNone(ax)
        self.assertIsNotNone(data)

if __name__ == "__main__":
    unittest.main()