import unittest
import polars as pl
import numpy as np
import matplotlib
from iohinspector.plots import plot_paretofronts_2d, plot_indicator_over_time
import tempfile, os
from iohinspector.indicators import HyperVolume, Epsilon, IGDPlus


matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


class TestPlotParetoFronts2D(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            "raw_y":          [0.1, 0.5, 0.9,   0.2, 0.5, 0.9,   0.3, 0.6, 0.9],
            "F2":             [0.2, 0.5, 0.8,   0.8, 0.2, 0.9,   0.7, 0.4, 0.1],
            "algorithm_name": ["A",  "A", "A",  "B", "B", "B",   "C", "C", "C"],
            "evaluations":    [1,    2,   3,    1,   2,   3,     1,   2,   3],
            "data_id":        [1,    1,   1,    2,   2,   2,     3,   3,   3]
        })

    def test_basic_call_returns_axes_and_data(self):
        ax, data = plot_paretofronts_2d(self.data)
        self.assertIsNotNone(ax)
        self.assertIsNotNone(data)

class TestPlotIndicatorOverTime(unittest.TestCase):         
    def setUp(self):
        self.data = pl.DataFrame({
            "raw_y":          [0.9, 0.7, 0.5, 0.3, 0.1],
            "F2":             [0.8, 0.6, 0.4, 0.2, 0.1],
            "algorithm_name": ["A"] * 5,
            "evaluations":    [1,10,100, 1000, 10000],
            "data_id":        [1] * 5
        })
        # Create a dict mapping evaluation to (raw_y, F2) point
        self.eval_points = dict(zip(self.data["evaluations"], zip(self.data["raw_y"], self.data["F2"])))

    def test_basic_call_returns_axes_and_data(self):
        # Use a simple indicator and check output DataFrame
        indicator = HyperVolume(reference_point=[1.0, 1.0])
        ax, data = plot_indicator_over_time(
            self.data,
            indicator=indicator,
            eval_steps=5,
            eval_min=1,
            eval_max=10_000,
            scale_eval_log=True,
            obj_vars=["raw_y", "F2"],
            free_var="algorithm_name"
        )
        self.assertIsNotNone(ax)
        self.assertIsNotNone(data)
        

if __name__ == "__main__":
    unittest.main()