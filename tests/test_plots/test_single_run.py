import unittest
import polars as pl
import numpy as np
import matplotlib
from iohinspector.plots.single_run import plot_heatmap_single_run

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt


class TestPlotHeatmapSingleRun(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            "data_id": [1]*5,
            "evaluations": [1,2,3,4,5],
            "x1": np.linspace(-5, 5, 5),
            "x2": np.linspace(-5, 5, 5)[::-1],
        })
        self.vars = ["x1", "x2"]
        self.var_mins = np.array([-5, -5])
        self.var_maxs = np.array([5, 5])

    def test_basic_call_returns_axes_and_data(self):
        ax, data = plot_heatmap_single_run(
            data=self.data,
            vars=self.vars,
            eval_var="evaluations",
            var_mins=self.var_mins,
            var_maxs=self.var_maxs,
        )
        self.assertIsNotNone(ax)
        self.assertIsNotNone(data)


if __name__ == "__main__":
    unittest.main()