import unittest
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from iohinspector.plots.single_run import plot_heatmap_single_run

class TestPlotHeatmapSingleRun(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            "data_id": [1]*5,
            "evaluations": [1,2,3,4,5],
            "x1": np.linspace(-5, 5, 5),
            "x2": np.linspace(-5, 5, 5)[::-1],
        })
        self.var_cols = ["x1", "x2"]
        self.x_mins = np.array([-5, -5])
        self.x_maxs = np.array([5, 5])

    def test_basic(self):
        dt_plot = plot_heatmap_single_run(
            data=self.data,
            var_cols=self.var_cols,
            eval_col="evaluations",
            scale_xlog=False,
            x_mins=self.x_mins,
            x_maxs=self.x_maxs,
        )
        self.assertEqual(dt_plot.shape, (2, 5))
        self.assertAlmostEqual(dt_plot.values.min(), 0)
        self.assertAlmostEqual(dt_plot.values.max(), 1)
        self.assertTrue(np.all((dt_plot.values >= 0) & (dt_plot.values <= 1)))

    def test_asserts_on_multiple_data_ids(self):
        data = pl.DataFrame({
            "data_id": [1, 2],
            "evaluations": [1, 2],
            "x1": [0, 1],
        })
        with self.assertRaises(AssertionError):
            plot_heatmap_single_run(data, ["x1"])

    def test_single_variable(self):
        data = pl.DataFrame({
            "data_id": [1]*3,
            "evaluations": [1, 2, 3],
            "x1": [-5, 0, 5],
        })
        dt_plot = plot_heatmap_single_run(
            data=data,
            var_cols=["x1"],
            eval_col="evaluations",
            scale_xlog=False,
            x_mins=[-5],
            x_maxs=[5],
            ax=None,
            file_name=None,
        )
        self.assertEqual(dt_plot.shape, (1, 3))
        np.testing.assert_allclose(dt_plot.values, [[0, 0.5, 1]])

    def test_non_default_eval_col(self):
        data = pl.DataFrame({
            "data_id": [1]*4,
            "evals": [1, 2, 3, 4],
            "x1": [0, 1, 2, 3],
            "x2": [3, 2, 1, 0],
        })
        dt_plot = plot_heatmap_single_run(
            data=data,
            var_cols=["x1", "x2"],
            eval_col="evals",
            scale_xlog=False,
            x_mins=[0, 0],
            x_maxs=[3, 3],
            ax=None,
            file_name=None,
        )
        self.assertEqual(dt_plot.shape, (2, 4))

if __name__ == "__main__":
    unittest.main()