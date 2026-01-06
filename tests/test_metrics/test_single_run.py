import unittest
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from iohinspector.metrics.single_run import get_heatmap_single_run_data


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

    def test_basic(self):
        dt_plot = get_heatmap_single_run_data(
            data=self.data,
            vars=self.vars,
            eval_var="evaluations",
            var_mins=self.var_mins,
            var_maxs=self.var_maxs,
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
            get_heatmap_single_run_data(data, ["x1"])

    def test_single_variable(self):
        data = pl.DataFrame({
            "data_id": [1]*3,
            "evaluations": [1, 2, 3],
            "x1": [-5, 0, 5],
        })
        dt_plot = get_heatmap_single_run_data(
            data=data,
            vars=["x1"],
            eval_var="evaluations",
            var_mins=[-5],
            var_maxs=[5],
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
        dt_plot = get_heatmap_single_run_data(
            data=data,
            vars=["x1", "x2"],
            eval_var="evals",
            var_mins=[0, 0],
            var_maxs=[3, 3],
        )
        self.assertEqual(dt_plot.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()