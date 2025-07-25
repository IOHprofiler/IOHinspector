import unittest
import polars as pl
import numpy as np
import warnings
from iohinspector.data_processing import normalize_objectives

class TestNormalizeObjectives(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            "raw_y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "other": [10, 20, 30, 40, 50]
        })

    def test_basic_normalization(self):
        normed = normalize_objectives(self.df, obj_cols=["raw_y"])
        self.assertIn("ert", normed.columns)
        arr = normed["ert"].to_numpy()
        np.testing.assert_allclose(arr, [1, 0.75, 0.5, 0.25, 0])

    def test_maximization(self):
        normed = normalize_objectives(self.df, obj_cols=["raw_y"], maximize=True)
        arr = normed["ert"].to_numpy()
        np.testing.assert_allclose(arr, [0, 0.25, 0.5, 0.75, 1])

    def test_bounds(self):
        bounds = {"raw_y": (0, 10)}
        normed = normalize_objectives(self.df, obj_cols=["raw_y"], bounds=bounds)
        arr = normed["ert"].to_numpy()
        np.testing.assert_allclose(arr, [0.9, 0.8, 0.7, 0.6, 0.5])

    def test_log_scale(self):
        df = pl.DataFrame({"raw_y": [1, 10, 100, 1000, 10000]})
        normed = normalize_objectives(df, obj_cols=["raw_y"], log_scale=True)
        arr = normed["ert"].to_numpy()
        np.testing.assert_allclose(arr, [1, 0.75, 0.5, 0.25, 0])

    def test_log_scale_with_zero_warns(self):
        df = pl.DataFrame({"raw_y": [0, 1, 10]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normed = normalize_objectives(df, obj_cols=["raw_y"], log_scale=True)
            self.assertTrue(any("Lower bound" in str(warn.message) for warn in w))
        arr = normed["ert"].to_numpy()
        self.assertTrue(np.all((arr >= 0) & (arr <= 1)))

    def test_multiple_objectives(self):
        df = pl.DataFrame({
            "raw_y": [1, 2, 3],
            "other": [10, 20, 30]
        })
        normed = normalize_objectives(df, obj_cols=["raw_y", "other"])
        arr_raw_y = normed["ert_raw_y"].to_numpy()
        np.testing.assert_allclose(arr_raw_y, [1.0, 0.5, 0.0])
        arr_other = normed["ert_other"].to_numpy()
        np.testing.assert_allclose(arr_other, [1.0, 0.5, 0.0])


    def test_column_prefix(self):
        normed = normalize_objectives(self.df, obj_cols=["raw_y"], prefix="normed")
        self.assertIn("normed", normed.columns)

    def test_dict_log_and_maximize(self):
        df = pl.DataFrame({"a": [1, 10, 100], "b": [3, 2, 1]})
        normed = normalize_objectives(
            df,
            obj_cols=["a", "b"],
            log_scale={"a": True, "b": False},
            maximize={"a": True, "b": False}
        )
        arr_raw_y = normed["ert_a"].to_numpy()
        np.testing.assert_allclose(arr_raw_y, [0.0, 0.5, 1.0])
        arr_other = normed["ert_b"].to_numpy()
        np.testing.assert_allclose(arr_other, [0.0, 0.5, 1.0])
        # a is maximized and log scaled, b is minimized and linear

if __name__ == "__main__":
    unittest.main()