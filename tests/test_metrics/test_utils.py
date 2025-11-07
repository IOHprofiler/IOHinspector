import unittest
import numpy as np
from iohinspector.metrics.utils import get_sequence
import polars as pl
from iohinspector.metrics import normalize_objectives, add_normalized_objectives, transform_fval
import warnings

class TestGetSequence(unittest.TestCase):
    """
    Unit tests for the `get_sequence` function, covering various scenarios:

    - Linear and logarithmic sequences with both float and integer outputs.
    - Edge cases such as minimum equals maximum, single-length sequences, and negative or reversed ranges.
    - Validation of output types, uniqueness when casting to int, and handling of float precision.
    - Ensures proper error handling when invalid parameters are provided (e.g., log scale with zero minimum).
    - Tests for correct sequence generation with large lengths and duplicate handling when casting to int.

    Each test verifies that the output matches expected values and types using NumPy's testing utilities and standard unittest assertions.
    """
    def test_linear_float(self):
        seq = get_sequence(0, 10, 5, scale_log=False, cast_to_int=False)
        expected = np.array([0., 2.5, 5., 7.5, 10.])
        np.testing.assert_allclose(seq, expected)
        self.assertEqual(seq.dtype, float)

    def test_linear_int(self):
        seq = get_sequence(0, 10, 5, scale_log=False, cast_to_int=True)
        self.assertTrue(np.issubdtype(seq.dtype, np.integer))
        self.assertEqual(seq[0], 0)
        self.assertEqual(seq[-1], 10)
        self.assertGreaterEqual(len(seq), 3)

    def test_log_float(self):
        seq = get_sequence(1, 1000, 4, scale_log=True, cast_to_int=False)
        expected = np.array([1., 10., 100., 1000.])
        np.testing.assert_allclose(seq, expected, rtol=1e-6)

    def test_log_int(self):
        seq = get_sequence(1, 1000, 4, scale_log=True, cast_to_int=True)
        expected = np.array([1, 10, 100, 1000])
        np.testing.assert_array_equal(seq, expected)

    def test_min_equals_max(self):
        seq = get_sequence(5, 5, 1, scale_log=False, cast_to_int=False)
        np.testing.assert_array_equal(seq, np.array([5.]))
        

    def test_len_one(self):
        seq = get_sequence(2, 8, 1, scale_log=False, cast_to_int=False)
        np.testing.assert_array_equal(seq, np.array([2.]))

    def test_log_min_zero_raises(self):
        with self.assertRaises(AssertionError):
            get_sequence(0, 10, 5, scale_log=True)

    def test_cast_to_int_uniqueness(self):
        seq = get_sequence(0, 1, 100, scale_log=False, cast_to_int=True)
        np.testing.assert_array_equal(seq, np.array([0, 1]))

    def test_negative_range(self):
        seq = get_sequence(-5, 5, 3, scale_log=False, cast_to_int=False)
        expected = np.array([-5., 0., 5.])
        np.testing.assert_allclose(seq, expected)

    def test_large_len(self):
        seq = get_sequence(0, 1, 1000, scale_log=False, cast_to_int=False)
        self.assertEqual(len(seq), 1000)
        self.assertAlmostEqual(seq[0], 0)
        self.assertAlmostEqual(seq[-1], 1)

    def test_log_scale_non_integer_len(self):
        seq = get_sequence(1, 100, 3, scale_log=True, cast_to_int=False)
        expected = np.array([1., 10., 100.])
        np.testing.assert_allclose(seq, expected, rtol=1e-6)

    def test_cast_to_int_with_duplicates(self):
        seq = get_sequence(0, 0.9, 10, scale_log=False, cast_to_int=True)
        np.testing.assert_array_equal(seq, np.array([0]))


class TestNormalizeObjectives(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            "raw_y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "other": [10, 20, 30, 40, 50]
        })

    def test_basic_normalization(self):
        normed = normalize_objectives(self.df, obj_vars=["raw_y"])
        self.assertIn("ert", normed.columns)
        arr = normed["ert"].to_numpy()
        np.testing.assert_allclose(arr, [1, 0.75, 0.5, 0.25, 0])

    def test_maximization(self):
        normed = normalize_objectives(self.df, obj_vars=["raw_y"], maximize=True)
        arr = normed["ert"].to_numpy()
        np.testing.assert_allclose(arr, [0, 0.25, 0.5, 0.75, 1])

    def test_bounds(self):
        bounds = {"raw_y": (0, 10)}
        normed = normalize_objectives(self.df, obj_vars=["raw_y"], bounds=bounds)
        arr = normed["ert"].to_numpy()
        np.testing.assert_allclose(arr, [0.9, 0.8, 0.7, 0.6, 0.5])

    def test_log_scale(self):
        df = pl.DataFrame({"raw_y": [1, 10, 100, 1000, 10000]})
        normed = normalize_objectives(df, obj_vars=["raw_y"], log_scale=True)
        arr = normed["ert"].to_numpy()
        np.testing.assert_allclose(arr, [1, 0.75, 0.5, 0.25, 0])

    def test_log_scale_with_zero_warns(self):
        df = pl.DataFrame({"raw_y": [0, 1, 10]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normed = normalize_objectives(df, obj_vars=["raw_y"], log_scale=True)
            self.assertTrue(any("Lower bound" in str(warn.message) for warn in w))
        arr = normed["ert"].to_numpy()
        self.assertTrue(np.all((arr >= 0) & (arr <= 1)))

    def test_multiple_objectives(self):
        df = pl.DataFrame({
            "raw_y": [1, 2, 3],
            "other": [10, 20, 30]
        })
        normed = normalize_objectives(df, obj_vars=["raw_y", "other"])
        arr_raw_y = normed["ert_raw_y"].to_numpy()
        np.testing.assert_allclose(arr_raw_y, [1.0, 0.5, 0.0])
        arr_other = normed["ert_other"].to_numpy()
        np.testing.assert_allclose(arr_other, [1.0, 0.5, 0.0])


    def test_column_prefix(self):
        normed = normalize_objectives(self.df, obj_vars=["raw_y"], prefix="normed")
        self.assertIn("normed", normed.columns)

    def test_dict_log_and_maximize(self):
        df = pl.DataFrame({"a": [1, 10, 100], "b": [3, 2, 1]})
        normed = normalize_objectives(
            df,
            obj_vars=["a", "b"],
            log_scale={"a": True, "b": False},
            maximize={"a": True, "b": False}
        )
        arr_raw_y = normed["ert_a"].to_numpy()
        np.testing.assert_allclose(arr_raw_y, [0.0, 0.5, 1.0])
        arr_other = normed["ert_b"].to_numpy()
        np.testing.assert_allclose(arr_other, [0.0, 0.5, 1.0])
        # a is maximized and log scaled, b is minimized and linear

    def test_add_normalized_objectives_basic(self):
        df = pl.DataFrame({
            "raw_y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "other": [10, 20, 30, 40, 50]
        })
        normed = add_normalized_objectives(df, obj_vars=["raw_y", "other"])
        self.assertIn("obj1", normed.columns)
        self.assertIn("obj2", normed.columns)
        arr_obj1 = normed["obj1"].to_numpy()
        arr_obj2 = normed["obj2"].to_numpy()
        np.testing.assert_allclose(arr_obj1, [0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_allclose(arr_obj2, [0, 0.25, 0.5, 0.75, 1])

    def test_add_normalized_objectives_with_bounds(self):
        df = pl.DataFrame({
            "raw_y": [1.0, 2.0, 3.0],
            "other": [10, 20, 30]
        })
        min_obj = pl.DataFrame({"raw_y": [0.0], "other": [0]})
        max_obj = pl.DataFrame({"raw_y": [10.0], "other": [40]})
        normed = add_normalized_objectives(df, obj_vars=["raw_y", "other"], min_obj=min_obj, max_obj=max_obj)
        arr_obj1 = normed["obj1"].to_numpy()
        arr_obj2 = normed["obj2"].to_numpy()
        np.testing.assert_allclose(arr_obj1, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(arr_obj2, [0.25, 0.5, 0.75])

    def test_add_normalized_objectives_single_objective(self):
        df = pl.DataFrame({"raw_y": [1, 2, 3]})
        normed = add_normalized_objectives(df, obj_vars=["raw_y"])
        self.assertIn("obj", normed.columns)
        arr = normed["obj"].to_numpy()
        np.testing.assert_allclose(arr, [0, 0.5, 1])

    def test_add_normalized_objectives_no_min_max(self):
        df = pl.DataFrame({"raw_y": [5, 10, 15]})
        normed = add_normalized_objectives(df, obj_vars=["raw_y"])
        arr = normed["obj"].to_numpy()
        np.testing.assert_allclose(arr, [0, 0.5, 1])

    def test_transform_fval_basic(self):
        df = pl.DataFrame({"raw_y": [1e-8, 1e-4, 1e-2, 1, 1e8]})
        res = transform_fval(df)
        arr = res["eaf"].to_numpy()
        # log10(1e-8) = -8, log10(1e8) = 8
        # normalized = (log10(x) - (-8)) / (8 - (-8)) = (log10(x) + 8) / 16
        expected = [np.abs((np.log10(x) - 8) / 16) for x in [1e-8, 1e-4, 1e-2, 1, 1e8]]
        np.testing.assert_allclose(arr, expected)

    def test_transform_fval_maximization(self):
        df = pl.DataFrame({"raw_y": [1e-8, 1e-4, 1e-2, 1, 1e8]})
        res = transform_fval(df, maximization=True)
        arr = res["eaf"].to_numpy()
        expected = [(np.log10(x) + 8) / 16 for x in [1e-8, 1e-4, 1e-2, 1, 1e8]]
        
        np.testing.assert_allclose(arr, expected)

    def test_transform_fval_minimization(self):
        df = pl.DataFrame({"raw_y": [1e-8, 1e-4, 1e-2, 1, 1e8]})
        res = transform_fval(df, maximization=False)
        arr = res["eaf"].to_numpy()
        expected = [1 - ((np.log10(x) + 8) / 16) for x in [1e-8, 1e-4, 1e-2, 1, 1e8]]
        np.testing.assert_allclose(arr, expected)

    def test_transform_fval_linear_scale(self):
        df = pl.DataFrame({"raw_y": [1e-8, 1e-4, 1e-2, 1, 1e8]})
        res = transform_fval(df, scale_log=False)
        arr = res["eaf"].to_numpy()
        expected = [1-(x - 1e-8) / (1e8 - 1e-8) for x in [1e-8, 1e-4, 1e-2, 1, 1e8]]
        np.testing.assert_allclose(arr, expected)

    def test_transform_fval_custom_bounds(self):
        df = pl.DataFrame({"raw_y": [0, 5, 10]})
        res = transform_fval(df, lb=0, ub=10, scale_log=False)
        arr = res["eaf"].to_numpy()
        # For minimization, 0 maps to 1, 10 maps to 0
        expected = [1 - (x / 10) for x in [0, 5, 10]]
        np.testing.assert_allclose(arr, expected)

    def test_transform_fval_varumn_name(self):
        df = pl.DataFrame({"score": [1, 10, 100]})
        res = transform_fval(df, lb=1, ub=100, scale_log=True, fval_var="score")
        arr = res["eaf"].to_numpy()
        expected = [1- (np.log10(x) - np.log10(1)) / (np.log10(100) - np.log10(1)) for x in [1, 10, 100]]
        np.testing.assert_allclose(arr, expected)

if __name__ == "__main__":
    unittest.main()