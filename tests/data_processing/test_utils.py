import unittest
import numpy as np
from iohinspector.data_processing.utils import get_sequence, geometric_mean
import polars as pl

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

class TestGeometricMean(unittest.TestCase):
    def test_geometric_mean_positive(self):
        s = pl.Series("a", [1, 10, 100])
        result = geometric_mean(s)
        expected = np.exp(np.mean(np.log([1, 10, 100])))
        self.assertAlmostEqual(result, expected)

    def test_geometric_mean_with_ones(self):
        s = pl.Series("a", [1, 1, 1, 1])
        result = geometric_mean(s)
        self.assertEqual(result, 1.0)

    def test_geometric_mean_single_value(self):
        s = pl.Series("a", [42])
        result = geometric_mean(s)
        self.assertAlmostEqual(result, 42.0)

    def test_geometric_mean_large_numbers(self):
        s = pl.Series("a", [1e10, 1e12, 1e14])
        result = geometric_mean(s)
        expected = np.exp(np.mean(np.log([1e10, 1e12, 1e14])))
        self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()