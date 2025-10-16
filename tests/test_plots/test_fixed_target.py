import unittest
import polars as pl
import os
from iohinspector.plots import plot_single_function_fixed_target
from iohinspector.manager import DataManager

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", "test_data"))

class TestSingleObjectiveFixedTarget(unittest.TestCase):

    def setUp(self):
        data_folders = [os.path.join(DATA_DIR, x) for x in sorted(os.listdir(DATA_DIR))]
        data_dir = data_folders[0]
        manager = DataManager()
        manager.add_folder(data_dir)
        self.df = manager.load(monotonic=True, include_meta_data=True)

    def test_basic_call_returns_dataframe(self):
        dt = plot_single_function_fixed_target(self.df)
        # Check that the result is a DataFrame and has expected columns
        self.assertTrue(hasattr(dt, "columns"))
        self.assertIn("value", dt.columns)
        self.assertIn("variable", dt.columns)
        self.assertIn("raw_y", dt.columns)
        self.assertIn("algorithm_name", dt.columns)
        sorted_dt = dt.sort_values("raw_y", ascending=True)
        values = sorted_dt["value"].to_numpy()
        # Check that as raw_y decreases, value does not increase (i.e., it decreases or stays the same)
        self.assertTrue(all(x >= y for x, y in zip(values, values[1:])), "value should decrease or stay the same as raw_y decreases")

if __name__ == "__main__":
    unittest.main() 