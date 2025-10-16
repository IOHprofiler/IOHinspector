import unittest
import polars as pl
import os
from iohinspector.plots import plot_ecdf
from iohinspector.manager import DataManager

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", "test_data"))

class TestECDF(unittest.TestCase):

    def setUp(self):
        data_folders = [os.path.join(DATA_DIR, x) for x in sorted(os.listdir(DATA_DIR))]
        data_dir = data_folders[0]
        manager = DataManager()
        manager.add_folder(data_dir)
        self.df = manager.load(monotonic=True, include_meta_data=True)

    def test_basic_call_returns_dataframe(self):
        dt = plot_ecdf(self.df)
        # Check that the result is a DataFrame and has expected columns
        self.assertTrue(hasattr(dt, "columns"))
        self.assertIn("evaluations", dt.columns)
        self.assertIn("eaf", dt.columns)
        sorted_dt = dt.sort_values("evaluations", ascending=True)
        values = sorted_dt["eaf"].to_numpy()
        # Check that as evaluations increases, eaf does not increase (i.e., it decreases or stays the same)
        self.assertTrue(all(x <= y for x, y in zip(values, values[1:])), "eaf should increase or stay the same as evaluations increases")

if __name__ == "__main__":
    unittest.main() 