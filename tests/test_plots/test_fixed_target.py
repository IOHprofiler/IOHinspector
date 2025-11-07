import unittest
import polars as pl
import matplotlib
import os
from iohinspector.plots import plot_single_function_fixed_target
from iohinspector.manager import DataManager

matplotlib.use("Agg")  # Use non-interactive backend for tests
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", "test_data"))


class TestPlotSingleFunctionFixedTarget(unittest.TestCase):

    def setUp(self):
        data_folders = [os.path.join(DATA_DIR, x) for x in sorted(os.listdir(DATA_DIR))]
        data_dir = data_folders[0]
        manager = DataManager()
        manager.add_folder(data_dir)
        self.data = manager.load(monotonic=True, include_meta_data=True)

    def test_basic_call_returns_axes_and_data(self):
        ax, data = plot_single_function_fixed_target(self.data)
        self.assertIsNotNone(ax)
        self.assertIsNotNone(data)

if __name__ == "__main__":
    unittest.main() 