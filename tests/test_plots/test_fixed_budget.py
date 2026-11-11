import os

import matplotlib
import pytest

from iohinspector.manager import DataManager
from iohinspector.plots import plot_single_function_fixed_budget

matplotlib.use("Agg")  # Use non-interactive backend for tests

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", "test_data"))


@pytest.fixture
def data():
    data_folders = [os.path.join(DATA_DIR, x) for x in sorted(os.listdir(DATA_DIR))]
    data_dir = data_folders[0]
    manager = DataManager()
    manager.add_folder(data_dir)
    return manager.load(monotonic=True, include_meta_data=True)


def test_basic_call_returns_axes_and_data(data):
    ax, dt = plot_single_function_fixed_budget(data)
    assert ax is not None
    assert dt is not None
