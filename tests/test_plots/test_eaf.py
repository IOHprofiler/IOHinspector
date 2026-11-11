import matplotlib
import polars as pl

from iohinspector.plots import (
    plot_eaf_diffs,
    plot_eaf_pareto,
    plot_eaf_single_objective,
)

matplotlib.use("Agg")  # Use non-interactive backend for tests

SINGLE_OBJECTIVE_DATA = pl.DataFrame(
    {
        "raw_y": [10, 8, 6, 20, 18, 16],
        "evaluations": [1, 2, 5, 1, 4, 5],
        "run_id": [1, 1, 1, 2, 2, 2],
    }
)

PARETO_DATA = pl.DataFrame(
    {
        "x": [1, 2, 3, 1, 2, 3],
        "y": [10, 8, 6, 20, 18, 16],
        "run_id": [1, 1, 1, 2, 2, 2],
    }
)

DATA1 = pl.DataFrame({"x": [1, 2, 3], "y": [10, 8, 6], "run_id": [1, 1, 1]})
DATA2 = pl.DataFrame({"x": [1, 2, 3], "y": [9, 7, 5], "run_id": [2, 2, 2]})


def test_eaf_single_objective_basic_call_returns_axes_and_data():
    ax, data = plot_eaf_single_objective(SINGLE_OBJECTIVE_DATA)
    assert ax is not None
    assert data is not None


def test_eaf_pareto_basic_call_returns_axes_and_data():
    ax, data = plot_eaf_pareto(PARETO_DATA, obj1_var="x", obj2_var="y")
    assert ax is not None
    assert data is not None


def test_eaf_diffs_basic_call_returns_axes_and_data():
    ax, data = plot_eaf_diffs(DATA1, DATA2, obj1_var="x", obj2_var="y")
    assert ax is not None
    assert data is not None
