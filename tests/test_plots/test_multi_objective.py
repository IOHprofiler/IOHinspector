import matplotlib
import polars as pl

from iohinspector.indicators import HyperVolume
from iohinspector.plots import plot_indicator_over_time, plot_paretofronts_2d

matplotlib.use("Agg")  # Use non-interactive backend for testing

PARETO_DATA = pl.DataFrame(
    {
        "raw_y": [0.1, 0.5, 0.9, 0.2, 0.5, 0.9, 0.3, 0.6, 0.9],
        "F2": [0.2, 0.5, 0.8, 0.8, 0.2, 0.9, 0.7, 0.4, 0.1],
        "algorithm_name": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "evaluations": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "run_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    }
)

INDICATOR_DATA = pl.DataFrame(
    {
        "raw_y": [0.9, 0.7, 0.5, 0.3, 0.1],
        "F2": [0.8, 0.6, 0.4, 0.2, 0.1],
        "algorithm_name": ["A"] * 5,
        "evaluations": [1, 10, 100, 1000, 10000],
        "run_id": [1] * 5,
    }
)


def test_paretofronts_2d_basic_call_returns_axes_and_data():
    ax, data = plot_paretofronts_2d(PARETO_DATA)
    assert ax is not None
    assert data is not None


def test_indicator_over_time_basic_call_returns_axes_and_data():
    # Use a simple indicator and check output DataFrame
    indicator = HyperVolume(reference_point=[1.0, 1.0])
    ax, data = plot_indicator_over_time(
        INDICATOR_DATA,
        indicator=indicator,
        eval_steps=5,
        eval_min=1,
        eval_max=10_000,
        scale_eval_log=True,
        obj_vars=["raw_y", "F2"],
        by=["algorithm_name"],
    )
    assert ax is not None
    assert data is not None
