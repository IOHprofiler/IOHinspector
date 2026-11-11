import warnings

import matplotlib
import polars as pl

matplotlib.use("Agg")  # Use non-interactive backend for tests

from iohinspector.indicators import HyperVolume
from iohinspector.plots import (
    plot_robustrank_changes,
    plot_robustrank_over_time,
    plot_tournament_ranking,
)

DATA = pl.DataFrame(
    {
        "algorithm_name": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "function_name": ["f1", "f2", "f3", "f1", "f2", "f3", "f1", "f2", "f3"],
        "raw_y": [1.0, 2.0, 1.7, 1.5, 2.8, 2.1, 0.9, 0.5, 1.6],
    }
)

OBJ_DATA = pl.DataFrame(
    {
        "algorithm_name": ["A"] * 9 + ["B"] * 9 + ["C"] * 9,
        "evaluations": [1, 10, 100] * 9,
        "f1": [
            # A: best at eval 1, B: best at eval 10, A: best at eval 100 (for run 1)
            0.8,
            1.5,
            0.7,  # A, run 1
            1.0,
            1.6,
            0.9,  # A, run 2
            0.9,
            1.4,
            0.8,  # A, run 3
            1.0,
            0.7,
            1.2,  # B, run 1
            1.2,
            0.8,
            1.3,  # B, run 2
            1.1,
            0.6,
            1.1,  # B, run 3
            1.5,
            1.5,
            0.1,  # C, run 1
            1.6,
            1.6,
            0.2,  # C, run 2
            1.4,
            1.4,
            0.3,  # C, run 3
        ],
        "f2": [
            1.0,
            2.0,
            0.9,  # A, run 1
            1.2,
            2.1,
            1.1,  # A, run 2
            1.1,
            2.2,
            1.0,  # A, run 3
            1.3,
            0.8,
            1.4,  # B, run 1
            1.5,
            0.9,
            1.5,  # B, run 2
            1.4,
            0.7,
            1.3,  # B, run 3
            2.0,
            2.0,
            0.1,  # C, run 1
            2.1,
            2.1,
            0.2,  # C, run 2
            1.9,
            1.9,
            0.3,  # C, run 3
        ],
        "f3": [
            2.0,
            3.0,
            1.8,  # A, run 1
            2.2,
            3.1,
            2.0,  # A, run 2
            2.1,
            3.2,
            1.9,  # A, run 3
            2.3,
            1.2,
            2.4,  # B, run 1
            2.5,
            1.3,
            2.5,  # B, run 2
            2.4,
            1.1,
            2.3,  # B, run 3
            3.0,
            3.0,
            0.1,  # C, run 1
            3.1,
            3.1,
            0.3,  # C, run 2
            2.9,
            2.9,
            0.2,  # C, run 3
        ],
        "run_id": [1] * 3
        + [2] * 3
        + [3] * 3
        + [4] * 3
        + [5] * 3
        + [6] * 3
        + [7] * 3
        + [8] * 3
        + [9] * 3,
        "function_id": [1] * 9 + [1] * 9 + [1] * 9,
    }
)


def test_tournament_ranking_basic_call_returns_axes_and_data():
    ax, dt = plot_tournament_ranking(DATA)
    assert ax is not None
    assert dt is not None


def test_robustrank_over_time_basic_call_returns_axes_and_data():
    evals = [1, 10, 100]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        axs, comparison, benchmark = plot_robustrank_over_time(
            OBJ_DATA,
            obj_vars=["f1", "f2", "f3"],
            evals=evals,
            indicator=HyperVolume(reference_point=[5.0, 5.0, 5.0]),
        )
    assert axs is not None
    assert comparison is not None
    assert benchmark is not None


def test_robustrank_changes_basic_call_returns_axes_and_data():
    evals = [1, 10, 100]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ax, dt = plot_robustrank_changes(
            OBJ_DATA,
            obj_vars=["f1", "f2", "f3"],
            evals=evals,
            indicator=HyperVolume(reference_point=[5.0, 5.0, 5.0]),
        )
    assert ax is not None
    assert dt is not None
