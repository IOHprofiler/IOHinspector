import matplotlib
import numpy as np
import polars as pl

from iohinspector.plots.single_run import plot_heatmap_single_run

matplotlib.use("Agg")  # Use non-interactive backend for tests

DATA = pl.DataFrame(
    {
        "run_id": [1] * 5,
        "evaluations": [1, 2, 3, 4, 5],
        "x1": np.linspace(-5, 5, 5),
        "x2": np.linspace(-5, 5, 5)[::-1],
    }
)


def test_basic_call_returns_axes_and_data():
    ax, data = plot_heatmap_single_run(
        data=DATA,
        vars=["x1", "x2"],
        eval_var="evaluations",
        var_mins=np.array([-5, -5]),
        var_maxs=np.array([5, 5]),
    )
    assert ax is not None
    assert data is not None
