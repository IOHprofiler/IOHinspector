import numpy as np
import polars as pl
import pytest

from iohinspector.metrics.single_run import get_heatmap_single_run_data

DATA = pl.DataFrame(
    {
        "run_id": [1] * 5,
        "evaluations": [1, 2, 3, 4, 5],
        "x1": np.linspace(-5, 5, 5),
        "x2": np.linspace(-5, 5, 5)[::-1],
    }
)


def test_basic():
    dt_plot = get_heatmap_single_run_data(
        data=DATA,
        vars=["x1", "x2"],
        eval_var="evaluations",
        var_mins=np.array([-5, -5]),
        var_maxs=np.array([5, 5]),
    )
    assert dt_plot.shape == (2, 5)
    assert dt_plot.to_numpy().min() == pytest.approx(0)
    assert dt_plot.to_numpy().max() == pytest.approx(1)
    assert np.all((dt_plot.to_numpy() >= 0) & (dt_plot.to_numpy() <= 1))


def test_asserts_on_multiple_run_ids():
    data = pl.DataFrame(
        {
            "run_id": [1, 2],
            "evaluations": [1, 2],
            "x1": [0, 1],
        }
    )
    with pytest.raises(AssertionError):
        get_heatmap_single_run_data(data, ["x1"])


def test_single_variable():
    data = pl.DataFrame(
        {
            "run_id": [1] * 3,
            "evaluations": [1, 2, 3],
            "x1": [-5, 0, 5],
        }
    )
    dt_plot = get_heatmap_single_run_data(
        data=data,
        vars=["x1"],
        eval_var="evaluations",
        var_mins=[-5],
        var_maxs=[5],
    )
    assert dt_plot.shape == (1, 3)
    np.testing.assert_allclose(dt_plot.to_numpy(), [[0, 0.5, 1]])


def test_non_default_eval_col():
    data = pl.DataFrame(
        {
            "run_id": [1] * 4,
            "evals": [1, 2, 3, 4],
            "x1": [0, 1, 2, 3],
            "x2": [3, 2, 1, 0],
        }
    )
    dt_plot = get_heatmap_single_run_data(
        data=data,
        vars=["x1", "x2"],
        eval_var="evals",
        var_mins=[0, 0],
        var_maxs=[3, 3],
    )
    assert dt_plot.shape == (2, 4)
