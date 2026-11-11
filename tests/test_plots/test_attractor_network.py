import matplotlib
import polars as pl

from iohinspector.plots import plot_attractor_network

matplotlib.use("Agg")  # Use non-interactive backend for tests

DATA = pl.DataFrame(
    {
        "x1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "x2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 2],
        "x3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "raw_y": [35, 33, 31, 29, 27, 23, 18, 16, 14, 12, 10, 9, 6],
        "evaluations": [
            1,
            42,
            81,
            121,
            161,
            201,
            241,
            281,
            321,
            361,
            401,
            442,
            481,
        ],
        "run_id": [1] * 13,
    }
)


def test_basic_call_returns_axes_and_data():
    ax, nodes, edges = plot_attractor_network(
        DATA,
        coord_vars=["x1", "x2", "x3"],
        fval_var="raw_y",
        eval_var="evaluations",
    )

    assert ax is not None
    assert nodes is not None
    assert edges is not None
