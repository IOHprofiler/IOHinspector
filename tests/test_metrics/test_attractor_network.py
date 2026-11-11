import polars as pl

from iohinspector.metrics import get_attractor_network


def test_basic():
    data = pl.DataFrame(
        {
            "x1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "x2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
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
    nodes, edges = get_attractor_network(
        data,
        coord_vars=["x1", "x2"],
        fval_var="raw_y",
        eval_var="evaluations",
    )
    # Check nodes DataFrame shape and content
    assert nodes.shape[1] == 5  # x1, x2, y, count, evals
    assert nodes.shape[0] >= 1
    # Check that node coordinates and y values are as expected
    assert "x1" in nodes.columns
    assert "x2" in nodes.columns
    assert "y" in nodes.columns
    assert "count" in nodes.columns
    assert "evals" in nodes.columns
    # Check that the first node matches the first stagnation point
    assert nodes["x1"][0] == 0
    assert nodes["x2"][0] == 0
    assert nodes["y"][0] == 35
    assert nodes["x1"][-1] == 10
    assert nodes["x2"][-1] == 10
    assert nodes["y"][-1] == 10
    # Check that counts and evals are positive
    assert (nodes["count"] > 0).all()
    assert (nodes["evals"] > 0).all()

    # Check edges DataFrame shape and content
    assert edges.shape[1] == 4  # start, end, count, stag_length_avg
    assert (edges["count"] > 0).all()
    assert (edges["stag_length_avg"] > 0).all()
    # Check that start and end refer to valid node indices
    assert edges["start"].is_in(range(nodes.height)).all()
    assert edges["end"].is_in(range(nodes.height)).all()
