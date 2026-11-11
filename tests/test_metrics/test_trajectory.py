from itertools import pairwise

import polars as pl

from iohinspector.metrics import get_trajectory

# Example data with two algorithms, two run_ids, and three evaluations each
DATA = pl.DataFrame(
    {
        "run_id": [1, 1, 1, 2, 2, 2],
        "algorithm_name": ["A", "A", "A", "B", "B", "B"],
        "evaluations": [1, 10, 20, 1, 10, 20],
        "raw_y": [0.5, 0.4, 0.3, 1.0, 0.9, 0.7],
    }
)


def test_basic_trajectory():
    result = get_trajectory(DATA)
    assert isinstance(result, pl.DataFrame)
    # Should have as many rows as input (since all evaluations present)
    assert result.shape[0] == 40  # 2 algorithms * 20 evaluations
    assert "evaluations" in result.columns
    assert "raw_y" in result.columns
    # Check that all evaluation points are present
    for algo in DATA["algorithm_name"].unique():
        evals = result.filter(pl.col("algorithm_name") == algo)["evaluations"].to_list()
        assert set(evals) == set(range(1, 21))
    # Check that raw_y is non-increasing for each algorithm
    for algo in DATA["algorithm_name"].unique():
        raw_y_values = (
            result.filter(pl.col("algorithm_name") == algo)
            .sort("evaluations")["raw_y"]
            .to_list()
        )
        assert all(x >= y for x, y in pairwise(raw_y_values))


def test_traj_length():
    # Only first two evaluations should be present
    result = get_trajectory(DATA, traj_length=1)
    for algo in DATA["algorithm_name"].unique():
        evals = result.filter(pl.col("algorithm_name") == algo)["evaluations"].to_list()
        assert set(evals) == set(range(1, 3))

    result = get_trajectory(DATA, traj_length=10)
    for algo in DATA["algorithm_name"].unique():
        evals = result.filter(pl.col("algorithm_name") == algo)["evaluations"].to_list()
        assert set(evals) == set(range(1, 12))


def test_min_fevals():
    # Start from evaluation 2
    result = get_trajectory(DATA, min_fevals=2)
    for algo in DATA["algorithm_name"].unique():
        evals = result.filter(pl.col("algorithm_name") == algo)["evaluations"].to_list()
        assert set(evals) == set(range(2, 21))


def test_custom_free_variables():
    # Use only run_id as free variable
    result = get_trajectory(DATA, by=[])
    assert "run_id" in result.columns
    assert "raw_y" in result.columns


def test_maximization():
    result = get_trajectory(DATA, maximization=True)

    for algo in DATA["algorithm_name"].unique():
        raw_y_values = (
            result.filter(pl.col("algorithm_name") == algo)
            .sort("evaluations")["raw_y"]
            .to_list()
        )
        assert all(x <= y for x, y in pairwise(raw_y_values))
