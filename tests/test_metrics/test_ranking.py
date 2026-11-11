import polars as pl

from iohinspector.indicators import HyperVolume
from iohinspector.metrics import (
    get_robustrank_changes,
    get_robustrank_over_time,
    get_tournament_ratings,
)

DATA = pl.DataFrame(
    {
        "algorithm_name": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "function_name": ["f1", "f2", "f3", "f1", "f2", "f3", "f1", "f2", "f3"],
        "raw_y": [1.0, 2.0, 1.7, 1.5, 2.8, 2.1, 0.9, 0.5, 1.6],
    }
)

# Multi-objective data with different targets and ranks
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
    }
)


def test_tournament_ratings_basic():
    result = get_tournament_ratings(DATA)
    assert isinstance(result, pl.DataFrame)
    assert "Rating" in result.columns
    assert "Deviation" in result.columns
    assert "algorithm_name" in result.columns
    assert len(result) == 3  # Three algorithms
    # Check that algorithms are ordered by rating: C, A, B
    sorted_algos = result.sort("Rating", descending=True)["algorithm_name"].to_list()
    assert sorted_algos == ["C", "A", "B"]


def test_basic_maximisation():
    result = get_tournament_ratings(DATA, maximization=True)
    assert isinstance(result, pl.DataFrame)
    assert "Rating" in result.columns
    assert "Deviation" in result.columns
    assert "algorithm_name" in result.columns
    assert len(result) == 3  # Three algorithms
    # Check that algorithms are ordered by rating: C, A, B
    sorted_algos = result.sort("Rating", descending=True)["algorithm_name"].to_list()
    assert sorted_algos == ["B", "A", "C"]


def test_single_function():
    data = pl.DataFrame(
        {
            "algorithm_name": ["A", "B"],
            "function_name": ["f1", "f1"],
            "raw_y": [1.0, 2.0],
        }
    )
    result = get_tournament_ratings(data, nrounds=25)
    assert len(result) == 2
    assert set(result["algorithm_name"]) == {"A", "B"}


def test_robustrank_over_time_basic():
    evals = [1, 10, 100]
    _comparison, _benchmark = get_robustrank_over_time(
        OBJ_DATA,
        obj_vars=["f1", "f2", "f3"],
        evals=evals,
        indicator=HyperVolume(reference_point=[5.0, 5.0, 5.0]),
    )


def test_robustrank_changes_basic():
    evals = [1, 10, 100]
    result = get_robustrank_changes(
        OBJ_DATA,
        obj_vars=["f1", "f2", "f3"],
        evals=evals,
        indicator=HyperVolume(reference_point=[5.0, 5.0, 5.0]),
    )
    for eval in evals:
        assert str(eval) in result
