import math

import polars as pl
import pytest

import iohinspector.align
from iohinspector.align import align_data

TWO_RUNS = pl.DataFrame(
    {
        "run_id": [1, 1, 1, 2, 2, 2],
        "evaluations": [1, 2, 5, 1, 4, 5],
        "raw_y": [10, 8, 6, 20, 18, 16],
    }
)


def _rows(df, *cols):
    return df.sort(["run_id", cols[0]]).select("run_id", *cols).rows()


def test_fixed_budget_minimization():
    result = align_data(TWO_RUNS, [1, 2, 3, 4, 5])
    assert _rows(result, "evaluations", "raw_y") == [
        (1, 1, 10), (1, 2, 8), (1, 3, 8), (1, 4, 8), (1, 5, 6),
        (2, 1, 20), (2, 2, 20), (2, 3, 20), (2, 4, 18), (2, 5, 16),
    ]  # fmt: skip


def test_fixed_budget_computes_best_so_far():
    df = pl.DataFrame(
        {"run_id": [1, 1, 1], "evaluations": [1, 2, 3], "raw_y": [5, 7, 6]}
    )
    assert align_data(df, [1, 2, 3], maximization=True)["raw_y"].to_list() == [5, 7, 7]
    assert align_data(df, [1, 2, 3])["raw_y"].to_list() == [5, 5, 5]


def test_before_the_first_evaluation_is_inf():
    df = pl.DataFrame({"run_id": [1], "evaluations": [3], "raw_y": [1.0]})
    assert align_data(df, [1, 3])["raw_y"].to_list() == [math.inf, 1.0]


@pytest.mark.parametrize(
    ("maximization", "y", "expected"),
    [
        (False, [10, 8, 9, 6], {9: 2, 6: 4, 1: math.inf}),
        (True, [1, 3, 2, 5], {2: 2, 5: 4, 9: math.inf}),
    ],
)
def test_fixed_target_first_hit(maximization, y, expected):
    df = pl.DataFrame({"run_id": [1] * 4, "evaluations": [1, 2, 3, 4], "raw_y": y})
    result = align_data(
        df,
        list(expected),
        x_col="raw_y",
        y_col="evaluations",
        maximization=maximization,
    )
    assert dict(result.select("raw_y", "evaluations").rows()) == expected


def test_wide_output():
    result = align_data(TWO_RUNS, [1, 2, 3, 4, 5], output="wide")
    assert {"evaluations", "1", "2"} == set(result.columns)
    assert result.height == 5


def test_group_columns():
    df = pl.DataFrame(
        {"exp_id": [1, 1, 2, 2], "evaluations": [1, 2, 1, 2], "raw_y": [5, 3, 7, 6]}
    )
    result = align_data(df, [1, 2], group_cols=("exp_id",))
    assert _rows(result.rename({"exp_id": "run_id"}), "evaluations", "raw_y") == [
        (1, 1, 5), (1, 2, 3), (2, 1, 7), (2, 2, 6),
    ]  # fmt: skip


def test_turbo_align_is_gone():
    with pytest.raises(ImportError, match="use align_data"):
        _ = iohinspector.align.turbo_align


def test_removed_arguments_hint():
    with pytest.raises(TypeError, match="no longer warns"):
        align_data(TWO_RUNS, [1], silence_warning=True)
