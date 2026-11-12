"""Parameters, positions and constraint penalties (docs/definitions/parameters.md)."""

import polars as pl
import pytest

from iohinspector.align import align_data
from iohinspector.metrics import (
    apply_penalty,
    get_logged_rows,
    get_parameter_sample,
    get_parameter_summary,
)

# Run 1 logs evaluation 3 twice; its parameter p is not monotone. Run 2 starts at 2,
# and its f = 6 at evaluation 4 ties its best without improving it.
RUNS = pl.DataFrame(
    {
        "run_id": [1, 1, 1, 1, 2, 2, 2],
        "algorithm_name": ["A"] * 4 + ["B"] * 3,
        "evaluations": [1, 3, 3, 5, 2, 3, 4],
        "raw_y": [9.0, 7.0, 8.0, 5.0, 8.0, 6.0, 6.0],
        "p": [0.5, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0],
    }
)


def _values(df, x_col):
    return df.sort("run_id", x_col).select("run_id", x_col, "p").rows()


def test_parameter_at_budget_is_the_last_row_carried_forward():
    sample = get_parameter_sample(RUNS, ["p"], [1, 3, 4, 6], by=[])
    assert _values(sample, "evaluations") == [
        (1, 1, 0.5), (1, 3, 1.0), (1, 4, 1.0), (1, 6, 3.0),
        (2, 1, None), (2, 3, 5.0), (2, 4, 6.0), (2, 6, 6.0),
    ]  # fmt: skip


def test_parameter_at_target_is_the_first_row_that_reaches_it():
    sample = get_parameter_sample(RUNS, ["p"], [8.0, 6.0, 5.0], target=True, by=[])
    assert _values(sample, "raw_y") == [
        (1, 5.0, 3.0), (1, 6.0, 3.0), (1, 8.0, 2.0),
        (2, 5.0, None), (2, 6.0, 5.0), (2, 8.0, 4.0),
    ]  # fmt: skip


def test_parameter_at_target_under_maximisation():
    sample = get_parameter_sample(
        RUNS, ["p"], [7.0, 8.0], target=True, by=[], maximization=True
    )
    assert _values(sample, "raw_y") == [
        (1, 7.0, 0.5), (1, 8.0, 0.5), (2, 7.0, 4.0), (2, 8.0, 4.0),
    ]  # fmt: skip


def test_summary_over_the_runs_with_a_value():
    df = pl.concat([RUNS, RUNS.with_columns(run_id=pl.col("run_id") + 2)])
    summary = get_parameter_summary(df, ["p"], [1, 4])
    assert summary.select("algorithm_name", "n_runs", "n_defined", "mean").rows() == [
        ("A", 2, 2, 0.5), ("A", 2, 2, 1.0), ("B", 2, 0, None), ("B", 2, 2, 6.0),
    ]  # fmt: skip
    row = summary.filter(algorithm_name="A", evaluations=4)
    assert row.select("parameter", "median", "sd", "q2", "q98").row(0) == (
        "p", 1.0, 0.0, 1.0, 1.0
    )


def test_summary_quantiles_are_type_7():
    df = pl.DataFrame(
        {"run_id": [1, 2, 3, 4], "evaluations": [1] * 4, "raw_y": [0.0] * 4}
    ).with_columns(p=pl.Series([1.0, 2.0, 4.0, 8.0]), algorithm_name=pl.lit("A"))
    row = get_parameter_summary(df, ["p"], [1]).row(0, named=True)
    # R: quantile(c(1, 2, 4, 8), c(.1, .25, .5, .98)) = 1.3, 1.75, 3, 7.76
    assert [row[q] for q in ("q10", "q25", "q50")] == pytest.approx([1.3, 1.75, 3])
    assert row["q98"] == pytest.approx(7.76)
    assert row["sd"] == pytest.approx(pl.Series([1.0, 2.0, 4.0, 8.0]).std())


def test_logged_rows_default_to_the_positions():
    df = RUNS.with_columns(
        x10=pl.lit(1.0), x2=pl.lit(2.0), x0=pl.Series([0.0, None] * 3 + [0.0])
    )
    rows = get_logged_rows(df)
    assert rows.columns == ["run_id", "algorithm_name", "evaluations", "x0", "x2", "x10"]
    assert rows["evaluations"].to_list() == [1, 3, 2, 4]
    pairs = get_logged_rows(RUNS, ["p", "raw_y"], by=["run_id"])
    assert pairs.columns == ["run_id", "evaluations", "p", "raw_y"]


V = pl.DataFrame({"raw_y": [1.0, 1.0, 1.0, 1.0], "violation": [0.0, 2.0, -3.0, 0.5]})


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"method": "ignore"}, [1, 1, 1, 1]),
        ({"method": "relative", "factor": 10}, [1, 21, 31, 6]),
        ({"method": "relative", "exponent": 2}, [1, 5, 10, 1.25]),
        ({"method": "absolute", "factor": 10}, [1, 11, 11, 11]),
        ({"method": "value", "factor": 100}, [1, 100, 100, 100]),
        ({"feasible": "nonpositive"}, [1, 3, 1, 1.5]),
        ({"feasible": lambda v: v <= 1}, [1, 3, 1, 1]),
        ({"maximization": True}, [1, -1, -2, 0.5]),
        ({"method": "absolute", "maximization": True}, [1, 0, 0, 0]),
    ],
)
def test_penalty(options, expected):
    assert apply_penalty(V, **options)["raw_y"].to_list() == expected


def test_penalty_comes_before_the_best_so_far():
    df = pl.DataFrame(
        {
            "run_id": [1, 1, 1],
            "evaluations": [1, 2, 3],
            "raw_y": [5.0, 1.0, 3.0],
            "violation": [0.0, 9.0, 0.0],
        }
    )
    aligned = align_data(apply_penalty(df), [1, 2, 3])
    assert aligned.sort("evaluations")["raw_y"].to_list() == [5.0, 5.0, 3.0]


def test_penalty_errors():
    with pytest.raises(ValueError, match="method"):
        apply_penalty(V, method="add")
    with pytest.raises(ValueError, match="feasible"):
        apply_penalty(V, feasible="positive")
    with pytest.raises(ValueError, match="violation"):
        apply_penalty(V.with_columns(violation=None))
