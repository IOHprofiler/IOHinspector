import math

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import iohinspector.align
from iohinspector.align import align_data
from iohinspector.metrics import (
    aggregate_convergence,
    get_data_ecdf,
    get_discritized_eaf_single_objective,
    get_eaf_data,
    get_parameter_sample,
    get_trajectory,
    get_value_sample,
)
from tests.golden.load import golden_renamed, logged

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


# AL-5: run 2 is first logged at evaluation 5, so it has no value at budget 1
LATE = pl.DataFrame(
    {
        "run_id": [1, 1, 2, 2],
        "algorithm_name": ["A"] * 4,
        "evaluations": [1, 10, 5, 10],
        "raw_y": [4.0, 2.0, 3.0, 1.0],
    }
)
DIRECTIONS = pytest.mark.parametrize("maximization", [False, True])


def _at(df, evaluation, run_id=None):
    df = df.filter(pl.col("evaluations") == evaluation)
    return df if run_id is None else df.filter(pl.col("run_id") == run_id)


@DIRECTIONS
def test_before_the_first_evaluation_is_null(maximization):
    df = align_data(LATE, [1, 5], maximization=maximization).sort("run_id")
    assert _at(df, 1)["raw_y"].to_list() == [4.0, None]
    assert _at(df, 5)["raw_y"].to_list() == [4.0, 3.0]
    wide = align_data(LATE, [1], output="wide", maximization=maximization)
    assert wide.select("1", "2").row(0) == (4.0, None)


@DIRECTIONS
def test_stopped_from_evals_used(maximization):
    # AL-9: run 1 used 10 evaluations but last logged 5, run 2 used 12; no rows added
    df = LATE.with_columns(evals_used=pl.Series([10, 10, 12, 12]))
    df = df.with_columns(evaluations=pl.Series([1, 5, 5, 10]))
    result = align_data(df, [5, 10, 11], maximization=maximization).sort("run_id")
    assert result.height == 6
    assert result["stopped"].to_list() == [False, False, True, False, False, False]
    assert _at(result, 11, run_id=1)["raw_y"].item() == (4.0 if maximization else 2.0)
    assert "stopped" not in align_data(LATE, [5], maximization=maximization).columns
    summary = aggregate_convergence(df, eval_max=12, maximization=maximization)
    assert _at(summary, 10)["n_active"].item() == 2
    assert _at(summary, 12)["n_active"].item() == 1


@DIRECTIONS
def test_fixed_budget_leaves_out_but_counts(maximization):
    result = _at(aggregate_convergence(LATE, maximization=maximization), 1)
    assert result.select("mean", "best", "worst", "n_runs", "n_defined").row(0) == (
        4.0, 4.0, 4.0, 2, 1,
    )  # fmt: skip


@DIRECTIONS
def test_ecdf_counts_a_run_without_a_value_as_reaching_nothing(maximization):
    result = get_data_ecdf(
        LATE,
        f_min=1,
        f_max=4,
        scale_f_log=False,
        eval_values=[1],
        maximization=maximization,
    )
    reached = 1.0 if maximization else 0.0  # run 1 is at 4, the bound
    assert result["eaf"].to_list() == [reached / 2]


@DIRECTIONS
def test_eaf_counts_a_run_without_a_value_as_reaching_nothing(maximization):
    result = get_discritized_eaf_single_objective(
        LATE,
        eval_values=[1],
        f_min=1,
        f_max=4,
        scale_f_log=False,
        f_targets=2,
        maximization=maximization,
    )
    assert result["1"].to_list() == [1.0, 0.5 if maximization else 0.0]
    aligned = get_eaf_data(LATE, eval_min=1, eval_max=1, maximization=maximization)
    assert _at(aligned, 1, run_id=2)["raw_y"].to_list() == [None]


@DIRECTIONS
def test_trajectory_is_null_before_the_first_evaluation(maximization):
    result = get_trajectory(LATE, maximization=maximization)
    assert _at(result, 4, run_id=2)["raw_y"].to_list() == [None]
    assert _at(result, 5, run_id=2)["raw_y"].to_list() == [3.0]


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


@DIRECTIONS
def test_fixed_target_inclusive_first_of_ties(maximization):
    # AL-1: f <= t reaches t; AL-3: the first of the rows with f = 5; AL-6: raw rows
    sign = -1 if maximization else 1
    y = [10.0, 5.0, 5.0, 7.0, 5.0, 2.0]
    df = pl.DataFrame({"run_id": [1] * 6, "evaluations": range(1, 7), "raw_y": y})
    expected = {6.0: 2, 5.0: 2, 4.9: 6, 2.0: 6, 1.0: math.inf}
    result = align_data(
        df.with_columns(pl.col("raw_y") * sign),
        [t * sign for t in expected],
        x_col="raw_y",
        y_col="evaluations",
        maximization=maximization,
    )
    rows = result.select(pl.col("raw_y") * sign, "evaluations").rows()
    assert dict(rows) == expected


def test_repeated_values_count_once():
    # targets from a table of several pairs repeat; each run is counted once
    assert align_data(TWO_RUNS, [5, 5, 1]).height == 4
    assert (
        align_data(TWO_RUNS, [8.0, 8.0], x_col="raw_y", y_col="evaluations").height == 2
    )
    summary = aggregate_convergence(
        TWO_RUNS.with_columns(algorithm_name=pl.lit("A")), budgets=[5, 5]
    )
    assert summary["n_runs"].to_list() == [2]


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


# Golden tests against IOHanalyzer's runtime and value samples (WP I03): the
# hitting times at five targets and the values at five budgets of every run.
RUN = ["function_id", "dimension", "algorithm_name", "run"]


@pytest.mark.parametrize("fixture", ["ioh040_min", "ioh040_max"])
@pytest.mark.parametrize(
    ("metric", "x_col", "y_col"),
    [("RT_sample", "raw_y", "evaluations"), ("FV_sample", "evaluations", "raw_y")],
)
def test_samples_match_iohanalyzer(fixture, metric, x_col, y_col):
    df = logged(fixture).with_columns(
        pl.col("evaluations").cast(pl.Int64), run=pl.col("source_run_index") + 1
    )
    names = {"target": "raw_y", "runtime": "evaluations", "RT": "gold", "f(x)": "gold"}
    gold = golden_renamed(fixture, metric).rename(names, strict=False)
    gold = gold.cast({x_col: pl.Int64} if x_col == "evaluations" else {})
    pair = RUN[:2]
    ours = pl.concat(
        align_data(
            group,
            gold.join(group.select(pair).unique(), on=pair)[x_col].unique(),
            group_cols=["run_id", *RUN],
            x_col=x_col,
            y_col=y_col,
            maximization=fixture == "ioh040_max",
        )
        for _, group in df.group_by(pair)
    )
    # a target never reached: inf here, NA in IOHanalyzer
    ours = ours.select(*RUN, x_col, ours=pl.col(y_col).cast(pl.Float64))
    ours = ours.with_columns(pl.col("ours").replace(math.inf, None))
    joined = gold.join(ours, on=[*RUN, x_col], how="left")
    assert joined.height == gold.height
    assert joined["ours"].is_null().to_list() == joined["gold"].is_null().to_list()
    joined = joined.drop_nulls("gold")
    # exact, but IOHanalyzer parses one logged value 1 ulp off (213348.3998485958)
    np.testing.assert_array_max_ulp(joined["ours"], joined["gold"].cast(pl.Float64), 1)


def _renamed(frame, old="evaluations", new="evals"):
    return frame.rename({old: new}) if old in frame.columns else frame


@pytest.mark.parametrize(
    "metric",
    [
        lambda d, e: aggregate_convergence(d, eval_var=e, budgets=[1, 3, 5]),
        lambda d, e: get_value_sample(d, [1, 3, 5], eval_var=e),
        lambda d, e: get_parameter_sample(d, ["p"], [1, 3, 5], eval_var=e),
        lambda d, e: get_trajectory(d, evaluation_variable=e),
        lambda d, e: get_eaf_data(d, eval_var=e),
        lambda d, e: get_data_ecdf(d, eval_var=e, eval_values=[1, 3, 5]),
        lambda d, e: get_discritized_eaf_single_objective(
            d, eval_var=e, eval_values=[1, 3, 5]
        ),
    ],
)
def test_any_name_of_the_evaluations_column(metric):
    # the fixed-budget view is chosen by eval_var, not by the name "evaluations"
    data = TWO_RUNS.with_columns(algorithm_name=pl.lit("A"), p=pl.col("raw_y"))
    expected = _renamed(metric(data, "evaluations"))
    assert_frame_equal(metric(_renamed(data), "evals"), expected, check_row_order=False)
