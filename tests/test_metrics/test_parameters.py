"""Parameters, positions and constraint penalties (docs/definitions/parameters.md)."""

from functools import cache

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from iohinspector import DataManager
from iohinspector.align import align_data
from iohinspector.metrics import (
    apply_penalty,
    get_logged_rows,
    get_parameter_sample,
    get_parameter_summary,
)
from iohinspector.metrics.parameters import QUANTILES
from tests.golden.load import fixture_path, golden, logged

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
        "p",
        1.0,
        0.0,
        1.0,
        1.0,
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
    assert rows.columns == [
        "run_id",
        "algorithm_name",
        "evaluations",
        "x0",
        "x2",
        "x10",
    ]
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


# Golden tests against IOHanalyzer (WP I03, tests/golden/README.md). Parameters by
# target on IOHexperimenter JSON data are not compared: IOHanalyzer reads the wrong
# rows there (PA-5).
KEY = ["pair_f", "pair_d", "ID", "run"]


@cache
def _rows(fixture):
    """Every logged row of ``fixture`` with IOHanalyzer's keys ``KEY``."""
    if fixture == "extdata":
        manager = DataManager()
        manager.add_dataset(fixture_path(fixture))
        names = ["function_id", "function_name", "dimension", "algorithm_name"]
        df = manager.load(False, include_columns=[*names, "source_run_index"])
        f = pl.coalesce("function_name", pl.col("function_id").cast(pl.String))
    else:
        df = logged(fixture)
        f = pl.col("function_id").cast(pl.Float64)
    return df.rename(lambda c: c.removesuffix("_logged")).with_columns(
        pair_f=f,
        pair_d=pl.col("dimension").cast(pl.Float64),
        ID="algorithm_name",
        run=pl.col("source_run_index") + 1,
    )


def _gold_sample(fixture, which):
    """IOHanalyzer's ``PAR_sample_by_{which}`` as rows ``KEY``, x, ``parId``,
    ``gold``, without the runs that have no value."""
    gold = golden(fixture, f"PAR_sample_by_{which}")
    runs = [c for c in gold.columns if c.startswith("run.")]
    gold = gold.unpivot(runs, index=KEY[:3] + ["parId", gold.columns[4]])
    run = pl.col("variable").str.slice(4).cast(pl.Int64)
    return gold.select(pl.exclude("variable", "value"), run=run, gold="value")


def _ours(df, x_values, pars, target=False, maximization=False):
    """``get_parameter_sample`` in the long form of ``_gold_sample``."""
    x_col, x = ("raw_y", "target") if target else ("evaluations", "runtime")
    sample = get_parameter_sample(
        df, pars, x_values, KEY, target=target, maximization=maximization
    )
    long = sample.unpivot(pars, index=[*KEY, x_col], variable_name="parId")
    return long.select(*KEY, "parId", pl.col(x_col).cast(pl.Float64).alias(x), "value")


@pytest.mark.parametrize("target", [False, True])
def test_samples_match_iohanalyzer_on_legacy_data(target):
    gold = _gold_sample("extdata", "FV" if target else "RT")
    x, pars = gold.columns[4], gold["parId"].unique().to_list()
    ours = _ours(_rows("extdata"), gold[x].unique(), pars, target, maximization=True)
    joined = gold.join(ours, on=[*KEY, "parId", x], how="left")
    assert joined["gold"].to_list() == joined["value"].to_list()


@pytest.mark.parametrize(
    ("fixture", "target"),
    [("extdata", False), ("extdata", True), ("ioh040_cons", False)],
)
def test_summaries_match_iohanalyzer(fixture, target):
    """On legacy data, and on JSON data that logs every evaluation."""
    gold = golden(fixture, f"PAR_summary_by_{'FV' if target else 'RT'}")
    x, x_col = gold.columns[4], "raw_y" if target else "evaluations"
    names = {"runs": "n_defined", x: x_col, "parId": "parameter"}
    gold = gold.rename(names | {f"{q}%": f"q{q}" for q in QUANTILES})
    ours = get_parameter_summary(
        _rows(fixture),
        gold["parameter"].unique().to_list(),
        gold[x_col].unique(),
        by=KEY[:3],
        target=target,
        maximization=fixture == "extdata",
    )
    ours = ours.with_columns(pl.col(x_col).cast(pl.Float64))
    joined = gold.join(ours, on=gold.columns[:5], suffix="_ours")
    assert joined.height == gold.height
    assert (joined["n_defined"] == joined["n_defined_ours"]).all()
    assert (joined["n_defined"] == joined["n_runs"]).all()
    for stat in ("mean", "median", "sd", *(f"q{q}" for q in QUANTILES)):
        np.testing.assert_allclose(joined[f"{stat}_ours"], joined[stat], rtol=1e-12)


@pytest.mark.parametrize(
    "fixture", ["ioh040_min", "ioh040_max", "ioh040_cons", "ioh040_pos"]
)
def test_samples_by_budget_on_json_data(fixture):
    """IOHanalyzer reads, per run, the first row of its RT matrix at or after b, which
    is the run's next logged evaluation, and NA for the runs that did not log it
    (PA-1, PA-5). Ours is the last row at or before b: equal where the run logged b."""
    df = _rows(fixture)
    gold = _gold_sample(fixture, "RT").drop_nulls("gold").sort("runtime")
    logged_at = df.select(*KEY, next=pl.col("evaluations").cast(pl.Float64)).sort(
        "next"
    )
    gold = gold.join_asof(
        logged_at, left_on="runtime", right_on="next", by=KEY, strategy="forward"
    )
    x_values = pl.concat([gold["runtime"], gold["next"]]).unique().cast(pl.UInt64)
    pars = gold["parId"].unique().to_list()
    ours = _ours(df, x_values, pars, maximization=fixture == "ioh040_max")
    joined = gold.join(ours, on=[*KEY, "parId", "runtime"]).join(
        ours.rename({"runtime": "next", "value": "at_next"}), on=[*KEY, "parId", "next"]
    )
    assert joined.height == gold.height
    assert joined["gold"].to_list() == joined["at_next"].to_list()
    at_b = joined.filter(pl.col("runtime") == pl.col("next"))
    assert not at_b.is_empty()
    assert at_b["gold"].to_list() == at_b["value"].to_list()


def test_positions_match_iohanalyzer():
    """``get_position_dsl`` drops each run's last row when its evaluations have gaps,
    and its ``FV`` is the best-so-far value."""
    df = _rows("ioh040_pos").with_columns(best=pl.col("raw_y").cum_min().over("run_id"))
    ours = get_logged_rows(df, ["x0", "x1", "x2", "best"], by=KEY)
    gold = golden("ioh040_pos", "positions").drop_nulls("x0")
    gold = gold.select(
        pl.col("pair_f", "pair_d"),
        ID="algId",
        run=pl.col("run_nr").cast(pl.Int64),
        evaluations=pl.col("runtime").cast(pl.UInt64),
        x0="x0",
        x1="x1",
        x2="x2",
        best="FV",
    )
    assert ours.height == gold.height + df["run_id"].n_unique()
    key = gold.columns[:5]
    ours = ours.drop("run_id").join(gold.select(key), on=key)
    assert_frame_equal(ours.sort(key), gold.sort(key))


def test_correlation_rows_match_iohanalyzer():
    df = _rows("ioh040_cons").filter(ID="es_small")
    ours = get_logged_rows(df, ["raw_y", "sigma"], by=KEY)
    gold = (
        golden("ioh040_cons", "Parameter_correlation")
        .select(
            *KEY,
            evaluations=pl.col("runtime").cast(pl.UInt64),
            raw_y="f(x)",
            sigma="sigma",
        )
        .with_columns(pl.col("run").cast(pl.Int64))
    )
    key = [*KEY, "evaluations"]
    assert_frame_equal(ours.drop("run_id").sort(key), gold.sort(key))


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("none", {"method": "ignore"}),
        ("add", {"method": "relative", "factor": 10}),
        # I03's "absolute" adds v on the rows with v > 0, which is "relative" with
        # c = 1, not IOHanalyzer's "Penalize Absolute" (c on every row)
        ("absolute", {"method": "relative"}),
    ],
)
def test_penalties_match_iohanalyzer(name, options):
    """Penalised, then best-so-far: IOHanalyzer's ``get_FV`` after the penalty."""
    df = _rows("ioh040_cons")
    aligned = align_data(apply_penalty(df, **options), range(1, 201), group_cols=KEY)
    gold = golden("ioh040_cons", f"penalty_{name}_FV_matrix").select(
        ID="algId",
        run=pl.col("run").cast(pl.Int64),
        evaluations=pl.col("row").cast(pl.UInt64),
        gold="fv",
    )
    joined = gold.join(aligned, on=["ID", "run", "evaluations"])
    assert joined.height == gold.height == df.height
    assert (joined["gold"] == joined["raw_y"]).all()
