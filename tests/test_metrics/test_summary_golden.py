"""Golden tests of the runtime and fixed-budget summaries and the overview tables
against IOHanalyzer (WP I03) and cocopp, ``docs/definitions/runtime.md`` and
``fixed-budget.md``.

IOHanalyzer charges every failed run the largest budget of the run set, for JSON
data its largest logged evaluation; RT-1 charges each run its own ``evals_used``.
The IOHanalyzer comparisons therefore set ``evals_used`` to that largest logged
evaluation (the IOHanalyzer budget rule of the runtime page's test tolerance).
"""

import warnings

import numpy as np
import polars as pl
import pytest

from iohinspector import DataManager
from iohinspector.metrics import (
    aggregate_convergence,
    aggregate_running_time,
    get_overview,
)
from tests.golden.load import fixture_path, golden_cocopp, golden_renamed, logged

SET = ["function_id", "dimension", "algorithm_name"]
PAIR = SET[:2]
FIXTURES = ["ioh040_min", "ioh040_max"]
#: IOHanalyzer's quantile columns and ours
QUANTILES = {f"{p}%": f"q{p}" for p in (2, 5, 10, 25, 50, 75, 90, 95, 98)}


def _data(fixture: str) -> pl.DataFrame:
    """The logged rows, with IOHanalyzer's budget rule as ``evals_used``."""
    df = logged(fixture).with_columns(pl.col("evaluations").cast(pl.Int64))
    return df.with_columns(evals_used=pl.col("evaluations").max().over(SET))


def _per_pair(fixture, gold, column, metric, values, **options):
    """``metric`` on each (function, dimension) pair, by run set, with the argument
    ``values`` set to the pair's ``column`` values of ``gold``."""
    return pl.concat(
        metric(
            group,
            by=SET,
            maximization=fixture == "ioh040_max",
            **{
                values: gold.join(group.select(PAIR).unique(), on=PAIR)[column].unique()
            },
            **options,
        )
        for _, group in _data(fixture).group_by(PAIR)
    )


def _close(joined: pl.DataFrame, pairs: dict[str, str], rtol: float = 1e-12):
    """Columns ``gold`` and ``ours`` of ``pairs`` agree: nulls (and IOHanalyzer's
    NaN for an empty mean) in the same places, the rest to ``rtol``. Our column
    has the suffix ``_right`` when the names are the same."""
    for gold, ours in pairs.items():
        g = joined[gold].cast(pl.Float64).fill_nan(None)
        o = joined[f"{ours}_right" if ours == gold else ours].cast(pl.Float64)
        assert g.is_null().to_list() == o.is_null().to_list(), (gold, ours)
        np.testing.assert_allclose(
            o.drop_nulls(), g.drop_nulls(), rtol=rtol, err_msg=gold
        )


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("k", [1, 10])
def test_runtime_summary_matches_iohanalyzer(fixture, k):
    gold = golden_renamed(fixture, f"RT_summary_par{k}").rename({"target": "raw_y"})
    ours = _per_pair(
        fixture,
        gold,
        "raw_y",
        aggregate_running_time,
        "targets",
        k=k,
        include_unfinished=False,
    )
    joined = gold.join(ours, on=[*SET, "raw_y"], how="left")
    assert joined.height == gold.height
    assert (joined["n_runs"] == 5).all()
    assert joined["n_success"].to_list() == joined["runs"].cast(pl.Int64).to_list()
    columns = ["mean", "median", "sd", "ERT"]
    _close(
        joined,
        {"ps": "success_ratio", f"PAR-{k}": f"par_{k}", **dict(zip(columns, columns))},
    )
    # quantiles are observed run lengths: exact
    _close(joined, QUANTILES, rtol=0)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_runtime_quantiles_include_unfinished_runs(fixture):
    # RT-5: by default a failure counts as inf, so a quantile is inf above the success
    # rate and IOHanalyzer's (successful runs only) where every run succeeded
    gold = golden_renamed(fixture, "RT_summary_par1").rename({"target": "raw_y"})
    ours = _per_pair(fixture, gold, "raw_y", aggregate_running_time, "targets")
    joined = gold.join(ours, on=[*SET, "raw_y"])
    _close(joined.filter(pl.col("ps") == 1), QUANTILES, rtol=0)
    some = joined.filter(pl.col("ps") < 1)
    assert some.height > 0
    assert (some["q98"] == np.inf).all()
    assert some.filter(pl.col("ps") < 0.4)["q50"].is_infinite().all()


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fixed_budget_summary_matches_iohanalyzer(fixture):
    # every run logs evaluation 1, so IOHanalyzer's fill before a run's first row
    # (FB-5) changes nothing here
    gold = golden_renamed(fixture, "FV_summary_geom").rename({"runtime": "evaluations"})
    gold = gold.cast({"evaluations": pl.Int64})
    ours = _per_pair(fixture, gold, "evaluations", aggregate_convergence, "budgets")
    joined = gold.join(ours, on=[*SET, "evaluations"], how="left")
    assert joined.height == gold.height
    assert joined["n_runs"].to_list() == joined["runs"].cast(pl.UInt32).to_list()
    assert (joined["n_defined"] == joined["n_runs"]).all()
    columns = ["mean", "median", "sd"]
    _close(joined, {**dict(zip(columns, columns)), **QUANTILES})
    # FB-3: no geometric mean when a value is <= 0; IOHanalyzer's is then 0 or NaN
    positive = joined.filter(pl.col("geometric_mean").is_not_null())
    _close(positive, {"geometric mean": "geometric_mean"})
    others = joined.filter(pl.col("geometric_mean").is_null())
    assert ((others["geometric mean"] == 0) | others["geometric mean"].is_nan()).all()


OVERVIEW = {
    "FV_overview": {
        "worst recorded": "worst_recorded",
        "worst reached": "worst_reached",
        "best reached": "best_reached",
        "mean reached": "mean_reached",
        "median reached": "median_reached",
        "runs": "n_runs",
        "succ": "n_best",
        "budget": "budget",
    },
    "RT_overview_par1": {
        "miminal runtime": "min_runtime",
        "maximal runtime": "max_runtime",
        "runs": "n_runs",
        "Budget": "budget",
    },
    "overview_par1": {"max evals used": "max_runtime"},
}


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("metric", OVERVIEW)
def test_overview_matches_iohanalyzer(fixture, metric):
    gold = golden_renamed(fixture, metric)
    ours = get_overview(_data(fixture), by=SET, maximization=fixture == "ioh040_max")
    joined = gold.join(ours, on=SET, how="left")
    assert joined.height == gold.height == ours.height
    _close(joined, OVERVIEW[metric])


def test_overview_budget_is_evals_used():
    # FB-4: the budget is the evaluations the runs used, not the last logged one
    df = pl.DataFrame(
        {
            "run_id": [1, 1, 2],
            "evaluations": [1, 3, 2],
            "raw_y": [5.0, 1.0, 2.0],
            "algorithm_name": "A",
            "evals_used": [10, 10, 7],
        }
    )
    row = get_overview(df).row(0, named=True)
    assert row["budget"] == 10
    assert row["max_runtime"] == 3
    assert get_overview(df.drop("evals_used"))["budget"].item() == 3


def test_ert_matches_cocopp():
    # RT-1 is cocopp's rule: a failed run is charged its own evaluations. cocopp
    # rounds three of the targets 10^(2 - k/5) to its next stricter grid level
    # (alignment.md, "How cocopp implements it"; AL-2 keeps targets exact), so
    # there it matches our next target.
    manager = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager.add_folder(fixture_path("coco"))
    df = manager.load(monotonic=False, include_columns=["evals_used"])  # raw_y is Δf
    gold = golden_cocopp("art_51targets").sort("target", descending=True)
    ours = aggregate_running_time(df, by=[], targets=gold["target"])
    ours = ours.sort("raw_y", descending=True)
    rows = [k + (k in (8, 11, 16)) for k in range(gold.height)]
    ours = ours[rows].select(pl.all().name.suffix("_right"))
    joined = pl.concat([gold, ours], how="horizontal")
    assert joined["n_runs"].to_list() == joined["n_runs_right"].to_list()
    assert joined["n_success"].to_list() == joined["n_success_right"].to_list()
    _close(joined, {"ert": "ERT_right", "mean_evals_success": "mean_right"})
