"""The legacy reader against IOHanalyzer's parsed extdata (WP I06, step 4).

The references are I03's goldens for the ``extdata`` fixture (IOHanalyzer at
``caecada^``): ``RT_sample`` and ``FV_sample`` are samples of IOHanalyzer's RT and
FV matrices, ``runtimes`` and ``funvals`` their row names, ``maxRT`` the per-run
budgets. Best-so-far values are computed from ``raw_y`` (maximisation, as the
fixture's ``meta`` says); IOHanalyzer reads them from the logged column.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from iohinspector.io import read_legacy
from tests.golden.load import fixture_path, golden


@pytest.fixture(scope="module")
def runs_and_trace() -> tuple[pl.DataFrame, pl.DataFrame]:
    result = read_legacy(fixture_path("extdata"), "extdata")
    runs = result.runs.with_columns(
        funcId=pl.coalesce("function_name", pl.col("function_id").cast(pl.String)),
        run=(pl.col("source_run_index") + 1).cast(pl.Int32),
    )
    trace = result.trace.join(runs.select("run_id", "funcId", "run"), on="run_id")
    trace = trace.with_columns(best=pl.col("raw_y").cum_max().over("run_id"))
    assert golden("extdata", "meta")["maximization"].to_list() == [True]
    assert runs["maximization"].all()
    return runs, trace


def test_logged_best_so_far_is_the_running_best(runs_and_trace):
    _, trace = runs_and_trace
    assert (trace["best"] == trace["best-so-far f(x)"]).all()


def test_maxrt(runs_and_trace):
    runs, trace = runs_and_trace
    last = trace.group_by("run_id").agg(last=pl.col("evaluations").max())
    ours = runs.join(last, on="run_id").select(
        "funcId", "run", maxRT=pl.max_horizontal("evals_used", "last").cast(pl.Int32)
    )
    ref = golden("extdata", "maxRT").select("funcId", "run", "maxRT")
    assert_frame_equal(ours.sort("funcId", "run"), ref.sort("funcId", "run"))


def test_runtimes_and_funvals(runs_and_trace):
    _, trace = runs_and_trace
    for name, column, ours in (
        ("runtimes", "rt", pl.col("evaluations").cast(pl.Float64)),
        ("funvals", "f", pl.col("best")),
    ):
        ref = golden("extdata", name).select("funcId", column).sort("funcId", column)
        got = trace.select("funcId", ours.alias(column)).unique().sort("funcId", column)
        assert_frame_equal(got, ref, check_exact=True)


def test_rt_sample(runs_and_trace):
    _, trace = runs_and_trace
    ref = golden("extdata", "RT_sample")
    targets = ref.select("funcId", "target").unique()
    hits = (
        trace.join(targets, on="funcId")
        .filter(pl.col("best") >= pl.col("target"))
        .group_by("funcId", "target", "run")
        .agg(RT=pl.col("evaluations").min().cast(pl.Int32))
    )
    ref = ref.select("funcId", "target", "run", "RT")
    key = ["funcId", "target", "run"]
    assert_frame_equal(hits.sort(key), ref.sort(key))


def test_fv_sample(runs_and_trace):
    _, trace = runs_and_trace
    ref = golden("extdata", "FV_sample")
    budgets = ref.select("funcId", "runtime").unique()
    values = (
        trace.join(budgets, on="funcId")
        .filter(pl.col("evaluations") <= pl.col("runtime"))
        .sort("row")
        .group_by("funcId", "runtime", "run")
        .agg(pl.col("best").last().cast(pl.Int32).alias("f(x)"))
    )
    ref = ref.select("funcId", "runtime", "run", "f(x)")
    key = ["funcId", "runtime", "run"]
    assert_frame_equal(values.sort(key), ref.sort(key))
