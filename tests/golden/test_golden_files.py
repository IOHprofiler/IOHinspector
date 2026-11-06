"""Sanity checks of the committed golden references (WP I03).

They check that the files exist and are self-consistent, using arithmetic that shares nothing with
IOHanalyzer. Comparisons against iohinspector belong to the metric work packages (I10 onwards).
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from tests.golden.load import (
    FIXTURES,
    call,
    cocopp_versions,
    fixture_path,
    golden,
    golden_cocopp,
    metrics,
    versions,
)

# metrics that every fixture with at least two algorithms must have (parity-matrix rows)
CORE = [
    "RT_sample", "FV_sample", "RT_summary_par1", "RT_summary_par10", "FV_summary", "ERT",
    "targets_log_linear", "budgets_log", "ECDF_by_RT", "AUC_by_RT", "hist_by_RT", "EAF",
]
MULTI = ["Aggr_funcId", "Heatmaps_by_FV", "KS_by_FV", "glicko2_by_FV", "glicko2_games_by_FV",
         "shapley_log", "CDP_by_RT", "EAF_Difference", "EAF_diff_Approximate"]


def test_versions_record_pinned_environment():
    v = versions()
    assert v["IOHanalyzer_git"].startswith("caecada^")
    for pkg in ("IOHanalyzer", "PlayerRatings", "eaf", "RVCompare", "arrow"):
        assert v["packages"][pkg]
    assert "2024-04-23" in v["cran_snapshot"]
    assert cocopp_versions()["cocopp"] == "2.9.1"


def test_every_fixture_exists_and_has_references():
    for name in FIXTURES:
        assert fixture_path(name).is_dir()
        assert metrics(name), name


@pytest.mark.parametrize("fixture", ["ioh040_min", "ioh040_max"])
def test_core_metrics_present(fixture):
    have = set(metrics(fixture))
    assert set(CORE + MULTI) <= have
    for m in CORE + MULTI:
        assert call(fixture, m)


def test_skips_are_recorded_not_missing():
    for key in versions()["skipped"]:
        fixture, rest = key.split("/", 1)
        assert not (rest.endswith(".parquet") and golden_exists(fixture, rest[: -len(".parquet")]))


def golden_exists(fixture, metric):
    try:
        golden(fixture, metric)
    except FileNotFoundError:
        return False
    return True


@pytest.mark.parametrize("fixture", ["ioh040_min", "ioh040_max"])
def test_ert_equals_mean_when_all_runs_succeed(fixture):
    rt = golden(fixture, "RT_sample")
    summ = golden(fixture, "RT_summary_par1")
    grouped = rt.group_by(["ID", "funcId", "DIM", "target"]).agg(
        pl.col("RT").mean().alias("mean_rt"),
        pl.col("RT").null_count().alias("failed"),
        pl.len().alias("n"),
    ).filter(pl.col("failed") == 0)
    joined = grouped.join(summ, on=["ID", "funcId", "DIM", "target"])
    assert joined.height > 0
    for row in joined.iter_rows(named=True):
        assert math.isclose(row["ERT"], row["mean_rt"], rel_tol=1e-12)
        assert row["runs"] == row["n"]


def test_ecdf_values_are_fractions():
    e = golden("ioh040_min", "ECDF_by_RT")
    for col in e.columns:
        if col in ("mean", "ecdf"):
            assert e[col].min() >= 0 and e[col].max() <= 1


def test_cocopp_art_matches_runtimes():
    rt = golden_cocopp("runtimes_51targets")
    art = golden_cocopp("art_51targets")
    assert art.height == 51
    for row in art.iter_rows(named=True):
        ev = rt.filter(pl.col("target") == row["target"])["evals"]
        succ = ev.drop_nans()
        assert succ.len() == row["n_success"]
        if succ.len() == ev.len():  # all runs succeed: aRT is the mean runtime
            assert math.isclose(row["ert"], succ.mean(), rel_tol=1e-12)


def test_cocopp_ecdf_is_monotone_and_bounded():
    e = golden_cocopp("ecdf_51targets")["ecdf"].to_list()
    assert e == sorted(e) and 0 <= e[0] and e[-1] <= 1
