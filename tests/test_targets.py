"""Target and budget grids (docs/definitions/targets.md), with golden tests against
IOHanalyzer's ``get_ECDF_targets``, ``seq_RT`` and ``get_target_dt`` (WP I03)."""

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iohinspector import DataManager
from iohinspector.metrics import get_ecdf, get_ecdf_auc
from iohinspector.targets import (
    SUITE_BOUNDS,
    bbob_targets,
    get_ecdf_targets,
    get_target_dt,
    seq_fv,
    seq_rt,
)
from tests.golden.load import fixture_path, golden_renamed, logged

FIXTURES = pytest.mark.parametrize("fixture", ["ioh040_min", "ioh040_max"])
PAIR = ["function_id", "dimension"]


def _per_pair(df: pl.DataFrame, column: str) -> dict:
    """``{(function, dimension): sorted values}`` of a golden or our table."""
    df = df.with_columns(pl.col(PAIR).cast(pl.Int64))
    grouped = df.group_by(PAIR).agg(pl.col(column).unique().sort()).rows()
    return {(f, d): np.array(values) for f, d, values in grouped}


def test_seq_fv_log_and_linear():
    np.testing.assert_allclose(seq_fv([1, 1000, 10], 4), [1, 10, 100, 1000])
    assert seq_fv([0, 3, 9], 4, log=False).tolist() == [0, 3, 6, 9]
    assert seq_fv([2, 2], 10).tolist() == [2]  # duplicates removed


def test_log_targets_end_exactly_at_the_observed_values():
    # 10 ** log10(18) is 17.999999999999996: a run at 18 must reach the top target
    assert seq_fv([2, 18], 10)[[0, -1]].tolist() == [2.0, 18.0]
    assert seq_fv([2, 18], 1).tolist() == [2.0]


def test_seq_fv_log_starts_at_smallest_positive_value():
    # TG-3: f = 0 is replaced by the smallest positive value, a user lower wins
    np.testing.assert_allclose(seq_fv([0, 0.01, 100], 5), [0.01, 0.1, 1, 10, 100])
    np.testing.assert_allclose(seq_fv([0, 0.01, 100], 3, lower=1), [1, 10, 100])
    with pytest.warns(UserWarning, match="1e-12"):
        assert seq_fv([0, -1], 3).tolist() == [1e-12]
    with pytest.raises(ValueError, match="positive"):
        seq_fv([1, 10], lower=0)


def test_seq_rt_rounds_and_keeps_the_largest_budget():
    # TG-5: 10 ** 2 computed in log space is not truncated to 99
    assert seq_rt([1, 100], 3).tolist() == [1, 10, 100]
    assert seq_rt([1, 10], 50)[-1] == 10
    assert seq_rt([1, 3], 3, log=False).tolist() == [1, 2, 3]
    assert seq_rt([1, 2], 4, log=False).tolist() == [1, 2]  # 1.33 1.67 round


def test_bbob_targets_hit_the_decades_exactly():
    targets = bbob_targets()
    assert len(targets) == 51
    assert targets[::5].tolist() == [10.0**k for k in range(2, -9, -1)]


@FIXTURES
@pytest.mark.parametrize("kind", ["log-linear", "linear", "bbob"])
def test_ecdf_targets_match_iohanalyzer(fixture, kind):
    df = logged(fixture)
    ours = _per_pair(get_ecdf_targets(df, kind), "target")
    name = {"log-linear": "log_linear"}.get(kind, kind)
    expected = _per_pair(golden_renamed(fixture, f"targets_{name}"), "target")
    assert ours.keys() == expected.keys()
    for pair, gold in expected.items():
        values = df.filter(function_id=pair[0], dimension=pair[1])["raw_y"]
        if kind == "log-linear" and values.min() <= 0:
            # TG-3: IOHanalyzer starts at 1e-12, iohinspector at the smallest
            # positive logged value
            assert gold[0] == 1e-12
            assert ours[pair][0] == values.filter(values > 0).min()
            assert len(ours[pair]) == len(gold)
            # the top target is the largest logged value; R's 10^x is 1 ulp off
            assert ours[pair][-1] == values.max()
            np.testing.assert_allclose(ours[pair][-1], gold[-1], rtol=1e-15, atol=0)
            continue
        # 10^x and R's text round trip of the logged values differ in the last
        # bit; bbob is built from exact decades (targets.md)
        rtol = 4e-15 if kind == "bbob" else 1e-15
        np.testing.assert_allclose(ours[pair], gold, rtol=rtol, atol=0)


@FIXTURES
@pytest.mark.parametrize("log", [True, False])
def test_budgets_match_iohanalyzer(fixture, log):
    df = logged(fixture)
    ours = {
        (f, d): seq_rt(group["evaluations"], 10, log=log)
        for (f, d), group in df.group_by(PAIR)
    }
    expected = _per_pair(
        golden_renamed(fixture, f"budgets_{'log' if log else 'linear'}"), "budget"
    )
    assert ours.keys() == expected.keys()
    for pair, gold in expected.items():
        assert ours[pair].tolist() == gold.tolist()


@FIXTURES
@pytest.mark.parametrize("perspective", ["fixed_target", "fixed_budget"])
def test_target_dt_matches_iohanalyzer(fixture, perspective):
    maximization = fixture == "ioh040_max"
    ours = get_target_dt(logged(fixture), perspective, maximization=maximization)
    which = "by_FV" if perspective == "fixed_target" else "by_RT"
    gold = golden_renamed(fixture, f"target_dt_{which}")
    ours, gold = (
        {pair: values.item() for pair, values in _per_pair(t, "target").items()}
        for t in (ours, gold)
    )
    if maximization and perspective == "fixed_target":
        # TG-4: IOHanalyzer takes the max of the 2 % quantiles; iohinspector the max
        # of the 98 % quantiles, so its default target is never easier
        assert (ours.pop((2, 32)), gold.pop((2, 32))) == pytest.approx((32, 25.48))
    assert ours == pytest.approx(gold, rel=1e-15)


def test_target_dt_needs_a_perspective():
    with pytest.raises(ValueError, match="fixed_target or fixed_budget"):
        get_target_dt(logged("ioh040_min"), "by_RT")


# Default bounds by suite (TG-8)

#: The ``target_bounds`` of the IOHexperimenter suite registry of ioh 0.4.2: the
#: release asset ``registry.json`` (its URL and sha256 are in the file) reduced to
#: ``target_bounds`` per suite.
REGISTRY = Path(__file__).parent / "ioh_registry_0.4.2.json"


def test_suite_bounds_match_the_registry():
    registry = json.loads(REGISTRY.read_text())
    assert registry["ioh_version"] == "0.4.2"
    both = {
        name: entry["target_bounds"]
        for name, entry in registry["suites"].items()
        if None
        not in (entry["target_bounds"]["lower"], entry["target_bounds"]["upper"])
    }
    assert SUITE_BOUNDS == {
        name.upper(): (bounds["lower"], bounds["upper"])
        for name, bounds in both.items()
    }
    # the table assumes log-scale precisions, which raw_y holds for these suites
    for bounds in both.values():
        assert bounds["scale"] == "log" and bounds["relative_to_optimum"]


def _suite_runs(suites, values=(5.0, 0.5)):
    """One run per suite on pair (1, 2), logging ``values``."""
    return pl.DataFrame(
        {
            "run_id": [i for i in range(len(suites)) for _ in values],
            "function_id": 1,
            "dimension": 2,
            "suite": [s for s in suites for _ in values],
            "evaluations": [e for _ in suites for e in range(1, len(values) + 1)],
            "raw_y": [v for _ in suites for v in values],
        }
    )


@pytest.mark.parametrize("suite", ["BBOB", "bbob", "SBOX"])
def test_registry_suites_get_their_bounds(suite):
    targets = get_ecdf_targets(_suite_runs([suite, suite]))["target"]
    np.testing.assert_allclose(targets, 10.0 ** np.linspace(-8, 2, 10), rtol=1e-15)
    assert targets[[0, -1]].to_list() == [1e-8, 1e2]
    linear = get_ecdf_targets(_suite_runs([suite]), "linear", n=2)["target"]
    assert linear.to_list() == [1e-8, 1e2]


@pytest.mark.parametrize(
    "suites", [["PBO"], ["unknown_suite"], ["BBOB", "unknown_suite"], [None]]
)
def test_other_suites_keep_the_observed_range(suites):
    targets = get_ecdf_targets(_suite_runs(suites), n=2)["target"]
    assert targets.to_list() == [0.5, 5.0]
    no_column = _suite_runs(suites).drop("suite")
    assert get_ecdf_targets(no_column, n=2)["target"].to_list() == [0.5, 5.0]


def test_the_caller_wins():
    bbob = _suite_runs(["BBOB"])
    assert get_ecdf_targets(bbob, n=2, lower=1e-3)["target"].to_list() == [1e-3, 1e2]
    assert get_ecdf_targets(bbob, n=2, upper=10)["target"].to_list() == [1e-8, 10]
    assert get_ecdf_targets(bbob, "bbob")["target"].equals(
        pl.Series("target", bbob_targets())
    )
    targets = pl.DataFrame({"function_id": [1], "dimension": [2], "target": [1.0]})
    assert get_ecdf(bbob, targets, by=())["ecdf"].to_list() == [0.0, 1.0]


def test_continuum_ecdf_and_aocc_follow_the_suite_bounds():
    # between 1e2 and 1e-8 (10 decades) q(0.5) = (2 - log10 0.5) / 10, q(5) one tenth less
    q = (2 - np.log10(0.5)) / 10
    bbob = _suite_runs(["BBOB"])
    ecdf = get_ecdf(bbob, by=(), continuous=True)
    np.testing.assert_allclose(ecdf["ecdf"], [0, q - 0.1, q], rtol=1e-12)
    aocc = get_ecdf_auc(bbob, by=(), continuous=True)["auc"].item()
    assert aocc == pytest.approx(q - 0.1, rel=1e-12)  # over [1, 2]: q(5)


def test_coco_bbob_data_gets_the_bbob_bounds():
    manager = DataManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manager.add_folder(fixture_path("coco"))
    data = manager.load(include_columns=["function_id", "dimension", "suite"])
    assert set(data["suite"]) == {"bbob"}
    targets = get_ecdf_targets(data)["target"]
    assert targets[[0, -1]].to_list() == [1e-8, 1e2]
