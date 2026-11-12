"""Target and budget grids (docs/definitions/targets.md), with golden tests against
IOHanalyzer's ``get_ECDF_targets``, ``seq_RT`` and ``get_target_dt`` (WP I03)."""

import numpy as np
import polars as pl
import pytest

from iohinspector.targets import (
    bbob_targets,
    get_ecdf_targets,
    get_target_dt,
    seq_fv,
    seq_rt,
)
from tests.golden.load import golden, logged

FIXTURES = pytest.mark.parametrize("fixture", ["ioh040_min", "ioh040_max"])
PAIR = ["function_id", "dimension"]


def _per_pair(df: pl.DataFrame, column: str) -> dict:
    """``{(function, dimension): sorted values}`` of a golden or our table."""
    df = df.rename({"funcId": "function_id", "DIM": "dimension"}, strict=False)
    df = df.with_columns(pl.col(PAIR).cast(pl.Int64))
    grouped = df.group_by(PAIR).agg(pl.col(column).unique().sort()).rows()
    return {(f, d): np.array(values) for f, d, values in grouped}


def test_seq_fv_log_and_linear():
    np.testing.assert_allclose(seq_fv([1, 1000, 10], 4), [1, 10, 100, 1000])
    assert seq_fv([0, 3, 9], 4, log=False).tolist() == [0, 3, 6, 9]
    assert seq_fv([2, 2], 10).tolist() == [2]  # duplicates removed


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
    expected = _per_pair(golden(fixture, f"targets_{name}"), "target")
    assert ours.keys() == expected.keys()
    for pair, gold in expected.items():
        values = df.filter(function_id=pair[0], dimension=pair[1])["raw_y"]
        if kind == "log-linear" and values.min() <= 0:
            # TG-3: IOHanalyzer starts at 1e-12, iohinspector at the smallest
            # positive logged value
            assert gold[0] == 1e-12
            assert ours[pair][0] == values.filter(values > 0).min()
            assert (len(ours[pair]), ours[pair][-1]) == (len(gold), gold[-1])
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
        (f, d): seq_rt(group["evaluations"], 10, log)
        for (f, d), group in df.group_by(PAIR)
    }
    expected = _per_pair(
        golden(fixture, f"budgets_{'log' if log else 'linear'}"), "budget"
    )
    assert ours.keys() == expected.keys()
    for pair, gold in expected.items():
        assert ours[pair].tolist() == gold.tolist()


@FIXTURES
@pytest.mark.parametrize("perspective", ["fixed_target", "fixed_budget"])
def test_target_dt_matches_iohanalyzer(fixture, perspective):
    maximization = fixture == "ioh040_max"
    ours = get_target_dt(logged(fixture), perspective, maximization)
    which = "by_FV" if perspective == "fixed_target" else "by_RT"
    gold = golden(fixture, f"target_dt_{which}")
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
