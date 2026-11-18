"""KS tests and Holm correction (``docs/definitions/comparisons.md``, CM-1 to CM-4).

The R values come from R 4.3.3 (``rocker/r-ver:4.3.3``): ``ks.test(x, y,
alternative = "greater", exact = ...)`` and ``p.adjust(p, "holm")``.
"""

import math

import numpy as np
import polars as pl
import pytest

from iohinspector.metrics import (
    get_ks_tests,
    get_runtime_sample,
    get_value_sample,
    holm,
    ks_test,
)
from tests.golden.load import golden, logged

INF = math.inf

# x, y, D+, exact p, asymptotic p (R 4.3.3)
R_KS = [
    (
        [10, 12, 15, 30, 40],
        [14, 35, 45, 50, 60],
        0.6,
        0.17857142857142855,
        0.16529888822158645,
    ),
    (
        [1, 2, 3, 1e9, 1e9],
        [2, 3, 1e9, 1e9, 1e9],
        0.2,
        0.71825396825396814,
        0.81873075307798182,
    ),
    (
        [1, 2, 2, 3, 5, 8, 8, 8],
        [2, 3, 3, 4, 6, 7, 9, 9, 10, 11, 12],
        5 / 11,
        0.099005054113413246,
        0.14750759720027606,
    ),
    ([5, 1, 4, 2], [3, 3, 3], 0.5, 0.34285714285714286, 0.42437284567694999),
    ([0.5, 0.7, 0.9], [0.1, 0.2, 0.3, 0.4], 0.0, 1.0, 1.0),
    ([3, 3, 3], [3, 3], 0.0, 1.0, 1.0),
    (
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        1 / 12,
        0.89182489337288084,
        0.92704091651259346,
    ),
]


@pytest.mark.parametrize("x,y,d,p_exact,p_asymp", R_KS)
def test_ks_test_matches_r(x, y, d, p_exact, p_asymp):
    for exact, p in ((True, p_exact), (False, p_asymp)):
        d_plus, p_value, method = ks_test(x, y, exact=exact)
        assert d_plus == pytest.approx(d, abs=1e-15)
        assert p_value == pytest.approx(p, rel=1e-12)
        assert method == ("exact" if exact else "asymptotic")


def test_ks_test_method_rule():
    # CM-1: exact when n m < 10000, R's rule
    assert ks_test(range(99), range(100))[2] == "exact"
    assert ks_test(range(100), range(100))[2] == "asymptotic"


def test_ks_test_failed_runs_are_ties():
    # +inf (failed runs) are the largest values and tie with each other. R's exact
    # p-value treats them as distinct (0.8333, as Inf - Inf is NaN in its tie
    # pattern); here they are ties, as for any other equal values (R with 1e9).
    x, y = [1, 2, 3, INF, INF], [2, 3, INF, INF, INF]
    assert ks_test(x, y, exact=True) == pytest.approx(ks_test(*R_KS[1][:2], exact=True))
    assert ks_test(x, y, exact=False)[1] == pytest.approx(
        0.81873075307798182, rel=1e-12
    )
    assert ks_test([INF, INF], [1, INF]) == (0.0, 1.0, "exact")
    assert ks_test([1, INF], [INF, INF])[0] == 0.5


def test_ks_test_ignores_nan_and_needs_values():
    assert ks_test([1, np.nan, 2], [3]) == ks_test([1, 2], [3])
    with pytest.raises(ValueError):
        ks_test([np.nan], [1])


def test_holm_matches_r():
    p = holm([0.01, np.nan, 0.04, 0.03, 0.2, 0.001])
    r = [0.04, np.nan, 0.09, 0.09, 0.2, 0.005]
    np.testing.assert_allclose(p, r, rtol=1e-15)
    np.testing.assert_allclose(holm([0.5, 0.5, 0.01, 0.02]), [1, 1, 0.04, 0.06])
    assert holm([0.3]).tolist() == [0.3]
    assert holm([]).tolist() == []


def _sample(groups: dict[str, list[float]], value: str = "runtime") -> pl.DataFrame:
    rows = [(g, v) for g, values in groups.items() for v in values]
    return pl.DataFrame(rows, schema=["algorithm_name", value], orient="row")


def test_get_ks_tests_table():
    sample = _sample({"A": [10, 12, 15, 30, 40], "B": [14, 35, 45, 50, 60]})
    out = get_ks_tests(sample, "runtime", alpha=0.4)
    assert out.columns == [
        "algorithm_name",
        "algorithm_name_vs",
        "n",
        "m",
        "d_plus",
        "p_value",
        "method",
        "p_adjusted",
        "better",
        "decision",
    ]
    ab, ba = out.sort("algorithm_name").rows(named=True)
    assert (ab["algorithm_name"], ab["algorithm_name_vs"]) == ("A", "B")
    assert ab["p_value"] == pytest.approx(0.17857142857142855, rel=1e-12)
    assert ab["p_adjusted"] == pytest.approx(
        min(1, 2 * min(ab["p_value"], ba["p_value"]))
    )
    assert ba["d_plus"] == 0 and ba["p_adjusted"] == 1
    assert ab["better"] and not ba["better"]
    assert (ab["decision"], ba["decision"]) == (1, -1)
    assert get_ks_tests(sample, "runtime")["decision"].to_list() == [0, 0]


def test_get_ks_tests_empty_samples():
    # an all-failed group with drop_failures: p = 1 against, 0 for the other
    sample = _sample({"A": [INF, INF], "B": [1, 2], "C": [INF]})
    out = get_ks_tests(sample, "runtime", drop_failures=True, alpha=0.05)
    p = {
        (r[0], r[1]): r[2]
        for r in out.select("algorithm_name", "algorithm_name_vs", "p_value").rows()
    }
    assert p[("A", "B")] == 1 and p[("B", "A")] == 0
    assert p[("A", "C")] is None and p[("C", "A")] is None
    assert out.filter(pl.col("p_value").is_null())["better"].to_list() == [False, False]
    # with failures counted as inf (CM-2) every group has a sample
    assert get_ks_tests(sample, "runtime")["p_value"].null_count() == 0


def test_get_ks_tests_views_are_families_and_higher_is_better():
    sample = pl.concat(
        [
            _sample({"A": [1, 2, 3], "B": [4, 5, 6]}, "value").with_columns(budget=10),
            _sample({"A": [4, 5, 6], "B": [1, 2, 3]}, "value").with_columns(budget=20),
        ]
    )
    out = get_ks_tests(sample, "value", view=["budget"], alpha=0.5)
    assert out.height == 4
    wins = out.filter("better").select("budget", "algorithm_name").rows()
    assert sorted(wins) == [(10, "A"), (20, "B")]
    up = get_ks_tests(
        sample, "value", view=["budget"], alpha=0.5, higher_is_better=True
    )
    assert sorted(up.filter("better").select("budget", "algorithm_name").rows()) == [
        (10, "B"),
        (20, "A"),
    ]


def test_get_ks_tests_by_several_columns():
    sample = _sample({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}).with_columns(
        sigma=pl.Series([1, 1, 2, 2, 1, 1, 2, 2])
    )
    out = get_ks_tests(sample, "runtime", by=["algorithm_name", "sigma"])
    assert out.height == 12
    assert out.columns[:4] == [
        "algorithm_name",
        "sigma",
        "algorithm_name_vs",
        "sigma_vs",
    ]
    assert out.schema["sigma_vs"] == pl.Int64


# ---- golden tests against IOHanalyzer's pairwise.test (asymptotic, failures dropped)

FIXTURES = ["ioh040_min", "ioh040_max", "ioh040_cons"]


def _pairs(fixture):
    return (
        golden(fixture, "KS_by_FV")
        .select("pair_f", "pair_d")
        .unique()
        .sort("pair_f", "pair_d")
        .rows()
    )


def _gold_matrix(fixture, metric, f, d):
    """IOHanalyzer's matrix p[i, j]: "i is better than j"; the rows are the
    algorithms in the order of the columns."""
    ref = golden(fixture, metric).filter(
        (pl.col("pair_f") == f) & (pl.col("pair_d") == d)
    )
    names = ref.columns[2:]
    return names, ref.select(names).to_numpy()


def _ours_matrix(out, names):
    p = {
        (a, c): v
        for a, c, v in out.select(
            "algorithm_name", "algorithm_name_vs", "p_adjusted"
        ).rows()
    }
    return np.array(
        [[p.get((a, c), np.nan) for c in names] for a in names], dtype=float
    )


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("which", ["by_FV", "by_RT"])
def test_ks_tests_match_iohanalyzer(fixture, which):
    data = logged(fixture)
    maximization = fixture == "ioh040_max"
    for f, d in _pairs(fixture):
        pair = data.filter((pl.col("function_id") == f) & (pl.col("dimension") == d))
        names, gold = _gold_matrix(fixture, f"KS_{which}", f, d)
        if which == "by_FV":  # the third of IOHanalyzer's five targets
            ref = golden(fixture, "RT_sample").filter(
                (pl.col("pair_f") == f) & (pl.col("pair_d") == d)
            )
            target = sorted(ref["target"].unique())[2]
            sample = get_runtime_sample(pair, [target], maximization=maximization)
            out = get_ks_tests(sample, "runtime", exact=False, drop_failures=True)
        else:  # the third of its five budgets
            ref = golden(fixture, "FV_sample").filter(
                (pl.col("pair_f") == f) & (pl.col("pair_d") == d)
            )
            budget = int(sorted(ref["runtime"].unique())[2])
            sample = get_value_sample(pair, [budget], maximization=maximization)
            out = get_ks_tests(
                sample, "value", exact=False, higher_is_better=maximization
            )
            if maximization:  # CM-3: IOHanalyzer tests the reverse direction
                gold = gold.T
        np.testing.assert_allclose(
            _ours_matrix(out, names), gold, rtol=1e-12, err_msg=f"{f} {d}"
        )
