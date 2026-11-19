import itertools
import math

import numpy as np
import polars as pl
import pytest

from iohinspector.metrics import (
    envelope_value,
    get_portfolio_ecdf,
    get_shapley_values,
    shapley_values,
)
from iohinspector.metrics.distributions import get_runtime_sample
from iohinspector.metrics.portfolio import _added
from iohinspector.targets import seq_rt
from tests.golden.load import golden, logged

TOL = 1e-12  # portfolio.md, "Test tolerance"


def _table_game(table):
    """v from a table indexed by the coalition's bits."""

    def value(coalitions):
        return table[coalitions @ (1 << np.arange(coalitions.shape[1]))]

    return value


def _by_orders(n, marginal):
    """The mean of marginal(S, a) over all n! orders, with S the bit code of the
    players before a."""
    phi = np.zeros(n)
    for order in itertools.permutations(range(n)):
        code = 0
        for a in order:
            phi[a] += marginal(code, a)
            code |= 1 << a
    return phi / math.factorial(n)


def _by_permutations(table, n):
    """Shapley values as the mean marginal contribution over all n! orders."""
    return _by_orders(n, lambda s, a: table[s | 1 << a] - table[s]) / table[-1]


def _plain(value):
    """``value`` as a custom value function: enumerated or sampled, never the
    closed form."""
    return lambda coalitions: value(coalitions)


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_exact_is_the_mean_over_all_orders(n):
    rng = np.random.default_rng(n)
    table = np.r_[0, rng.random((1 << n) - 1)]  # any game, not only a portfolio
    phi, se = shapley_values(n, _table_game(table))
    np.testing.assert_allclose(phi, _by_permutations(table, n), rtol=TOL)
    assert abs(phi.sum() - 1) < TOL and not se.any()


@pytest.mark.parametrize("n", range(1, 16))
def test_closed_form_equals_enumeration(n):
    # Q-SH-2: the default value in closed form, against enumeration up to the limit
    rng = np.random.default_rng(n)
    ecdf, width = rng.random((n, 30)).round(1), rng.random(30)  # with ties
    ecdf[:, :3] = 0  # steps no player reaches
    value = envelope_value(ecdf, width)
    phi, se = shapley_values(n, value)
    enumerated, _ = shapley_values(n, _plain(value))
    np.testing.assert_allclose(phi, enumerated, rtol=TOL)
    assert abs(phi.sum() - 1) < TOL and not se.any()


def test_closed_form_for_any_number_of_players():
    n = 40
    rng = np.random.default_rng(0)
    ecdf, width = rng.random((n, 200)) ** 3, rng.random(200)
    ecdf[1] = ecdf[0]  # equal players, equal values
    value = envelope_value(ecdf, width)
    phi, se = shapley_values(n, value, exact_limit=0, samples=2, seed=0)
    assert not se.any() and phi[0] == pytest.approx(phi[1], rel=TOL)
    assert abs(phi.sum() - 1) < TOL
    sampled, se = shapley_values(n, _plain(value), samples=10, seed=1)
    assert (np.abs(sampled - phi) < 4 * se).all()  # 40 players: 4, not 3
    with pytest.raises(ValueError, match="n_players"):
        shapley_values(n - 1, value)


def test_sampled_values_are_within_three_standard_errors():
    rng = np.random.default_rng(6)
    value = _plain(envelope_value(rng.random((6, 50)), rng.random(50)))
    exact, _ = shapley_values(6, value)
    sampled, se = shapley_values(6, value, exact_limit=5, samples=20, seed=1)
    assert (se > 0).all()
    assert (np.abs(sampled - exact) < 3 * se).all()
    again, _ = shapley_values(6, value, exact_limit=5, samples=20, seed=1)
    np.testing.assert_array_equal(sampled, again)  # seeded


def test_sampling_covers_every_position():
    # SH-2: unlike IOHanalyzer's truncated permutations, the coalitions of every
    # size are sampled; in the unanimity game only the last position counts
    n = 16

    def value(coalitions):
        return coalitions.all(axis=1).astype(float)

    phi, se = shapley_values(n, value, samples=3, seed=0)
    np.testing.assert_allclose(phi, 1 / n, rtol=TOL)
    assert not se.any()


def test_values_are_cached():
    calls = []

    def value(coalitions):
        calls.append(len(coalitions))
        return coalitions.sum(axis=1).astype(float)

    phi, _ = shapley_values(4, value)  # exact: one call on all 16 coalitions
    assert calls == [16]
    np.testing.assert_allclose(phi, 0.25, rtol=TOL)
    calls.clear()
    shapley_values(18, value, samples=4, seed=0)
    assert sum(calls) <= 2**18  # every coalition at most once (here far fewer)
    assert sum(calls) < 2 * 18 * 18 * 4


def test_custom_marginal_contribution():
    # SH-4: the marginal is a parameter; it gets the (cached) value function
    rng = np.random.default_rng(3)
    table = np.r_[0, rng.random(7)]
    seen = []

    def marginal(value, coalitions, a):
        seen.append(a)
        with_a = coalitions.copy()
        with_a[:, a] = True
        return np.maximum(value(with_a) - value(coalitions), 0)

    phi, _ = shapley_values(3, _table_game(table), marginal=marginal)
    expected = _by_orders(3, lambda s, a: max(table[s | 1 << a] - table[s], 0))
    np.testing.assert_allclose(phi, expected / table[-1], rtol=TOL)
    assert seen == [0, 1, 2]


def test_zero_portfolio_value_is_nan():
    phi, se = shapley_values(3, lambda s: np.zeros(len(s)))
    assert np.isnan(phi).all() and np.isnan(se).all()


# The ECDF-envelope value (SH-1) on small data, against a direct computation.


def _runs(algorithm, curves, function_id=1, dimension=2):
    """One run per list of (evaluations, value) rows."""
    rows = [
        (f"{algorithm}{i}", algorithm, function_id, dimension, e, y)
        for i, curve in enumerate(curves)
        for e, y in curve
    ]
    names = ["run_id", "algorithm_name", "function_id", "dimension"]
    return pl.DataFrame(
        rows, schema=[*names, "evaluations", "raw_y"], orient="row"
    ).with_columns(pl.col("function_id", "dimension").cast(pl.Int32))


DATA = pl.concat(
    [
        _runs("A", [[(1, 9.0), (4, 3.0), (10, 1.0)], [(1, 5.0), (8, 2.0)]]),
        _runs("B", [[(1, 8.0), (2, 2.0)], [(1, 6.0), (3, 4.0)], [(1, 9.0)]]),
        _runs("C", [[(1, 7.0), (6, 0.5)]]),
        _runs("A", [[(1, 4.0), (5, 1.0)]], function_id=2),
        _runs("B", [[(1, 2.0)]], function_id=2),
        _runs("C", [[(1, 3.0), (9, 2.0)]], function_id=2),
    ]
)
TARGETS = pl.DataFrame(
    {
        "function_id": [1, 1, 1, 2, 2],
        "dimension": [2] * 5,
        "target": [4.0, 2.0, 1.0, 2.0, 1.0],
    }
).cast({"function_id": pl.Int32, "dimension": pl.Int32})


def _direct_value(data, targets, coalition, lo, hi, log=False, maximization=False):
    """v(S) by integrating max_{a ∈ S} ECDF_a over [lo, hi] on a fine grid of the
    step function's own breakpoints, pair by pair and target by target."""
    axis = math.log10 if log else (lambda b: b)
    total = 0.0
    for f, d, t in targets.rows():
        runs = data.filter(
            (pl.col("function_id") == f) & (pl.col("dimension") == d)
        ).partition_by("run_id")
        hit = {}
        for run in runs:
            reached = run.filter(
                (pl.col("raw_y") >= t) if maximization else (pl.col("raw_y") <= t)
            )
            first = reached["evaluations"].min() if reached.height else math.inf
            hit.setdefault(run["algorithm_name"][0], []).append(first)
        points = sorted(
            {lo, hi, *(min(max(b, lo), hi) for h in hit.values() for b in h)}
        )
        for left, right in itertools.pairwise(points):
            best = max(
                (np.mean(np.array(hit[a]) <= left) for a in coalition if a in hit),
                default=0.0,
            )
            total += best * (axis(right) - axis(left)) / (axis(hi) - axis(lo))
    return total


def _direct_shapley(data, targets, lo, hi, **kw):
    players = sorted(data["algorithm_name"].unique())
    n = len(players)
    table = np.array(
        [
            _direct_value(
                data,
                targets,
                [p for i, p in enumerate(players) if k >> i & 1],
                lo,
                hi,
                **kw,
            )
            for k in range(1 << n)
        ]
    )
    return _by_permutations(table, n)


@pytest.mark.parametrize(
    "lo,hi,log", [(1, 10, False), (1, 10, True), (2, 6, False), (1, 30, True)]
)
def test_portfolio_value_is_the_area_of_the_virtual_best_ecdf(lo, hi, log):
    result = get_shapley_values(DATA, TARGETS, b_lo=lo, b_hi=hi, scale_eval_log=log)
    assert result.columns == ["algorithm_name", "shapley", "se"]
    assert result["algorithm_name"].to_list() == ["A", "B", "C"]
    expected = _direct_shapley(DATA, TARGETS, lo, hi, log=log)
    np.testing.assert_allclose(result["shapley"], expected, rtol=TOL)
    assert abs(result["shapley"].sum() - 1) < TOL


def test_portfolio_value_under_maximization():
    data = DATA.with_columns(-pl.col("raw_y"))
    targets = TARGETS.with_columns(-pl.col("target"))
    result = get_shapley_values(data, targets, maximization=True)
    expected = _direct_shapley(data, targets, 1, 10, maximization=True)
    np.testing.assert_allclose(result["shapley"], expected, rtol=TOL)


@pytest.mark.filterwarnings("ignore:the groups of by")
def test_portfolio_by_other_columns_and_sampling():
    data = DATA.with_columns(seed=pl.col("run_id").str.slice(1).cast(pl.Int64) % 2)
    by = ["algorithm_name", "seed"]
    exact = get_shapley_values(data, TARGETS, by=by, pairs="all")
    assert exact.columns == ["algorithm_name", "seed", "shapley", "se"]
    assert exact.height == 5  # (C, 1) has no runs
    assert (exact["se"] == 0).all()
    sampled = get_shapley_values(  # the default marginal, given: sampled
        data,
        TARGETS,
        by=by,
        pairs="all",
        marginal=_added,
        exact_limit=2,
        samples=50,
        seed=4,
    )
    assert (sampled["se"] > 0).all()
    assert (np.abs(sampled["shapley"] - exact["shapley"]) <= 3 * sampled["se"]).all()


def test_portfolio_default_targets_and_errors():
    result = get_shapley_values(DATA)  # 10 log-linear targets per pair
    assert abs(result["shapley"].sum() - 1) < TOL
    nothing = get_shapley_values(DATA, TARGETS.with_columns(target=pl.lit(-5.0)))
    assert nothing["shapley"].is_nan().all()  # no target reached: v(N) = 0
    with pytest.raises(ValueError, match="by"):
        get_shapley_values(DATA, TARGETS, by=[])
    with pytest.raises(ValueError, match="budget range"):
        get_shapley_values(DATA, TARGETS, b_lo=10, b_hi=5)
    with pytest.raises(TypeError, match="unknown"):
        get_shapley_values(DATA, TARGETS, scale_log=True)


# SH-4, review of #44: another portfolio value and another marginal contribution on
# the ECDF data of get_shapley_values, through the public functions only.


def test_portfolio_ecdf_is_the_game_of_get_shapley_values():
    players, ecdf, width = get_portfolio_ecdf(DATA, TARGETS)
    assert players["algorithm_name"].to_list() == ["A", "B", "C"]
    assert ecdf.shape == (3, len(width))
    assert abs(width.sum() - TARGETS.height) < TOL  # 1 per (pair, target)
    phi, _ = shapley_values(3, envelope_value(ecdf, width))
    expected = get_shapley_values(DATA, TARGETS)["shapley"]
    np.testing.assert_allclose(phi, expected, rtol=TOL)


def _members(ecdf, s):
    return [ecdf[b] for b in range(len(ecdf)) if s >> b & 1]


def test_another_portfolio_value():
    # the parallel portfolio, every member run once and independently: per step the
    # probability that at least one member has reached the target
    _, ecdf, width = get_portfolio_ecdf(DATA, TARGETS, scale_eval_log=True)

    def parallel(coalitions):
        return (1 - np.prod(1 - coalitions[:, :, None] * ecdf, axis=1)) @ width

    def direct(s):
        miss = np.prod([1 + 0 * width, *(1 - e for e in _members(ecdf, s))], axis=0)
        return (1 - miss) @ width

    table = np.array([direct(s) for s in range(8)])
    phi, _ = shapley_values(3, parallel)
    np.testing.assert_allclose(phi, _by_permutations(table, 3), rtol=TOL)
    assert abs(phi.sum() - 1) < TOL
    default = get_shapley_values(DATA, TARGETS, scale_eval_log=True)["shapley"]
    assert np.abs(phi - default.to_numpy()).max() > 1e-3  # another game


def test_another_marginal_contribution():
    # not a difference of portfolio values: the share of the budget axis on which a
    # is strictly ahead of every member of S; v (the default) then only normalises
    _, ecdf, width = get_portfolio_ecdf(DATA, TARGETS)

    def ahead(_value, coalitions, a):
        best = (coalitions[:, :, None] * ecdf).max(axis=1)
        return (ecdf[a] > best) @ width

    def direct(s, a):
        best = np.max([0 * ecdf[a], *_members(ecdf, s)], axis=0)
        return (ecdf[a] > best) @ width

    value = envelope_value(ecdf, width)
    expected = _by_orders(3, direct) / (ecdf.max(axis=0) @ width)
    phi, _ = shapley_values(3, value, marginal=ahead)
    np.testing.assert_allclose(phi, expected, rtol=TOL)
    sampled, se = shapley_values(
        3, value, marginal=ahead, exact_limit=0, samples=50, seed=2
    )
    assert (np.abs(sampled - expected) <= 3 * se + TOL).all()


# Golden tests against IOHanalyzer (WP I03): get_shapley_values and
# get_marg_contrib_ecdf with 5 log-linear targets. IOHanalyzer's portfolio value is
# a different game (portfolio.md, "Differences"): R's ecdf() of the successful
# runtimes only, summed over a 50-point grid instead of integrated. Played through
# shapley_values as a custom value function (SH-4), it reproduces IOHanalyzer.

GOLDEN = ["ioh040_min", "ioh040_max", "ioh040_cons"]


def _iohanalyzer_value(fixture, scale_log):
    """``generate_data.ECDF_raw`` as a matrix (player × (pair, target, grid point))
    and v(S) = Σ max_{a ∈ S} hit, as ``get_marg_contrib_ecdf``."""
    data = logged(fixture)
    maximization = fixture == "ioh040_max"
    grid = seq_rt(data["evaluations"], 50, log=scale_log)
    raw = golden(fixture, "ECDF_raw")
    players = sorted(raw["ID"].unique())
    rows = {p: [] for p in players}
    for (f, d), targets in raw.partition_by(["funcId", "DIM"], as_dict=True).items():
        pair = data.filter((pl.col("function_id") == f) & (pl.col("dimension") == d))
        sample = get_runtime_sample(
            pair, targets["target"].unique().sort(), maximization=maximization
        )
        for p in players:
            for _, rt in sample.filter(algorithm_name=p).group_by(
                "target", maintain_order=True
            ):
                ok = rt["runtime"].filter(rt["runtime"].is_finite()).to_numpy()
                hit = (ok[:, None] <= grid).mean(axis=0) if len(ok) else 0 * grid
                rows[p].append(hit)
    ecdf = np.array([np.concatenate(rows[p]) for p in players])
    return players, ecdf, grid


@pytest.mark.parametrize("fixture", GOLDEN)
def test_iohanalyzer_ecdf_raw_is_reproduced(fixture):
    players, ecdf, grid = _iohanalyzer_value(fixture, scale_log=False)
    raw = golden(fixture, "ECDF_raw")
    for i, p in enumerate(players):
        ref = raw.filter(ID=p).sort("funcId", "DIM", "target", "rt")
        np.testing.assert_allclose(np.sort(ecdf[i]), np.sort(ref["hit"]), rtol=TOL)
        assert len(grid) == ref["rt"].n_unique()


@pytest.mark.parametrize("fixture", GOLDEN)
def test_iohanalyzer_marginal_contributions(fixture):
    players, ecdf, _ = _iohanalyzer_value(fixture, scale_log=True)
    value = envelope_value(ecdf, np.ones(ecdf.shape[1]))
    n = len(players)
    prefixes = np.tril(np.ones((n + 1, n), dtype=bool), -1)  # ∅, {1}, {1, 2}, ...
    contrib = np.diff(value(prefixes))
    gold = golden(fixture, "marg_contrib_ecdf").sort("j")
    assert gold["ID"].to_list() == players
    np.testing.assert_allclose(contrib, gold["contrib"], rtol=TOL)


@pytest.mark.parametrize("scale", ["log", "lin"])
@pytest.mark.parametrize("fixture", GOLDEN)
def test_iohanalyzer_shapley_estimate(fixture, scale):
    """IOHanalyzer draws its permutations without a seed we can reproduce. With
    three players its estimate is (m(∅) + mean of 5 draws of m({b}) + m(N∖a)) / 3,
    so it lies between the values with the smallest and largest m({b}); with two
    players it is exact."""
    players, ecdf, _ = _iohanalyzer_value(fixture, scale_log=scale == "log")
    n = len(players)
    value = envelope_value(ecdf, np.ones(ecdf.shape[1]))
    exact, _ = shapley_values(n, value)
    gold = golden(fixture, f"shapley_{scale}").sort("ID")
    assert gold["ID"].to_list() == players
    total = value(np.ones((1, n), dtype=bool))[0]
    if n == 2:
        np.testing.assert_allclose(gold["shapley"], exact, rtol=TOL)
        return
    for a in range(n):
        alone = np.eye(n, dtype=bool)[[a]]
        rest = ~alone
        pairs = [
            np.eye(n, dtype=bool)[[a]] | np.eye(n, dtype=bool)[[b]]
            for b in range(n)
            if b != a
        ]
        middle = [value(p)[0] - value(p & ~alone)[0] for p in pairs]
        ends = value(alone)[0] + total - value(rest)[0]
        lo, hi = (ends + min(middle)) / 3 / total, (ends + max(middle)) / 3 / total
        assert lo - TOL <= gold["shapley"][a] <= hi + TOL


@pytest.mark.parametrize("fixture", ["ioh040_min", "ioh040_max"])
def test_agreed_value_differs_from_iohanalyzer(fixture):
    # SH-1: ECDFs over all runs (EC-1), integrated exactly (EC-4); IOHanalyzer's
    # values differ, and the test names the difference rather than copying it
    raw = golden(fixture, "ECDF_raw")
    targets = raw.select(
        pl.col("funcId").cast(pl.Int32).alias("function_id"),
        pl.col("DIM").cast(pl.Int32).alias("dimension"),
        "target",
    ).unique()
    ours = get_shapley_values(
        logged(fixture),
        targets,
        scale_eval_log=True,
        maximization=fixture == "ioh040_max",
    )
    gold = golden(fixture, "shapley_log").sort("ID")
    assert abs(ours["shapley"].sum() - 1) < TOL
    assert np.abs(ours["shapley"].to_numpy() - gold["shapley"].to_numpy()).max() > 0.01
