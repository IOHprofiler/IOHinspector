"""Glicko-2 and Elo tournament ratings (``docs/definitions/glicko2.md``, GL-1 to GL-4).

The golden tests feed IOHanalyzer's recorded game lists (``glicko2_games_by_*``,
the data frame it passed to ``PlayerRatings::glicko2``) into ``glicko2`` (GL-3).
"""

import math

import numpy as np
import polars as pl
import pytest

from iohinspector.metrics import (
    elo,
    get_tournament_ranking,
    get_tournament_ratings,
    glicko2,
)
from iohinspector.metrics.ranking import fmin
from tests.golden.load import golden, logged


def _games(rows):
    return pl.DataFrame(
        rows, schema=["round", "player", "opponent", "score"], orient="row"
    )


@pytest.mark.parametrize(
    "f,a,b,tol,r",
    [  # R 4.3.3: optimize(f, c(a, b), tol = tol)$minimum
        (lambda x: (x - 1 / 3) ** 2, 0, 1, None, 0.33333333333333331),
        (math.sin, 0, 2 * math.pi, None, 4.7123910812000886),
        (lambda x: abs(x - 2) + 0.1 * x**2, -5, 5, None, 1.9999798813102465),
        (lambda x: math.inf if x < 0.5 else (x - 1) ** 2, 0, 3, None, 1.0),
        (lambda x: (x - 1 / 3) ** 2, 0, 1, 1e-10, 0.33333333333333331),
    ],
)
def test_fmin_matches_r_optimize(f, a, b, tol, r):
    assert fmin(f, a, b, *([tol] if tol else [])) == r


def test_glicko2_glickman_example():
    # glicko2.md: Glickman's worked example, PlayerRatings 1.1-0 with tau = 0.5
    games = _games([(1, 0, 1, 1.0), (1, 0, 2, 0.0), (1, 0, 3, 0.0)])
    init = ([1500, 1400, 1550, 1700], [200, 30, 100, 300], 0.06)
    row = glicko2(games, 4, tau=0.5, init=init).row(0, named=True)
    assert row["rating"] == pytest.approx(1464.050671, rel=1e-9)
    assert row["deviation"] == pytest.approx(151.5165213, rel=1e-9)
    assert row["volatility"] == pytest.approx(0.0599958339, rel=1e-9)
    assert (row["games"], row["wins"], row["draws"], row["losses"]) == (3, 1, 0, 2)


def test_glicko2_deviation_grows_while_sitting_out():
    games = _games([(1, 0, 1, 1.0), (2, 0, 2, 0.5), (3, 0, 1, 0.5)])
    out = glicko2(games, 3)
    played = glicko2(_games([(1, 0, 1, 1.0), (2, 0, 1, 0.5), (3, 0, 1, 0.5)]), 2)
    # player 1 sat out round 2: its deviation grew by its volatility first
    assert out["deviation"][1] > played["deviation"][1]
    assert out["games"].to_list() == [3, 2, 1]


@pytest.mark.parametrize("fixture", ["ioh040_min", "ioh040_max", "ioh040_cons"])
@pytest.mark.parametrize("which", ["by_FV", "by_RT"])
def test_glicko2_matches_playerratings(fixture, which):
    # bit-identical but for ioh040_min by_RT on macOS (4e-12: exp and log of
    # different C libraries)
    games = golden(fixture, f"glicko2_games_{which}").select(
        pl.col("Weeks").alias("round"),
        (pl.col("Player1") - 1).alias("player"),
        (pl.col("Player2") - 1).alias("opponent"),
        pl.col("Scores").alias("score"),
    )
    gold = golden(fixture, f"glicko2_{which}")
    names = sorted(gold["Player"])  # the players are indices into the sorted names
    ours = glicko2(games, len(names), tau=1.2).with_columns(Player=pl.Series(names))
    joined = gold.join(ours, on="Player")
    assert joined.height == len(names)
    for theirs, mine in [
        ("Rating", "rating"),
        ("Deviation", "deviation"),
        ("Volatility", "volatility"),
    ]:
        np.testing.assert_allclose(
            joined[mine], joined[theirs], rtol=1e-11, err_msg=theirs
        )
    for theirs, mine in [
        ("Games", "games"),
        ("Win", "wins"),
        ("Draw", "draws"),
        ("Loss", "losses"),
    ]:
        assert joined[mine].to_list() == joined[theirs].cast(pl.Int64).to_list()


def test_elo_update():
    out = elo(_games([(1, 0, 1, 1.0), (1, 1, 0, 0.5)]), 2)
    first = 1500 + 20 * 0.5
    e = 1 / (1 + 10 ** ((first - (3000 - first)) / 400))
    second = (3000 - first) + 20 * (0.5 - e)
    assert out["rating"].to_list() == pytest.approx([3000 - second, second], rel=1e-15)
    assert out["volatility"].is_null().all()
    assert out["deviation"][0] == pytest.approx(np.std([first, 3000 - second], ddof=1))


# ---- the tournament


def _runs(values: dict[str, list[float]], fid: int = 1) -> pl.DataFrame:
    """One logged row per run: evaluation 1 with each value; NaN never logs a
    value that reaches any target below it."""
    rows = [
        (a, f"{a}{fid}{i}", v) for a, vs in values.items() for i, v in enumerate(vs)
    ]
    df = pl.DataFrame(rows, schema=["algorithm_name", "run_id", "raw_y"], orient="row")
    return df.with_columns(function_id=fid, dimension=2, evaluations=1)


TARGET = pl.DataFrame({"function_id": [1], "dimension": [2], "target": [1.0]})


def test_tournament_outcomes():
    # A always reaches the target, B never, C never: A beats both, B and C draw
    data = _runs({"A": [0.5, 0.0], "B": [5.0, 6.0], "C": [7.0]})
    out = get_tournament_ranking(data, TARGET, seed=3, rounds=10)
    rows = {r["algorithm_name"]: r for r in out.rows(named=True)}
    assert out["algorithm_name"][0] == "A"
    assert (rows["A"]["wins"], rows["A"]["games"]) == (20, 20)
    assert (rows["B"]["draws"], rows["B"]["losses"]) == (10, 10)
    # fixed budget, maximising: the larger value wins
    up = get_tournament_ranking(
        data, TARGET, seed=3, rounds=10, perspective="fixed_budget", maximization=True
    )
    assert up["algorithm_name"][0] == "C"


def test_tournament_is_seeded():
    data = pl.concat(
        [_runs({"A": [0.5, 3.0, 0.1], "B": [0.9, 0.2]}, fid) for fid in (1, 2)]
    )
    targets = pl.DataFrame(
        {"function_id": [1, 2], "dimension": [2, 2], "target": [1.0, 0.4]}
    )
    for method in ("glicko2", "elo"):
        one = get_tournament_ranking(data, targets, seed=7, method=method)
        assert one.equals(get_tournament_ranking(data, targets, seed=7, method=method))
        assert not one.equals(
            get_tournament_ranking(data, targets, seed=8, method=method)
        )
        assert one["games"].to_list() == [50, 50]  # 25 rounds, 2 pairs, 1 opponent


def test_tournament_by_and_defaults():
    data = _runs({"A": [0.5, 0.0, 2.0, 1.5], "B": [5.0, 0.1]}).with_columns(
        sigma=pl.Series([1, 2, 1, 2, 1, 2])
    )
    out = get_tournament_ranking(data, by=["algorithm_name", "sigma"], seed=0, tau=1.2)
    assert out.columns[:3] == ["algorithm_name", "sigma", "rating"]
    assert out.height == 4 and (out["games"] == 75).all()
    with pytest.raises(ValueError, match="method"):
        get_tournament_ranking(data, seed=0, method="trueskill")


def test_get_tournament_ratings_is_deprecated():
    data = _runs({"A": [1.0], "B": [2.0]}).rename({"function_id": "function_name"})
    with pytest.warns(DeprecationWarning, match="get_tournament_ranking"):
        get_tournament_ratings(data)


def test_tournament_on_a_fixture():
    out = get_tournament_ranking(logged("ioh040_min"), seed=1)
    assert out["algorithm_name"].to_list()[-1] == "rs"
    assert (out["games"] == 6 * 25 * 2).all()
