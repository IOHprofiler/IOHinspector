"""Tournament ratings (``docs/definitions/glicko2.md``, GL-1 to GL-4) and robust
rankings."""

import itertools
import math
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import polars as pl
from skelo.model.elo import EloEstimator

from .._upgrade import upgrade_hints
from ..indicators import add_indicator
from ..targets import targets_per_pair
from .distributions import get_runtime_sample, get_value_sample
from .stats import per_pair

#: Glicko-2 scale: rating points per Glicko-2 unit is 1 / Q.
Q = math.log(10) / 400
#: R's default tolerance of ``optimize``, ``.Machine$double.eps^0.25``.
FMIN_TOL = sys.float_info.epsilon**0.25


# fmin mirrors R's Brent_fmin line by line, so that the volatility matches
# PlayerRatings; its length and branching are R's (accepted complexity warnings).
def fmin(  # pylint: disable=too-many-locals,too-many-branches
    f: Callable[[float], float], a: float, b: float, tol: float = FMIN_TOL
) -> float:
    """The minimiser of ``f`` on [a, b] by Brent's method: a line-by-line port of
    R 4.3.3's ``optimize`` (``Brent_fmin`` in ``stats/src/optimize.c``), where a
    non-finite value counts as the largest double."""

    # #lizard forgives  (a line-by-line port of R's C routine)
    def fn(x: float) -> float:
        y = f(x)
        return y if math.isfinite(y) else sys.float_info.max

    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(sys.float_info.epsilon)
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fv = fw = fx = fn(x)
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            return x
        p = q = r = 0.0
        if abs(e) > tol1:  # fit a parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)  # a golden-section step
            d = c * e
        else:  # a parabolic-interpolation step
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + tol1 if d > 0.0 else x - tol1
        fu = fn(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu


def _games(games: pl.DataFrame):
    """The columns of a game list as numpy arrays, and its rounds in order."""
    cols = [games[c].to_numpy() for c in ("round", "player", "opponent", "score")]
    return cols, np.unique(cols[0])


def _record(player, opponent, score, n: int) -> np.ndarray:
    """Games, wins, draws and losses of each player in a list of games."""
    both = np.concatenate([player, opponent])
    mine = np.concatenate([score, 1 - score])
    counts = [both, *(both[mine == s] for s in (1, 0.5, 0))]
    return np.stack([np.bincount(c, minlength=n) for c in counts])


# glicko2 mirrors PlayerRatings::glicko2 (R and C) line by line, so that the
# ratings match it; its length is PlayerRatings' (accepted complexity warnings).
def _table(rating, deviation, volatility, record: np.ndarray) -> pl.DataFrame:
    """The ratings table: one row per player."""
    counts = zip(("games", "wins", "draws", "losses"), record.astype(np.int64))
    return pl.DataFrame(
        {
            "rating": pl.Series(rating, dtype=pl.Float64),
            "deviation": pl.Series(deviation, dtype=pl.Float64),
            "volatility": pl.Series(volatility, dtype=pl.Float64),
            **dict(counts),
        }
    )


def glicko2(  # pylint: disable=too-many-locals
    games: pl.DataFrame,
    n_players: int,
    *,
    tau: float = 0.5,
    init: tuple = (1500, 350, 0.06),
    rdmax: float = 350,
) -> pl.DataFrame:
    """Glicko-2 ratings from a game list, each round one rating period (GL-2).

    A port of ``PlayerRatings::glicko2`` 1.1-0 without first-move advantage: a
    player's deviation grows with its volatility over the periods it sat out, and
    is capped at ``rdmax``; the new volatility minimises PlayerRatings' objective
    with R's ``optimize`` (``fmin``), so given the same games the ratings agree.

    Args:
        games: ``round``, ``player`` and ``opponent`` (integers 0 .. n_players - 1)
            and ``score``, the player's score (1 win, 0.5 draw, 0 loss), in the
            order played.
        n_players: the number of players.
        tau: the system constant τ (PlayerRatings' and IOHanalyzer's is 1.2).
        init: the initial rating, deviation and volatility, each a number or one
            value per player.
        rdmax: the largest deviation.

    Returns:
        pl.DataFrame: per player, in index order: ``rating``, ``deviation``,
            ``volatility``, ``games``, ``wins``, ``draws``, ``losses``.
    """
    # #lizard forgives  (a line-by-line port of PlayerRatings)
    (rnd, white, black, score), rounds = _games(games)
    rating, deviation, vol = (
        np.broadcast_to(np.asarray(v, float), n_players) for v in init
    )
    rats = Q * (rating - 1500)
    devs = (Q * deviation) ** 2
    vols = vol.copy()
    lag = np.zeros(n_players)
    played = np.zeros(n_players, dtype=bool)
    rdmax2 = Q * Q * rdmax * rdmax
    tau2 = tau * tau
    for k in rounds:
        mask = rnd == k
        w, b, s = white[mask], black[mask], score[mask].astype(float)
        both = np.concatenate([w, b])
        play = both[np.sort(np.unique(both, return_index=True)[1])]
        devs[play] = np.minimum(devs[play] + lag[play] * vols[play] ** 2, rdmax2)
        g = 1 / np.sqrt(1 + 3 * (1 / math.pi) ** 2 * devs)
        e_w = 1 / (1 + np.exp(g[b] * (rats[b] - rats[w])))
        e_b = 1 / (1 + np.exp(g[w] * (rats[w] - rats[b])))
        # interleaved, so that each player's sums add up in game order, as in C
        index = np.ravel(np.column_stack([w, b]))
        vval = np.column_stack(
            [g[b] ** 2 * e_w * (1 - e_w), g[w] ** 2 * e_b * (1 - e_b)]
        )
        dscore = np.column_stack([g[b] * (s - e_w), g[w] * (1 - s - e_b)])
        vval = np.bincount(index, np.ravel(vval), minlength=n_players)
        dscore = np.bincount(index, np.ravel(dscore), minlength=n_players)
        if tau > 0:
            for i in play:
                prv = 2 * math.log(vols[i])
                cdev, vvi, dsco = devs[i], 1 / vval[i], dscore[i]
                delta = vvi * dsco

                def nllh(z, prv=prv, cdev=cdev, vvi=vvi, delta=delta):
                    denom = cdev + math.exp(z) + vvi
                    return (
                        (z - prv) * (z - prv) / tau2
                        + math.log(denom)
                        + delta * delta / denom
                    )

                z = fmin(nllh, prv - 4 * tau, prv + 4 * tau)
                vols[i] = min(math.exp(z / 2), Q * rdmax)
        devs[play] = devs[play] + vols[play] ** 2
        devs = np.minimum(1 / (1 / devs + vval), rdmax2)
        rats = rats + devs * dscore
        played[both] = True
        lag[played] += 1
        lag[play] = 0
    record = _record(white, black, score, n_players)
    return _table(rats / Q + 1500, np.sqrt(devs) / Q, vols, record)


def elo(
    games: pl.DataFrame, n_players: int, *, k: float = 20, init: float = 1500
) -> pl.DataFrame:
    """Elo ratings from a game list, rated one game at a time in the given order
    (GL-4): E = 1 / (1 + 10^((R_c - R_a) / 400)), R_a' = R_a + k (S - E), the
    defaults of skelo 0.1.5. Elo has no deviation; ``deviation`` is the standard
    deviation of a player's ratings after its games in the last 5 % of the rounds
    (at least the last round), as iohinspector 0.0.8 reported.

    Args:
        games: as for ``glicko2``.
        n_players: the number of players.
        k: the K factor.
        init: the initial rating.

    Returns:
        pl.DataFrame: as ``glicko2``, with a null ``volatility``.
    """
    (rnd, player, opponent, score), rounds = _games(games)
    first = rounds[-max(1, len(rounds) - math.ceil(0.95 * len(rounds)))]
    rows = zip(*(v.tolist() for v in (rnd, player, opponent, score)))
    ratings, history = _elo_updates(rows, n_players, k=k, init=init, first=first)
    spread = [np.std(h, ddof=1) if len(h) > 1 else None for h in history]
    record = _record(player, opponent, score, n_players)
    return _table(ratings, spread, [None] * n_players, record)


def _elo_updates(rows, n_players: int, *, k: float, init: float, first: int):
    """The final Elo ratings, and each player's ratings after its games from round
    ``first`` on."""
    ratings = [float(init)] * n_players
    history: list[list[float]] = [[] for _ in range(n_players)]
    for r, a, c, s in rows:
        change = k * (s - 1 / (1 + 10 ** ((ratings[c] - ratings[a]) / 400)))
        ratings[a] += change
        ratings[c] -= change
        if r >= first:
            history[a].append(ratings[a])
            history[c].append(ratings[c])
    return ratings, history


def _matches(data, targets, by, *, fixed_target, maximization, **columns):
    """Per (function, dimension) pair and target, each group's values oriented so
    that smaller is better, with no value as ``inf`` (a failure, GL-1)."""
    sample, value = (
        (get_runtime_sample, "runtime") if fixed_target else (get_value_sample, "value")
    )
    x = "target" if fixed_target else "budget"
    sign = -1.0 if maximization and not fixed_target else 1.0
    for _, pair, at in per_pair(data, targets):
        at = at.to_list() if fixed_target else at.cast(pl.Int64).to_list()
        values = sample(pair, at, by, maximization=maximization, **columns)
        for match in values.sort(x).partition_by(x):
            groups = match.group_by(by).agg(pl.col(value).cast(pl.Float64) * sign)
            groups = groups.sort(by)
            yield (
                groups.select(by),
                [v.fill_null(np.inf).to_numpy() for v in groups[value]],
            )


def _play(players, rounds: int, rng: np.random.Generator) -> list[tuple]:
    """One game per round between each unordered pair of ``players`` ((id,
    values) pairs): one value of each drawn at random, the smaller wins, equal
    values draw. Returns (player, opponent, scores) per pair."""
    games = []
    for (i, x), (j, y) in itertools.combinations(players, 2):
        a = x[rng.integers(len(x), size=rounds)]
        c = y[rng.integers(len(y), size=rounds)]
        games.append((i, j, (a < c) + 0.5 * (a == c)))
    return games


def _tournament(data, targets, by, *, rounds, rng, **options):
    """The games of the tournament (GL-1), with the groups of ``by`` as players
    0 .. n - 1 in the order of ``groups``."""
    groups = data.select(by).unique().sort(by)
    index = groups.with_row_index("player")
    games = []
    for names, values in _matches(data, targets, by, **options):
        ids = names.join(index, on=by, nulls_equal=True, maintain_order="left")
        games += _play(zip(ids["player"].to_numpy(), values), rounds, rng)
    players, opponents, scores = zip(*games)
    games = pl.DataFrame(
        {
            "round": np.tile(np.arange(1, rounds + 1), len(scores)),
            "player": np.repeat(players, rounds),
            "opponent": np.repeat(opponents, rounds),
            "score": np.concatenate(scores),
        }
    )
    return groups, games.sort("round", maintain_order=True)


def _rate(games, n_players: int, *, method: str, tau: float, rng) -> pl.DataFrame:
    """Glicko-2 ratings, or Elo ratings with each round in random order."""
    if method == "glicko2":
        return glicko2(games, n_players, tau=tau)
    rounds = np.unique(games["round"].to_numpy())
    order = [rng.permutation(np.flatnonzero(games["round"] == r)) for r in rounds]
    return elo(games[np.concatenate(order)], n_players)


def get_tournament_ranking(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    targets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    seed: int,
    perspective: str = "fixed_target",
    method: str = "glicko2",
    rounds: int = 25,
    tau: float = 0.5,
    maximization: bool = False,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
) -> pl.DataFrame:
    """Ratings of the groups of ``by`` from a tournament of random games
    (``docs/definitions/glicko2.md``, GL-1 to GL-4).

    Each round, on each (function, dimension) pair and target, every unordered pair
    of groups plays one game: one run of each is drawn at random, with
    replacement, and the better runtime (fixed target) or value (fixed budget)
    wins. A failed run loses to a successful one, two failures draw, equal values
    draw. Each round is one Glicko-2 rating period; Elo rates the games one by one,
    within a round in random order. The same ``seed`` gives the same ratings.

    Args:
        data: the runs, with ``run_id``, ``function_id``, ``dimension``,
            ``eval_var``, ``fval_var`` and ``by``.
        targets: targets (budgets) per pair: ``function_id``, ``dimension`` and
            ``target``; one per pair from ``get_target_dt`` when None.
        by: the columns that make a group (a player).
        seed: the seed of the random games.
        perspective: ``"fixed_target"`` or ``"fixed_budget"``.
        method: ``"glicko2"`` or ``"elo"`` (K = 20).
        rounds: the number of rounds.
        tau: Glicko-2's system constant τ (IOHanalyzer's is 1.2).
        maximization: whether larger function values are better.
        eval_var: the column of evaluation numbers.
        fval_var: the column of function values.

    Returns:
        pl.DataFrame: per group, best first: the ``by`` columns, ``rating``,
            ``deviation``, ``volatility`` (null for Elo), ``games``, ``wins``,
            ``draws`` and ``losses``.
    """
    if method not in ("glicko2", "elo"):
        raise ValueError(f"method {method!r}: glicko2 or elo")
    by = list(by)
    columns = {"eval_var": eval_var, "fval_var": fval_var}
    targets = targets_per_pair(
        data, targets, perspective, maximization=maximization, by=by, **columns
    )
    rng = np.random.default_rng(seed)
    groups, games = _tournament(
        data,
        targets,
        by,
        rounds=rounds,
        rng=rng,
        fixed_target=perspective == "fixed_target",
        maximization=maximization,
        **columns,
    )
    ratings = _rate(games, groups.height, method=method, tau=tau, rng=rng)
    return pl.concat([groups, ratings], how="horizontal").sort(
        "rating", descending=True, maintain_order=True
    )


@upgrade_hints
def get_tournament_ratings(
    data: pl.DataFrame,
    by: Sequence[str] = ("algorithm_name",),
    fid_vars: Iterable[str] = ["function_name"],
    fval_var: str = "raw_y",
    nrounds: int = 25,
    maximization: bool = False,
) -> pl.DataFrame:
    """Calculate ELO tournament ratings for algorithms competing on multiple problems.

    Args:
        data (pl.DataFrame): The data object containing algorithm performance data.
        by (Sequence[str], optional): Which columns specify the algorithms that will
            compete. Defaults to ["algorithm_name"].
        fid_vars (Iterable[str], optional): Which columns denote the problems on which competition occurs. Defaults to ["function_name"].
        fval_var (str, optional): Which column contains the performance values. Defaults to "raw_y".
        nrounds (int, optional): Number of tournament rounds to play. Defaults to 25.
        maximization (bool, optional): Whether the performance metric is being maximized. Defaults to False.

    Returns:
        pl.DataFrame: A DataFrame with ELO ratings, deviations, and algorithm
            identifiers.

    Deprecated: use ``get_tournament_ranking(data, method="elo", seed=...)``, which
    plays the tournament of ``docs/definitions/glicko2.md`` (GL-4).
    """
    warnings.warn(
        "get_tournament_ratings is deprecated; use get_tournament_ranking(data, "
        'method="elo", seed=...), or the default method="glicko2"',
        DeprecationWarning,
        stacklevel=3,
    )
    fids = data[fid_vars].unique()
    by = list(by)
    aligned_comps = data.pivot(
        index=by,
        on=fid_vars,
        values=fval_var,
        aggregate_function=pl.element(),
    )
    players = aligned_comps[by]
    n_players = players.shape[0]
    first = len(by)
    comp_arr = np.array(aligned_comps[aligned_comps.columns[first:]])

    rng = np.random.default_rng()
    fids = [i for i in range(len(fids))]
    lplayers = [i for i in range(n_players)]
    records = []
    for r in range(nrounds):
        for fid in fids:
            for p1 in lplayers:
                for p2 in lplayers:
                    if p1 == p2:
                        continue
                    s1 = rng.choice(comp_arr[p1][fid], 1)[0]
                    s2 = rng.choice(comp_arr[p2][fid], 1)[0]
                    if s1 == s2:
                        won = 0.5
                    else:
                        won = abs(float(maximization) - float(s1 < s2))

                    records.append([r, p1, p2, won])
    dt_comp = pd.DataFrame.from_records(
        records, columns=["round", "p1", "p2", "outcome"]
    )
    dt_comp = dt_comp.sample(frac=1).sort_values("round")
    model = EloEstimator(key1_field="p1", key2_field="p2", timestamp_field="round").fit(
        dt_comp, dt_comp["outcome"]
    )
    model_dt = model.rating_model.to_frame()
    ratings = np.array(model_dt[np.isnan(model_dt["valid_to"])]["rating"])
    deviations = (
        model_dt.query(f"valid_from >= {nrounds * 0.95}").groupby("key")["rating"].std()
    )

    rating_dt_elo = pd.DataFrame(
        [
            ratings,
            deviations,
            *players[players.columns],
        ]
    ).transpose()
    rating_dt_elo.columns = ["Rating", "Deviation", *players.columns]
    return pl.from_pandas(rating_dt_elo)


def _indicator_table(
    data: pl.DataFrame, obj_vars: Iterable[str], evals: list[int], indicator: object
) -> pd.DataFrame:
    """The indicator at ``evals``: one row per algorithm and run, as robustranking
    takes it. Runs are numbered 1, 2, ... per algorithm in the order of ``data``, and
    robustranking pairs the runs of the algorithms by that number."""
    run = pl.col("run_id").is_first_distinct().cum_sum().over("algorithm_name")
    data = data.with_columns(run=run)
    df = add_indicator(data, indicator, obj_vars=obj_vars, evals=evals).to_pandas()
    df_part = df[["evaluations", indicator.var_name, "algorithm_name", "run"]]
    dt_pivoted = pd.pivot(
        df_part,
        index=["algorithm_name", "run"],
        columns=["evaluations"],
        values=[indicator.var_name],
    ).reset_index()
    dt_pivoted.columns = ["algorithm_name", "run"] + list(evals)
    return dt_pivoted


def get_robustrank_over_time(
    data: pl.DataFrame,
    obj_vars: Iterable[str],
    evals: Iterable[int],
    indicator: object,
):
    """Calculate robust ranking data over multiple time points for multi-objective optimization.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization trajectory data.
        obj_vars (Iterable[str]): Which columns correspond to the objective values.
        evals (Iterable[int]): Evaluation time points at which to calculate rankings.
        indicator (object): Indicator object from iohinspector.indicators for performance measurement.

    Returns:
        tuple: A tuple containing (comparison, benchmark) objects for robust ranking analysis.
    """
    from robustranking import Benchmark
    from robustranking.comparison import MOBootstrapComparison

    dt_pivoted = _indicator_table(data, obj_vars, evals, indicator)
    benchmark = Benchmark()
    benchmark.from_pandas(dt_pivoted, "algorithm_name", "run", evals)
    comparison = MOBootstrapComparison(
        benchmark,
        alpha=0.05,
        minimise=indicator.minimize,
        bootstrap_runs=1000,
        aggregation_method=np.mean,
    )

    return comparison, benchmark


def get_robustrank_changes(
    data: pl.DataFrame,
    obj_vars: Iterable[str],
    evals: Iterable[int],
    indicator: object,
):
    """Calculate robust ranking changes across multiple evaluation time points.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization trajectory data.
        obj_vars (Iterable[str]): Which columns correspond to the objective values.
        evals (Iterable[int]): Evaluation time points at which to calculate ranking changes.
        indicator (object): Indicator object from iohinspector.indicators for performance measurement.

    Returns:
        dict: A dictionary of comparison objects for each evaluation time point showing ranking changes.
    """
    from robustranking import Benchmark
    from robustranking.comparison import BootstrapComparison

    dt_pivoted = _indicator_table(data, obj_vars, evals, indicator)

    comparisons = {
        f"{eval}": BootstrapComparison(
            Benchmark().from_pandas(dt_pivoted, "algorithm_name", "run", eval),
            alpha=0.05,
            minimise=indicator.minimize,
            bootstrap_runs=1000,
        )
        for eval in evals
    }

    return comparisons
