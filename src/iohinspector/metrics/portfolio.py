"""Shapley values of algorithm portfolios (``docs/definitions/portfolio.md``,
SH-1 to SH-4)."""

import math
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

from ..targets import PAIR
from .ecdf import _hits, _prepare, _restarted
from .restarts import restarts

__all__ = [
    "envelope_value",
    "get_portfolio_ecdf",
    "get_shapley_values",
    "shapley_values",
]

#: v(S) for a batch of coalitions: a boolean matrix, one row per coalition and one
#: column per player, to one value per row.
Value = Callable[[np.ndarray], np.ndarray]
#: m(v, S, a): the marginal contribution of player ``a`` to each coalition (row) of
#: ``S``, none of which contains ``a``, with ``v`` the (cached) value function.
Marginal = Callable[[Value, np.ndarray, int], np.ndarray]

CHUNK = 1 << 22  # floats per block when the default value function takes a maximum


def _added(value: Value, coalitions: np.ndarray, player: int) -> np.ndarray:
    """The default marginal contribution v(S ∪ {a}) − v(S)."""
    joined = coalitions.copy()
    joined[:, player] = True
    return value(joined) - value(coalitions)


def _all_coalitions(n: int) -> np.ndarray:
    """Every coalition of ``n`` players; row k has player i iff bit i of k is set."""
    return (np.arange(1 << n)[:, None] >> np.arange(n)) & 1 == 1


def _codes(coalitions: np.ndarray) -> np.ndarray:
    return coalitions @ (1 << np.arange(coalitions.shape[1], dtype=np.int64))


def _cached(value: Value) -> Value:
    """``value`` evaluated once per distinct coalition (SH-4: caching of v(S))."""
    seen: dict[bytes, float] = {}

    def cached(coalitions: np.ndarray) -> np.ndarray:
        keys = [row.tobytes() for row in np.packbits(coalitions, axis=1)]
        new = {k: i for i, k in enumerate(keys) if k not in seen}
        if new:
            values = np.asarray(value(coalitions[list(new.values())]), dtype=float)
            seen.update(zip(new, values))
        return np.array([seen[k] for k in keys])

    return cached


def _exact(value: Value, marginal: Marginal, n: int):
    """Shapley values by enumerating every coalition (SH-2)."""
    table = np.asarray(value(_all_coalitions(n)), dtype=float)

    def lookup(coalitions: np.ndarray) -> np.ndarray:
        return table[_codes(coalitions)]

    sizes = np.arange(n)
    weight = np.array(
        [math.factorial(s) * math.factorial(n - s - 1) for s in sizes], dtype=float
    ) / math.factorial(n)
    phi = np.empty(n)
    for a in range(n):
        others = _all_coalitions(n - 1)
        coalitions = np.insert(others, a, False, axis=1)
        m = marginal(lookup, coalitions, a)
        phi[a] = weight[others.sum(axis=1)] @ m
    return phi, np.zeros(n), table[-1]


def _sampled(value: Value, marginal: Marginal, n: int, *, samples: int, seed):
    """Stratified estimate (SH-2): for every player a and position j, ``samples``
    random coalitions of j − 1 other players; φ_a is the mean over positions of
    the mean marginal contribution, with its standard error."""
    rng = np.random.default_rng(seed)
    value = _cached(value)
    phi, se = np.empty(n), np.empty(n)
    size = np.repeat(np.arange(n), samples)  # coalition size j − 1, per sample
    for a in range(n):
        order = np.argsort(rng.random((n * samples, n - 1)), axis=1)
        chosen = order < size[:, None]  # the first j − 1 of a random permutation
        coalitions = np.insert(chosen, a, False, axis=1)
        m = np.asarray(marginal(value, coalitions, a), dtype=float)
        m = m.reshape(n, samples)
        phi[a] = m.mean(axis=1).mean()
        var = m.var(axis=1, ddof=1) if samples > 1 else np.full(n, np.nan)
        se[a] = math.sqrt(var.sum() / samples) / n
    return phi, se, value(np.ones((1, n), dtype=bool))[0]


def shapley_values(  # pylint: disable=too-many-arguments
    n_players: int,
    value: Value,
    *,
    marginal: Marginal | None = None,
    exact_limit: int = 15,
    samples: int = 5,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shapley values of a cooperative game, normalised by v(N)
    (``docs/definitions/portfolio.md``, SH-2 to SH-4).

    φ_a = Σ_{S ⊆ N∖{a}} |S|! (|N| − |S| − 1)! / |N|! · m(S, a), divided by v(N), so
    that the values sum to 1 when m is the default v(S ∪ {a}) − v(S).

    The default portfolio value (``envelope_value``) with the default marginal is
    computed in closed form, exactly for any number of players (Q-SH-2). Any other
    value or marginal: up to ``exact_limit`` players (2^15 coalitions by default)
    every coalition is enumerated and v is evaluated once on all of them. Above
    that, φ_a is estimated by stratified sampling over **all** positions: for each
    position j = 1..|N|, ``samples`` random permutations with a at position j, that
    is ``samples`` random coalitions of j − 1 other players; φ_a is the mean over
    positions of the mean marginal contribution. v is evaluated once per distinct
    coalition.

    Args:
        n_players: |N|; players are the indices 0..n_players − 1.
        value: v, called with a boolean matrix (one row per coalition, one column per
            player) and returning one number per row. v(∅) is part of the game; the
            definitions use v(∅) = 0.
        marginal: m(v, S, a) for a boolean matrix S of coalitions without player a,
            returning one number per row; ``v`` is the cached value function. Default
            v(S ∪ {a}) − v(S).
        exact_limit: the largest number of players computed exactly by
            enumeration; not used by the closed form.
        samples: g, the coalitions per (player, position) when sampling; at least 2
            for a standard error.
        seed: the random seed when sampling.

    Returns:
        (φ, se): the normalised Shapley values and their standard errors (0 when
        exact). Both are NaN when v(N) is 0.
    """
    if n_players < 1:
        raise ValueError("a game needs at least one player")
    if marginal is None and isinstance(value, _Envelope):
        if len(value.ecdf) != n_players:
            raise ValueError("n_players is not the number of rows of the ECDF")
        (phi, total), se = value.shapley(), np.zeros(n_players)
    elif n_players <= exact_limit:
        phi, se, total = _exact(value, marginal or _added, n_players)
    else:
        phi, se, total = _sampled(
            value, marginal or _added, n_players, samples=samples, seed=seed
        )
    if total == 0:
        return np.full(n_players, np.nan), np.full(n_players, np.nan)
    return phi / total, se / total


class _Envelope:
    """SH-1's value v(S) = Σ width · max_{a ∈ S} ecdf[a], with its Shapley values
    in closed form (Q-SH-2)."""

    def __init__(self, ecdf: np.ndarray, width: np.ndarray):
        self.ecdf, self.width = ecdf, width

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        n, m = self.ecdf.shape
        out = np.empty(len(coalitions))
        step = max(1, CHUNK // max(m, 1))
        for start in range(0, len(coalitions), step):
            stop = start + step
            block = coalitions[start:stop]
            best = np.zeros((len(block), m))
            for a in range(n):
                np.maximum(best, block[:, a, None] * self.ecdf[a], out=best)
            out[start:stop] = best @ self.width
        return out

    def shapley(self) -> tuple[np.ndarray, float]:
        """(φ, v(N)) unnormalised. v is a sum of maximum games, one per step: with
        the ECDFs of a step sorted ascending, e_(1) ≤ … ≤ e_(n) and e_(0) = 0, the
        player in place i gets Σ_{j ≤ i} (e_(j) − e_(j−1)) / (n − j + 1)."""
        n, m = self.ecdf.shape
        share = 1 / (n - np.arange(n))[:, None]  # 1 / (n − j + 1)
        step = max(1, CHUNK // n)
        phi = np.zeros(n)
        for start in range(0, m, step):
            block = self.ecdf[:, start : start + step]
            order = np.argsort(block, axis=0, kind="stable")
            ranked = np.take_along_axis(block, order, axis=0)
            ranked = np.cumsum(np.diff(ranked, axis=0, prepend=0) * share, axis=0)
            gain = np.empty_like(block)
            np.put_along_axis(gain, order, ranked, axis=0)
            phi += gain @ self.width[start : start + step]
        return phi, float(self.ecdf.max(axis=0) @ self.width)


def envelope_value(ecdf: np.ndarray, width: np.ndarray) -> Value:
    """The portfolio value of SH-1 on the output of ``get_portfolio_ecdf``:
    v(S) = Σ width · max_{a ∈ S} ecdf[a], the area under the virtual best of S.
    With the default marginal, ``shapley_values`` computes its Shapley values in
    closed form, exactly for any number of players (Q-SH-2)."""
    return _Envelope(ecdf, width)


def _portfolio(  # pylint: disable=R0913,R0914
    data: pl.DataFrame,
    targets: pl.DataFrame | None,
    by: list[str],
    *,
    b_lo: float,
    b_hi: float | None,
    scale_eval_log: bool,
    scale_f_log: bool,
    maximization: bool,
    pairs: str,
    divide_by_dimension: bool,
    stopped: str,
    fval_var: str,
    eval_var: str,
):
    """``get_portfolio_ecdf`` with its options checked."""
    # #lizard forgives  (the options of get_shapley_values, passed on as one)
    data, targets = _prepare(
        data, targets, by, scale_f_log=scale_f_log, pairs=pairs, fval_var=fval_var
    )
    curve = list(dict.fromkeys([*by, *PAIR, "target"]))  # one ECDF per player and step
    data = data.with_columns(
        inc=1 / pl.col("run_id").n_unique().over(curve[:-1])  # EC-1: all runs
    )
    x = pl.col(eval_var).cast(pl.Float64)
    if divide_by_dimension:
        x = x / pl.col("dimension")
    lo = float(b_lo)
    hi = float(data.select(x.max()).item() if b_hi is None else b_hi)
    if lo >= hi or (scale_eval_log and lo <= 0):
        raise ValueError(f"budget range [{lo}, {hi}] is empty or not positive")
    hits = _hits(data, targets, maximization, fval_var, eval_var).with_columns(
        target=pl.col(fval_var).cast(pl.Float64)
    )
    if restarts(stopped):  # EC-7, per player, pair and target up to hi
        keep = hits.select(*curve, eval_var, "inc")
        extra = _restarted(
            data,
            hits,
            by,
            curve,
            hi,
            continuous=False,
            maximization=maximization,
            divide_by_dimension=divide_by_dimension,
            fval_var=fval_var,
            eval_var=eval_var,
        )
        hits = pl.concat([keep, extra.cast(keep.schema)])
    x = x.clip(lo, hi)
    if scale_eval_log:
        x, lo, hi = x.log10(), math.log10(lo), math.log10(hi)
    step = [*PAIR, "target"]
    hits = hits.with_columns(x=x)
    players = data.select(by).unique().sort(by, nulls_last=True)
    starts = targets.select(*PAIR, pl.col("target").cast(pl.Float64), x=pl.lit(lo))
    starts = starts.join(data.select(PAIR).unique(), on=PAIR, how="semi")
    grid = pl.concat([starts, hits.select(*step, "x")]).unique()
    grid = players.join(grid, how="cross").with_columns(inc=pl.lit(0.0))
    hits = hits.select(*curve, "x", "inc")
    steps = (
        pl.concat(
            [hits.with_columns(grid=False), grid.select(*curve, "x", "inc", grid=True)]
        )
        .sort([*curve, "x", "grid"], nulls_last=True)
        .with_columns(ecdf=pl.col("inc").cum_sum().over(curve))
        .filter("grid")
        .with_columns(
            width=pl.col("x").shift(-1, fill_value=hi).over(curve) - pl.col("x")
        )
    )
    n = players.height
    ecdf = steps["ecdf"].to_numpy().reshape(n, -1)
    width = steps["width"].to_numpy().reshape(n, -1)[0] / (hi - lo)
    keep = width > 0
    columns, index = np.unique(ecdf[:, keep], axis=1, return_inverse=True)
    return players, columns, np.bincount(index.ravel(), weights=width[keep])


#: The ``get_ecdf`` options of the portfolio functions and their defaults.
ECDF_OPTIONS = {
    "scale_f_log": True,
    "maximization": False,
    "pairs": "common",
    "divide_by_dimension": False,
    "stopped": "restart",
    "fval_var": "raw_y",
    "eval_var": "evaluations",
}


def _ecdf_options(options: dict) -> dict:
    unknown = set(options) - set(ECDF_OPTIONS)
    if unknown:
        raise TypeError(f"unknown options {sorted(unknown)}")
    return ECDF_OPTIONS | options


def get_portfolio_ecdf(
    data: pl.DataFrame,
    targets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    b_lo: float = 1.0,
    b_hi: float | None = None,
    scale_eval_log: bool = False,
    **options,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """The ECDF data of the portfolio game of ``get_shapley_values``, to define
    another portfolio value or marginal contribution on (SH-4).

    The single-target ECDF of each group of ``by`` on every (pair, target) is a step
    function over [``b_lo``, ``b_hi``]. All groups share the steps of a (pair,
    target); identical steps are merged, so a step loses its pair, target and
    budget, and its width is summed.

    Args: as in ``get_shapley_values``.

    Returns:
        (players, ecdf, width): the groups of ``by`` (one row per player, in the
        order of the rows of ``ecdf``), the ECDF of each player on each step (players
        × steps) and the width of each step on the budget axis, divided by the
        length of the axis (the widths of one (pair, target) sum to 1).
    """
    if not by:
        raise ValueError("by names the players: at least one column")
    return _portfolio(
        data,
        targets,
        list(by),
        b_lo=b_lo,
        b_hi=b_hi,
        scale_eval_log=scale_eval_log,
        **_ecdf_options(options),
    )


def get_shapley_values(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    targets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    b_lo: float = 1.0,
    b_hi: float | None = None,
    scale_eval_log: bool = False,
    marginal: Marginal | None = None,
    exact_limit: int = 15,
    samples: int = 5,
    seed: int | None = None,
    **options,
) -> pl.DataFrame:
    """The Shapley value of each group of ``by`` (a player, such as an algorithm) in
    the portfolio of all groups (``docs/definitions/portfolio.md``).

    The value of a portfolio S is the virtual best of S (SH-1): on every (function,
    dimension) pair and target, the largest single-target ECDF of a member (the
    fraction of all its runs that reached the target, EC-1), integrated exactly over
    [``b_lo``, ``b_hi``] as in ``get_ecdf_auc`` (EC-4), summed over pairs and
    targets. The Shapley values are normalised by v(N) and sum to 1 (SH-3); they
    are exact for any number of groups, in closed form (SH-2, Q-SH-2). With a
    custom ``marginal`` they are exact up to ``exact_limit`` groups and sampled,
    seeded, above.
    Another value or marginal contribution on the same ECDF data: call
    ``shapley_values`` on the output of ``get_portfolio_ecdf``.

    Args:
        data, targets, by: as in ``get_ecdf``; ``by`` is the players (D17), at least
            one column.
        b_lo, b_hi, scale_eval_log: the budget range and axis, as in
            ``get_ecdf_auc``.
        marginal, exact_limit, samples, seed: as in ``shapley_values`` (SH-4);
            ``exact_limit``, ``samples`` and ``seed`` only apply with a
            ``marginal``.
        **options: ``scale_f_log``, ``maximization``, ``pairs``,
            ``divide_by_dimension``, ``stopped`` (simulated restarts up to
            ``b_hi`` by default, EC-7), ``fval_var`` and ``eval_var`` as in
            ``get_ecdf``.

    Returns:
        pl.DataFrame: ``by``, ``shapley`` and ``se`` (its standard error; 0 when
        exact), one row per group.
    """
    players, ecdf, width = get_portfolio_ecdf(
        data,
        targets,
        by,
        b_lo=b_lo,
        b_hi=b_hi,
        scale_eval_log=scale_eval_log,
        **options,
    )
    phi, se = shapley_values(
        players.height,
        envelope_value(ecdf, width),
        marginal=marginal,
        exact_limit=exact_limit,
        samples=samples,
        seed=seed,
    )
    return players.with_columns(shapley=pl.Series(phi), se=pl.Series(se))
