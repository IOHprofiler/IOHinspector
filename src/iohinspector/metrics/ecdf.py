"""ECDF, its area (AUC, AOCC), the fixed-budget ECDF and its area, and
multi-function rank tables (``docs/definitions/ecdf.md``, EC-1 to EC-10)."""

import math
import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

from .._upgrade import upgrade_hints
from ..align import align_data
from ..targets import PAIR, get_ecdf_budgets, get_ecdf_targets, seq_rt
from .restarts import continued, failed, gained, restarts, run_sets, split
from .utils import runs_of

ALL = "_all"  # the one group when ``by`` is empty


def _weights(data: pl.DataFrame, by: list[str], pairs: str):
    """``data`` on the pairs used (EC-2), with ``weight`` 1 / (|P| n): each pair of a
    group of ``by`` has equal weight, and within a pair each run."""
    if pairs not in ("common", "all"):
        raise ValueError(f"pairs {pairs!r}: common or all")
    groups = [c for c in by if c not in PAIR]
    if groups:
        sets = data.select(*groups, *PAIR).unique()
        every = (
            sets.select(groups).unique().join(sets.select(PAIR).unique(), how="cross")
        )
        missing = every.join(sets, on=[*groups, *PAIR], how="anti", nulls_equal=True)
        if missing.height:
            _warn_pairs(missing.sort(missing.columns), len(groups), pairs)
        if missing.height and pairs == "common":
            drop = missing.select(PAIR).unique()
            data = data.join(drop, on=PAIR, how="anti", maintain_order="left")
            if data.is_empty():
                raise ValueError("the groups of by share no pair; use pairs='all'")
    n_runs = pl.col("run_id").n_unique().over(list(dict.fromkeys([*by, *PAIR])))
    n_pairs = pl.struct(PAIR).n_unique().over(by)
    return data.with_columns(weight=1 / (n_runs * n_pairs))


def _warn_pairs(missing: pl.DataFrame, n: int, pairs: str) -> None:
    rows = [f"{', '.join(map(str, r[:n]))} has no {r[n:]}" for r in missing.rows()]
    more = f"; and {len(rows) - 10} more" if len(rows) > 10 else ""
    action = (
        "only the pairs every group has are used (pairs='common')"
        if pairs == "common"
        else "each group is averaged over its own pairs (pairs='all'), "
        "so the curves are hard to compare"
    )
    warnings.warn(
        f"the groups of by do not have the same {tuple(PAIR)} pairs: {action}. "
        + "; ".join(rows[:10])
        + more,
        stacklevel=6,
    )


def _aligned(data, grid, column, **options):
    """Each run aligned on its pair's values of ``grid[column]`` (``align_data``
    with ``options``), with ``dy`` = 1/m for its m values."""
    parts = grid.partition_by(PAIR, as_dict=True)
    return pl.concat(
        align_data(runs, parts[key][column], **options).with_columns(
            dy=1 / pl.len().over("run_id")
        )
        for key, runs in data.partition_by(PAIR, as_dict=True).items()
    )


def _hits(data, targets, maximization, fval_var, eval_var):
    """Increments ``dy`` = 1/m of each (run, target) pair at its first hitting time
    (EC-1: inclusive, AL-1; ties to the first row, AL-3), with the other columns of
    the row that reached it."""
    aligned = _aligned(
        data,
        targets,
        "target",
        x_col=fval_var,
        y_col=eval_var,
        maximization=maximization,
        eval_var=eval_var,
    )
    return aligned.filter(pl.col(eval_var).is_finite())


def _quality(data, targets, maximization, scale_f_log, fval_var):
    """Each run's normalised quality ``q`` (EC-5) on every row, and its increment
    ``dy``: the log10 (or linear) distance of its best-so-far value from the worse
    bound, between the pair's smallest and largest target. q is the fraction of a
    continuum of targets between the bounds that the run has reached."""
    bounds = targets.group_by(PAIR).agg(
        lo=pl.col("target").min(), hi=pl.col("target").max()
    )
    f = pl.col(fval_var).cast(pl.Float64)  # clip would truncate float bounds to int
    best = f.cum_max() if maximization else f.cum_min()
    lo, hi = pl.col("lo"), pl.col("hi")
    scale = (lambda e: e.log10()) if scale_f_log else (lambda e: e)
    v = scale(pl.col("best").clip(lo, hi))
    q = (v - scale(lo)) if maximization else (scale(hi) - v)
    reached = (pl.col("best") >= hi) if maximization else (pl.col("best") <= lo)
    q = (
        pl.when(hi == lo)
        .then(reached.cast(pl.Float64))
        .otherwise(q / (scale(hi) - scale(lo)))
    )
    previous = pl.col("q").shift(1, fill_value=0).over("run_id")
    return (
        data.join(bounds, on=PAIR, maintain_order="left")  # rows in run order
        .with_columns(best=best.over("run_id"))
        .with_columns(q=q)
        .with_columns(dy=pl.col("q") - previous)
        .drop("lo", "hi")
    )


def _prepare(
    data: pl.DataFrame,
    targets: pl.DataFrame | None,
    by,
    *,
    scale_f_log,
    pairs,
    fval_var,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``data`` on the pairs that have targets and are used (EC-2), with ``weight``,
    and the targets (by default 10 per pair, log-linear or linear)."""
    if targets is None:
        kind = "log-linear" if scale_f_log else "linear"
        targets = get_ecdf_targets(data, kind, fval_var=fval_var)
    targets = targets.cast({c: data.schema[c] for c in PAIR})
    data = data.join(targets.select(PAIR), on=PAIR, how="semi", maintain_order="left")
    return _weights(data, by, pairs), targets


def _restarted(  # pylint: disable=R0913
    data: pl.DataFrame,
    steps: pl.DataFrame,
    by: list[str],
    key: list[str],
    horizon: float,
    *,
    samples: list[float] | None = None,
    area: tuple[float, bool] | None = None,
    continuous: bool,
    maximization: bool,
    divide_by_dimension: bool,
    fval_var: str,
    eval_var: str,
) -> pl.DataFrame:
    """The increments that simulated restarts add to ``steps`` (EC-7), per curve of
    ``key`` up to the budget ``horizon``: ``key``, ``eval_var`` and ``inc``. The
    restarts of a run draw from its group of ``by`` in its pair. ``steps`` are the
    hits of targets (``fval_var`` the target, ``inc`` the weighted increment) or,
    ``continuous``, the rows of ``_quality`` with ``inc``. With ``samples`` (x
    values) only what the ECDF needs to be read at them: the increments summed at
    the last evaluation at or before each. With ``area`` (b_lo and the log axis
    of ``get_ecdf_auc``), only their area up to ``horizon``: per curve and chunk
    an increment at x = 0, each increment weighted by the part of the axis it
    covers (``_area``)."""
    sets = list(dict.fromkeys([*by, *PAIR]))
    if continuous:  # each increment of q cut where the set's runs end
        runs = run_sets(steps, sets, eval_var, pl.col("q").max())
        steps = steps.filter(pl.col("dy") != 0).with_columns(
            lo=pl.col("q") - pl.col("dy")
        )
        steps = split(runs, steps.rename({"q": "hi"}))
        steps = steps.with_columns(inc=(pl.col("hi") - pl.col("lo")) * pl.col("weight"))
    else:
        level = pl.col(fval_var).cast(pl.Float64) * (1.0 if maximization else -1.0)
        runs = run_sets(data, sets, eval_var, level.max())
        steps = failed(runs, steps.with_columns(level=level), "level")
    steps = steps.with_columns(horizon=_evaluations(horizon, divide_by_dimension))
    at = None
    if samples is not None:
        x = pl.DataFrame({"_x": samples}, schema={"_x": pl.Float64})
        at = steps.select(key).unique().join(x, how="cross")
        at = at.select(*key, _last_at(divide_by_dimension).alias(eval_var))
        at = at.filter(pl.col(eval_var) >= 0)
    weighted = None
    if area is not None:
        x, lo, hi = _axis(_x(eval_var, divide_by_dimension), area[0], horizon, area[1])
        weighted = (hi - x) / (hi - lo)
    return continued(steps, runs, key, eval_var, at, weighted)


def _x(eval_var: str, divide_by_dimension: bool) -> pl.Expr:
    """The x axis: the evaluations, divided by the dimension if asked."""
    x = pl.col(eval_var).cast(pl.Float64)
    return x / pl.col("dimension") if divide_by_dimension else x


def _last_at(divide_by_dimension: bool) -> pl.Expr:
    """The last evaluation e with x(e) <= ``_x``, x being the evaluations, divided
    by the row's dimension if ``divide_by_dimension`` (as ``_increments``)."""
    x = pl.col("_x")
    if not divide_by_dimension:
        return x.floor().cast(pl.Int64)
    d = pl.col("dimension").cast(pl.Float64)
    e = (x * d).floor()  # then the rounding of x * d undone, as e / d is compared
    e = pl.when((e + 1) / d <= x).then(e + 1).otherwise(e)
    return pl.when(e / d > x).then(e - 1).otherwise(e).cast(pl.Int64)


def _evaluations(budget: float, divide_by_dimension: bool) -> pl.Expr:
    """``budget`` on the x axis in evaluations, per row's dimension if divided."""
    evaluations = pl.lit(float(budget))
    if divide_by_dimension:
        evaluations = evaluations * pl.col("dimension")
    return evaluations.floor().cast(pl.Int64)


def _stopped_before(data, by, eval_var, budget: pl.Expr, per_pair=None):
    """``n_restarted`` per group of ``by``: its runs with B_i below ``budget``, the
    largest budget of their pair (EC-7), from the pair columns or ``per_pair``."""
    runs = runs_of(data, list(dict.fromkeys([*by, *PAIR])), eval_var)
    if per_pair is not None:
        runs = runs.join(per_pair, on=PAIR)
    return runs.group_by(by).agg(n_restarted=(pl.col("evals_used") < budget).sum())


def _increments(  # pylint: disable=R0913,R0914
    data: pl.DataFrame,
    targets: pl.DataFrame | None,
    by: list[str],
    *,
    continuous: bool = False,
    scale_f_log: bool = True,
    maximization: bool = False,
    pairs: str = "common",
    divide_by_dimension: bool = False,
    stopped: str = "restart",
    horizon: float | None = None,
    samples: list[float] | None = None,
    area: tuple[float, bool] | None = None,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
) -> tuple[pl.DataFrame, float, pl.DataFrame | None]:
    """The weighted ECDF increments ``inc`` at ``x`` per group of ``by`` (with a 0 at
    x = 0 for every group), with simulated restarts up to ``horizon`` (by default the
    largest budget in the selection), that largest budget, and with restarts
    ``n_restarted`` per group of ``by``. With ``samples``, the restarts' increments
    are only right at those x values, with ``area`` only their area (``_restarted``)."""
    # #lizard forgives  (the options of get_ecdf and get_ecdf_auc, passed on as one)
    restart = restarts(stopped)
    data, targets = _prepare(
        data, targets, by, scale_f_log=scale_f_log, pairs=pairs, fval_var=fval_var
    )
    x = _x(eval_var, divide_by_dimension)
    x_max = data.select(x.max()).item()
    if continuous:
        steps = _quality(data, targets, maximization, scale_f_log, fval_var)
    else:
        steps = _hits(data, targets, maximization, fval_var, eval_var)
    steps = steps.with_columns(inc=pl.col("dy") * pl.col("weight"))
    keep = [*dict.fromkeys([*by, *PAIR]), eval_var, "inc"]
    counts = None
    if restart:
        until = float(x_max if horizon is None else horizon)
        counts = _stopped_before(
            data, by, eval_var, _evaluations(until, divide_by_dimension)
        )
        extra = _restarted(
            data,
            steps,
            by,
            keep[:-2],
            until,
            samples=samples,
            area=area,
            continuous=continuous,
            maximization=maximization,
            divide_by_dimension=divide_by_dimension,
            fval_var=fval_var,
            eval_var=eval_var,
        )
        steps = pl.concat([steps.select(keep), extra.cast(steps.select(keep).schema)])
    zero = data.select(by).unique().with_columns(x=pl.lit(0.0), inc=pl.lit(0.0))
    steps = steps.filter(pl.col("inc") != 0).select(*by, x=x, inc="inc")
    return pl.concat([zero, steps]), x_max, counts


def _one_group(data: pl.DataFrame, by: Sequence[str]):
    """``data`` and ``by`` as a list; without ``by`` one group, a column ``ALL``."""
    if by:
        return data, list(by)
    return data.with_columns(pl.lit(0).alias(ALL)), [ALL]


def _cumulate(steps: pl.DataFrame, by: list[str], order: Sequence[str] = ()):
    return steps.sort([*by, "x", *order]).with_columns(
        ecdf=pl.col("inc").cum_sum().over(by)
    )


def _step_function(steps, by, x_values):
    """The cumulated ``ecdf`` of the increments at each x where it steps (EC-3), or
    sampled at ``x_values``: ``by``, ``x`` and ``ecdf``."""
    steps = steps.with_columns(grid=pl.lit(False))
    if x_values is not None:
        grid = pl.DataFrame({"x": list(x_values)}, schema={"x": pl.Float64})
        grid = steps.select(by).unique().join(grid, how="cross")
        steps = pl.concat([steps, grid.with_columns(inc=0.0, grid=pl.lit(True))])
    steps = _cumulate(steps, by, ["grid"])
    if x_values is None:
        steps = steps.unique([*by, "x"], keep="last", maintain_order=True)
    else:
        steps = steps.filter("grid")
    return steps.select(*by, "x", "ecdf").drop(ALL, strict=False)


def _axis(x: pl.Expr, lo: float, hi: float, log: bool):
    """``x`` clipped to [lo, hi], and the interval, on a linear or log10 axis."""
    if lo >= hi or (log and lo <= 0):
        raise ValueError(f"range [{lo}, {hi}] is empty or not positive")
    x = x.clip(lo, hi)
    if log:
        return x.log10(), math.log10(lo), math.log10(hi)
    return x, lo, hi


def _area(steps, by, x, lo, hi):
    """The normalised area under each group's ECDF over [lo, hi] (EC-4, EC-10).
    ``x`` maps the step positions onto that interval, keeping their order."""
    width = pl.col("x").shift(-1, fill_value=hi).over(by) - pl.col("x")
    steps = _cumulate(steps, by).with_columns(x=x)
    steps = steps.with_columns(area=pl.col("ecdf") * width)
    auc = pl.col("area").sum() / (hi - lo)
    return steps.group_by(by, maintain_order=True).agg(auc=auc).drop(ALL, strict=False)


@upgrade_hints
def get_ecdf(
    data: pl.DataFrame,
    targets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    x_values: Iterable[float] | None = None,
    eval_var: str = "evaluations",
    **options,
) -> pl.DataFrame:
    """The ECDF of each group of ``by`` (``docs/definitions/ecdf.md``).

    The fraction of (run, target) pairs whose target the run reached within x
    evaluations: f <= t (f >= t when maximising) and T <= x, both inclusive (EC-1).
    It is averaged over the (function, dimension) pairs of a group, each pair with
    equal weight and, within a pair, each target and each run (EC-2). A run that
    stops (B_i, its ``evals_used`` or else its last logged evaluation) without
    reaching a target is continued by simulated restarts by default (EC-7):
    runs drawn uniformly from its group in its pair until one reaches the target,
    as in COCO. The ECDF is their expected value, computed from the runs'
    runtimes, up to the largest of ``x_values`` or else the largest evaluation in
    ``data``. It is exact (EC-3): without ``x_values`` the result has one row per x
    at which the ECDF steps, with its value from x on, and a row at x = 0 (value
    0) for every group.

    Args:
        data: the runs' logged rows in evaluation order (``DataManager.load``): at
            least ``run_id``, the pair columns, ``by``, ``eval_var`` and ``fval_var``.
            Best-so-far values are computed here (AL-6).
        targets: targets per pair, columns ``function_id``, ``dimension`` and
            ``target`` (``targets.get_ecdf_targets``). Defaults to 10 log-linear
            targets per pair over its observed values (TG-1, TG-2), or over 1e-8
            to 1e2 for BBOB and SBOX data with a ``suite`` column (TG-8); pairs
            without targets are left out.
        by: the columns that make a group (a curve): descriptors or runs columns
            (D17). Pair columns in ``by`` give one curve per pair.
        x_values: budgets to sample the ECDF at, for display (EC-3).
        eval_var: the evaluations column.
        **options: keyword-only:

            - ``continuous`` (False): the continuum ECDF (EC-5, EC-6), the mean
              over runs of the normalised quality of the best-so-far value between
              the pair's smallest and largest target. It is the fraction of
              infinitely many targets between them, log-uniform (uniform without
              ``scale_f_log``), that the run has reached. By default the bounds
              are each pair's observed range, or its suite's (TG-8).
            - ``scale_f_log`` (True): default targets log-linear, else linear; for
              the continuum ECDF also the scale of the quality.
            - ``maximization`` (False): larger values are better.
            - ``pairs`` (``"common"``): average over the pairs every group has, or
              ``"all"``: over each group's own pairs. Both warn, naming the
              missing pairs, when the groups' pairs differ (EC-2).
            - ``divide_by_dimension`` (False): x is evaluations / dimension.
            - ``stopped`` (``"restart"``): simulated restarts (EC-7), or
              ``"carry_forward"``: none, a run keeps its final value (IOHanalyzer).
              The EAF (``get_eaf``) carries forward by default, so the AOCC is
              the volume under it only with the same ``stopped`` on both sides.
            - ``fval_var`` (``"raw_y"``).

    Returns:
        pl.DataFrame: ``by``, ``eval_var`` (float) and ``ecdf``; with restarts also
            ``n_restarted``, the runs of the group that stopped before the
            largest budget computed (EC-7).
    """
    data, by = _one_group(data, by)
    x_values = None if x_values is None else list(x_values)
    horizon = max(x_values) if x_values else None
    steps, _, counts = _increments(
        data,
        targets,
        by,
        eval_var=eval_var,
        horizon=horizon,
        samples=x_values,
        **options,
    )
    ecdf = _step_function(steps, by, x_values).rename({"x": eval_var})
    return _with_counts(ecdf, counts, by)


@upgrade_hints
def get_ecdf_auc(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    targets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    b_lo: float = 1.0,
    b_hi: float | None = None,
    scale_eval_log: bool = False,
    **options,
) -> pl.DataFrame:
    """The normalised area under the ECDF of each group of ``by`` over
    [``b_lo``, ``b_hi``] (EC-4): the exact integral of the ECDF divided by
    b_hi - b_lo, on a linear axis or, with ``scale_eval_log``, on log10 of the
    budget. It lies in [0, 1]. Simulated restarts (EC-7, the default) continue
    the runs up to b_hi.

    With ``continuous=True`` this is the AOCC (EC-5): the area under the continuum
    ECDF, which is the mean over runs of the area under each run's normalised
    quality. The AUC of the target ECDF converges to it as the number of targets,
    log-spaced between the same bounds, grows (``docs/definitions/ecdf.md``,
    "Equivalence").

    Args:
        data, targets, by: as in ``get_ecdf``.
        b_lo: the lower budget; 1 by default.
        b_hi: the upper budget; by default the largest budget in ``data`` (divided
            by the dimension with ``divide_by_dimension``).
        scale_eval_log: integrate over log10 of the budget.
        **options: the options of ``get_ecdf``, ``eval_var`` included.

    Returns:
        pl.DataFrame: ``by`` and ``auc``; with restarts also ``n_restarted``,
            the runs of the group that stopped before b_hi.
    """
    data, by = _one_group(data, by)
    area = (b_lo, scale_eval_log)
    steps, x_max, counts = _increments(
        data, targets, by, horizon=b_hi, area=area, **options
    )
    hi = x_max if b_hi is None else b_hi
    auc = _area(steps, by, *_axis(pl.col("x"), b_lo, hi, scale_eval_log))
    return _with_counts(auc, counts, by)


def _value_increments(  # pylint: disable=R0913,R0914
    data: pl.DataFrame,
    budgets: pl.DataFrame | None,
    by: list[str],
    *,
    maximization: bool = False,
    pairs: str = "common",
    stopped: str = "restart",
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """The weighted fixed-budget ECDF increments ``inc`` per group of ``by`` along
    x = -level (the value, its negative when maximising, so that the ECDF
    accumulates along x), and a 0 at x = -inf; with restarts (EC-7) also
    ``n_restarted`` per group of ``by``.

    Each row at which a run improves is a piece: the levels (lo, hi] that the run
    reaches at its evaluation T, cut at the final levels of its run set with
    restarts. Over x in [-hi, -lo) the run adds (1/m) sum_b E(b - T) (EC-9 by the
    duality, EC-7), which is the number of budgets b >= T without restarts."""
    # #lizard forgives  (the options of the fixed-budget ECDF, passed on as one)
    restart = restarts(stopped)
    if budgets is None:
        budgets = get_ecdf_budgets(data, eval_var=eval_var)
    budgets = budgets.cast({c: data.schema[c] for c in PAIR}).unique()
    data = data.join(budgets.select(PAIR), on=PAIR, how="semi", maintain_order="left")
    data = _weights(data, by, pairs)
    sets = list(dict.fromkeys([*by, *PAIR]))
    level = pl.col(fval_var).cast(pl.Float64) * (1.0 if maximization else -1.0)
    largest = budgets.group_by(PAIR).agg(m=pl.len(), last=pl.col("budget").max())
    previous = pl.col("hi").shift(1, fill_value=-math.inf).over("run_id")
    pieces = (
        data.with_columns(hi=level.cum_max().over("run_id"))
        .with_columns(lo=previous)
        .filter(pl.col("hi") > pl.col("lo"))
        .join(largest, on=PAIR)
        .filter(pl.col(eval_var) <= pl.col("last"))  # later ones reach no budget
    )
    counts, gain = None, pl.lit(0.0)
    if restart:
        runs = run_sets(data, sets, eval_var, level.max())
        pieces = split(runs, pieces)
        counts = _stopped_before(data, by, eval_var, pl.col("last"), largest)
        lists = pieces.select("run_set", *PAIR).unique()
        lists = lists.join(budgets.group_by(PAIR).agg("budget"), on=PAIR)
        budget = {r: np.array(b) for r, b in lists.select("run_set", "budget").rows()}
        gain = pl.Series(gained(pieces, runs, budget, eval_var).sum(axis=1))
    pieces = pieces.with_columns(gain=gain).with_row_index("_piece")
    reached = (
        pieces.select("_piece", *PAIR, eval_var)
        .join(budgets, on=PAIR)
        .group_by("_piece")
        .agg(reached=(pl.col("budget") >= pl.col(eval_var)).sum())
    )
    pieces = pieces.join(reached, on="_piece").with_columns(
        reached=pl.col("reached") + pl.col("gain")
    )
    change = pl.col("reached") - pl.col("reached").shift(1, fill_value=0).over("run_id")
    steps = pieces.sort("run_id", "hi", descending=[False, True]).select(
        *by, x=-pl.col("hi"), inc=change * pl.col("weight") / pl.col("m")
    )
    zero = data.select(by).unique().with_columns(x=-math.inf, inc=pl.lit(0.0))
    return pl.concat([zero, steps]), counts


def _with_counts(result: pl.DataFrame, counts: pl.DataFrame | None, by: list[str]):
    """``result`` with each group's ``n_restarted`` from ``counts``, if any."""
    if counts is None:
        return result
    if by == [ALL]:
        n = counts["n_restarted"]
        return result.with_columns(n_restarted=pl.lit(n.item(), dtype=n.dtype))
    return result.join(
        counts, on=by, how="left", maintain_order="left", nulls_equal=True
    )


@upgrade_hints
def get_fixed_budget_ecdf(
    data: pl.DataFrame,
    budgets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    x_values: Iterable[float] | None = None,
    fval_var: str = "raw_y",
    **options,
) -> pl.DataFrame:
    """The fixed-budget ECDF of each group of ``by`` (``docs/definitions/ecdf.md``,
    EC-9).

    At a value v, the fraction of (run, budget) pairs whose best-so-far value at
    the budget is v or better: V_i(b) <= v (>= when maximising), so the run reached
    v within the budget. Before its first logged row a run has no value and counts
    as not reached. A run that stopped before the budget (B_i < b, B_i its
    ``evals_used`` or else its last logged evaluation) is continued by simulated
    restarts by default (EC-7), drawn from the runs of its group in its pair, as
    in ``get_ecdf``: the ECDF is their expected value up to the pair's largest
    budget. Averaged over the (function, dimension) pairs of a group as
    ``get_ecdf`` (EC-2). Exact: without ``x_values`` the result
    has one row per value at which the ECDF steps, from the worst value to the
    best, each with the ECDF from there to the next row, and a first row at
    v = -inf (inf when maximising) with value 0 for every group.

    Args:
        data: the runs' logged rows in evaluation order, as for ``get_ecdf``.
        budgets: budgets per pair, columns ``function_id``, ``dimension`` and
            ``budget``. Defaults to 10 log-spaced budgets per pair
            (``targets.get_ecdf_budgets``).
        by: the columns that make a group, as for ``get_ecdf``.
        x_values: values to sample the ECDF at, for display.
        fval_var: the function value column.
        **options: keyword-only: ``maximization`` (False), ``pairs``
            (``"common"``) and ``eval_var`` (``"evaluations"``), as for
            ``get_ecdf``; ``stopped``: ``"restart"`` (default) or
            ``"carry_forward"``, the run's final value (IOHanalyzer's rule).

    Returns:
        pl.DataFrame: ``by``, ``fval_var`` (float) and ``ecdf``; with restarts
            also ``n_restarted``, the runs of the group that stopped before
            their pair's largest budget.
    """
    data, by = _one_group(data, by)
    sign = -1.0 if options.get("maximization") else 1.0
    steps, counts = _value_increments(data, budgets, by, fval_var=fval_var, **options)
    xs = None if x_values is None else [sign * v for v in x_values]
    ecdf = _step_function(steps, by, xs).with_columns(pl.col("x") * sign)
    return _with_counts(ecdf.rename({"x": fval_var}), counts, by)


@upgrade_hints
def get_fixed_budget_ecdf_auc(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    budgets: pl.DataFrame | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    f_lo: float | None = None,
    f_hi: float | None = None,
    scale_f_log: bool = True,
    fval_var: str = "raw_y",
    **options,
) -> pl.DataFrame:
    """The normalised area under the fixed-budget ECDF of each group of ``by`` over
    the values [``f_lo``, ``f_hi``] (EC-10): its exact integral divided by
    f_hi - f_lo, on log10 of the value (``scale_f_log``, default) or a linear axis.
    It lies in [0, 1]; larger is better whether minimising or maximising.

    It equals the mean, over the budgets, of the continuum ECDF
    (``get_ecdf(continuous=True)``) at those budgets with the bounds f_lo and f_hi
    for every pair.

    Args:
        data, budgets, by: as in ``get_fixed_budget_ecdf``.
        f_lo, f_hi: the value range. By default the smallest and largest bound of
            the pairs in ``data``, each pair's bounds being those of its default
            targets (``targets.get_ecdf_targets``): its suite's (TG-8) or its
            observed range.
        scale_f_log: integrate over log10 of the value.
        fval_var: the function value column.
        **options: the options of ``get_fixed_budget_ecdf``, simulated
            restarts by default.

    Returns:
        pl.DataFrame: ``by`` and ``auc``; with restarts also ``n_restarted``.
    """
    if f_lo is None or f_hi is None:
        kind = "log-linear" if scale_f_log else "linear"
        bounds = get_ecdf_targets(data, kind, n=2, fval_var=fval_var)
        bounds = bounds["target"].to_numpy()
        f_lo = float(bounds.min()) if f_lo is None else f_lo
        f_hi = float(bounds.max()) if f_hi is None else f_hi
    data, by = _one_group(data, by)
    steps, counts = _value_increments(data, budgets, by, fval_var=fval_var, **options)
    if not options.get("maximization"):
        auc = _area(steps, by, *_axis(pl.col("x"), f_lo, f_hi, scale_f_log))
    else:
        x, lo, hi = _axis(-pl.col("x"), f_lo, f_hi, scale_f_log)
        auc = _area(steps, by, -x, -hi, -lo)  # mirrored, so x keeps its order
    return _with_counts(auc, counts, by)


def get_rank_table(
    summary: pl.DataFrame,
    value: str = "ERT",
    *,
    higher_is_better: bool = False,
) -> pl.DataFrame:
    """``summary`` with the ``rank`` of each row's ``value`` within its pair (EC-8).

    For a multi-function rank table, ``summary`` has one row per group and pair:
    the ERT at the pair's target (lower is better) or the mean value at the pair's
    budget (``higher_is_better`` when maximising); ``targets.get_target_dt`` gives
    the default target or budget per pair. Ties get the average rank, ``inf``
    ties at the bottom, and missing values (null or NaN) rank after it, as ties.
    """
    v = pl.col(value).cast(pl.Float64).fill_nan(None)
    ranked = (-v if higher_is_better else v).rank("average")
    missing = v.count() + (v.null_count() + 1) / 2
    rank = pl.when(v.is_null()).then(missing).otherwise(ranked)
    return summary.with_columns(rank.over(PAIR).alias("rank"))


def _ecdf_on_grid(  # pylint: disable=R0913,R0914,R0917
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    by: Sequence[str] = ("algorithm_name",),
    f_min: float | None = None,
    f_max: float | None = None,
    scale_f_log: bool = True,
    eval_values: Iterable[int] | None = None,
    eval_min: int | None = None,
    eval_max: int | None = None,
    scale_eval_log: bool = True,
    maximization: bool = False,
) -> pl.DataFrame:
    """``get_data_ecdf`` without its warning: the options of 0.0.8 on
    ``get_ecdf(continuous=True)``, in the 0.0.8 columns (``plot_ecdf``)."""
    # #lizard forgives  (the public signature of 0.0.8, mapped onto get_ecdf)
    data = data.with_columns(pl.lit(0).alias(c) for c in PAIR if c not in data)
    if eval_values is None:
        evals = data[eval_var]
        eval_values = seq_rt(
            evals, 50, log=scale_eval_log, lower=eval_min, upper=eval_max
        )
    targets = None  # the bounds: f_min and f_max, else each pair's observed range
    if f_min is not None or f_max is not None:
        y = pl.col(fval_var)
        lo = y.min() if f_min is None else pl.lit(float(f_min))
        hi = y.max() if f_max is None else pl.lit(float(f_max))
        targets = data.group_by(PAIR).agg(target=pl.concat_list(lo, hi))
        targets = targets.explode("target")
    ecdf = get_ecdf(
        data,
        targets,
        by,
        x_values=eval_values,
        eval_var=eval_var,
        continuous=True,
        scale_f_log=scale_f_log,
        maximization=maximization,
        stopped="carry_forward",
        fval_var=fval_var,
    )
    eaf = ecdf.select(pl.col(eval_var).cast(pl.Int64), *by, eaf="ecdf")
    return eaf.sort(eval_var, maintain_order=True)


@upgrade_hints
def get_data_ecdf(  # pylint: disable=R0913,R0917,W0613
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    by: Sequence[str] = ("algorithm_name",),
    f_min: float | None = None,
    f_max: float | None = None,
    scale_f_log: bool = True,
    eval_values: Iterable[int] | None = None,
    eval_min: int | None = None,
    eval_max: int | None = None,
    scale_eval_log: bool = True,
    maximization: bool = False,
) -> pl.DataFrame:
    """Deprecated in 0.2.0: use ``get_ecdf(data, continuous=True, x_values=...)``.

    The continuum ECDF (EC-5, EC-6) at the budgets ``eval_values`` (by default 50
    from ``eval_min`` to ``eval_max``, log-spaced with ``scale_eval_log``), between
    the bounds ``f_min`` and ``f_max`` (log10 with ``scale_f_log``); a bound not
    given is the observed one of each (function, dimension) pair, or its suite's
    when neither is given (TG-8). Returns
    ``eval_var``, ``by`` and ``eaf`` as in 0.0.8.

    Since 0.2.0 this is ``get_ecdf``, so it differs from 0.0.8 where 0.0.8 differed
    from the definitions (``docs/definitions/ecdf.md``): the default bounds are per
    pair (not shared by all functions; a log range starts at the smallest positive
    value, TG-3), each pair has equal weight and only the pairs every group has are
    used (EC-2), and the means of the other columns are gone. Data without
    ``function_id`` and ``dimension`` is one pair, as before. As in 0.0.8 there are
    no simulated restarts (EC-7: ``stopped="carry_forward"``).
    """
    # #lizard forgives  (the 0.0.x signature, kept while deprecated)
    arguments = locals()
    warnings.warn(
        "get_data_ecdf is deprecated since 0.2.0; use get_ecdf(data, "
        "continuous=True, x_values=...)",
        DeprecationWarning,
        stacklevel=3,
    )
    return _ecdf_on_grid(**arguments)
