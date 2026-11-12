"""ECDF, its area (AUC, AOCC) and multi-function rank tables
(``docs/definitions/ecdf.md``, EC-1 to EC-8)."""

import math
import warnings
from collections.abc import Iterable, Sequence

import polars as pl

from .._upgrade import upgrade_hints
from ..align import align_data
from ..targets import PAIR, get_ecdf_targets, seq_rt
from .utils import transform_fval


def _weights(data: pl.DataFrame, by: list[str], pair: list[str], pairs: str):
    """``data`` on the pairs used (EC-2), with ``weight`` 1 / (|P| n): each pair of a
    group of ``by`` has equal weight, and within a pair each run."""
    if pairs not in ("common", "all"):
        raise ValueError(f"pairs {pairs!r}: common or all")
    groups = [c for c in by if c not in pair]
    if groups:
        sets = data.select(*groups, *pair).unique()
        every = (
            sets.select(groups).unique().join(sets.select(pair).unique(), how="cross")
        )
        missing = every.join(sets, on=[*groups, *pair], how="anti", nulls_equal=True)
        if missing.height:
            _warn_pairs(missing.sort(missing.columns), len(groups), pair, pairs)
        if missing.height and pairs == "common":
            drop = missing.select(pair).unique()
            data = data.join(drop, on=pair, how="anti", maintain_order="left")
            if data.is_empty():
                raise ValueError("the groups of by share no pair; use pairs='all'")
    n_runs = pl.col("run_id").n_unique().over(list(dict.fromkeys([*by, *pair])))
    n_pairs = pl.struct(pair).n_unique().over(by)
    return data.with_columns(weight=1 / (n_runs * n_pairs))


def _warn_pairs(missing: pl.DataFrame, n: int, pair: list[str], pairs: str) -> None:
    rows = [f"{', '.join(map(str, r[:n]))} has no {r[n:]}" for r in missing.rows()]
    more = f"; and {len(rows) - 10} more" if len(rows) > 10 else ""
    action = (
        "only the pairs every group has are used (pairs='common')"
        if pairs == "common"
        else "each group is averaged over its own pairs (pairs='all'), "
        "so the curves are hard to compare"
    )
    warnings.warn(
        f"the groups of by do not have the same {tuple(pair)} pairs: {action}. "
        + "; ".join(rows[:10])
        + more,
        stacklevel=6,
    )


def _hits(  # pylint: disable=R0913,R0917
    data, targets, keys, pair, maximization, fval_var, eval_var
):
    """Increments ``dy`` = 1/m of each (run, target) pair at its first hitting time
    (EC-1: inclusive, AL-1; ties to the first row, AL-3)."""
    parts = targets.partition_by(pair, as_dict=True)
    aligned = pl.concat(
        align_data(
            runs,
            parts[key]["target"],
            group_cols=["run_id", *keys],
            x_col=fval_var,
            y_col=eval_var,
            maximization=maximization,
        ).with_columns(dy=1 / pl.len().over("run_id"))
        for key, runs in data.partition_by(pair, as_dict=True).items()
    )
    return aligned.filter(pl.col(eval_var).is_finite())


def _quality(  # pylint: disable=R0913,R0917
    data, targets, pair, maximization, scale_f_log, fval_var
):
    """Increments ``dy`` of each run's normalised quality q (EC-5): the log10 (or
    linear) distance of its best-so-far value from the worse bound, between the
    pair's smallest and largest target. q is the fraction of a continuum of targets
    between the bounds that the run has reached."""
    bounds = targets.group_by(pair).agg(
        lo=pl.col("target").min(), hi=pl.col("target").max()
    )
    best = pl.col(fval_var).cum_max() if maximization else pl.col(fval_var).cum_min()
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
        data.join(bounds, on=pair, maintain_order="left")  # rows in run order
        .with_columns(best=best.over("run_id"))
        .with_columns(q=q)
        .with_columns(dy=pl.col("q") - previous)
        .filter(pl.col("dy") != 0)
    )


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
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    pair: Sequence[str] = PAIR,
) -> tuple[pl.DataFrame, float]:
    """The weighted ECDF increments ``inc`` at ``x`` per group of ``by`` (with a 0 at
    x = 0 for every group), and the largest budget in the selection."""
    # #lizard forgives  (the options of get_ecdf and get_ecdf_auc, passed on as one)
    pair = list(pair)
    if targets is None:
        kind = "log-linear" if scale_f_log else "linear"
        targets = get_ecdf_targets(data, kind, fval_var=fval_var, pair=pair)
    targets = targets.cast({c: data.schema[c] for c in pair})
    used = targets.select(pair).unique()
    data = data.join(used, on=pair, how="semi", maintain_order="left")
    data = _weights(data, by, pair, pairs)
    if continuous:
        steps = _quality(data, targets, pair, maximization, scale_f_log, fval_var)
    else:
        keys = list(dict.fromkeys([*by, *pair, "weight"]))
        steps = _hits(data, targets, keys, pair, maximization, fval_var, eval_var)
    x = pl.col(eval_var).cast(pl.Float64)
    if divide_by_dimension:
        x = x / pl.col("dimension")
    zero = data.select(by).unique().with_columns(x=pl.lit(0.0), inc=pl.lit(0.0))
    steps = steps.select(*by, x=x, inc=pl.col("dy") * pl.col("weight"))
    return pl.concat([zero, steps]), data.select(x.max()).item()


def _cumulate(steps: pl.DataFrame, by: list[str], order: Sequence[str] = ()):
    return steps.sort([*by, "x", *order]).with_columns(
        ecdf=pl.col("inc").cum_sum().over(by)
    )


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
    equal weight and, within a pair, each target and each run (EC-2). It is exact
    (EC-3): without ``x_values`` the result has one row per x at which the ECDF
    steps, with its value from x on, and a row at x = 0 (value 0) for every group.

    Args:
        data: the runs' logged rows in evaluation order (``DataManager.load``): at
            least ``run_id``, the pair columns, ``by``, ``eval_var`` and ``fval_var``.
            Best-so-far values are computed here (AL-6).
        targets: targets per pair, columns ``function_id``, ``dimension`` and
            ``target`` (``targets.get_ecdf_targets``). Defaults to 10 log-linear
            targets per pair over its observed values (TG-1, TG-2); pairs without
            targets are left out.
        by: the columns that make a group (a curve): descriptors or runs columns
            (D17). Pair columns in ``by`` give one curve per pair.
        x_values: budgets to sample the ECDF at, for display (EC-3).
        eval_var: the evaluations column.
        **options: keyword-only:

            - ``continuous`` (False): the continuum ECDF (EC-5, EC-6), the mean
              over runs of the normalised quality of the best-so-far value between
              the pair's smallest and largest target. It is the fraction of
              infinitely many targets between them, log-uniform (uniform without
              ``scale_f_log``), that the run has reached. Pass
              ``get_ecdf_targets(data, "bbob")`` for the BBOB bounds 1e-8 and 1e2.
            - ``scale_f_log`` (True): default targets log-linear, else linear; for
              the continuum ECDF also the scale of the quality.
            - ``maximization`` (False): larger values are better.
            - ``pairs`` (``"common"``): average over the pairs every group has, or
              ``"all"``: over each group's own pairs. Both warn, naming the
              missing pairs, when the groups' pairs differ (EC-2).
            - ``divide_by_dimension`` (False): x is evaluations / dimension.
            - ``fval_var`` (``"raw_y"``) and ``pair`` (``("function_id",
              "dimension")``).

    Returns:
        pl.DataFrame: ``by``, ``eval_var`` (float) and ``ecdf``.
    """
    by = list(by)
    steps, _ = _increments(data, targets, by, eval_var=eval_var, **options)
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
    return steps.select(*by, pl.col("x").alias(eval_var), "ecdf")


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
    budget. It lies in [0, 1].

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
        pl.DataFrame: ``by`` and ``auc``.
    """
    by = list(by)
    steps, x_max = _increments(data, targets, by, **options)
    lo, hi = b_lo, x_max if b_hi is None else b_hi
    if lo >= hi or (scale_eval_log and lo <= 0):
        raise ValueError(f"budget range [{lo}, {hi}] is empty or not positive")
    x = pl.col("x").clip(lo, hi)
    if scale_eval_log:
        x, lo, hi = x.log10(), math.log10(lo), math.log10(hi)
    width = x.shift(-1, fill_value=hi).over(by) - x
    steps = _cumulate(steps, by).with_columns(area=pl.col("ecdf") * width)
    auc = pl.col("area").sum() / (hi - lo)
    return steps.group_by(by, maintain_order=True).agg(auc=auc)


def get_rank_table(
    summary: pl.DataFrame,
    value: str = "ERT",
    *,
    higher_is_better: bool = False,
    pair: Sequence[str] = PAIR,
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
    return summary.with_columns(rank.over(list(pair)).alias("rank"))


@upgrade_hints
def get_data_ecdf(
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    by: Sequence[str] = ("algorithm_name",),
    f_min: int = None,
    f_max: int = None,
    scale_f_log: bool = True,
    eval_values: Iterable[int] = None,
    eval_min: int = None,
    eval_max: int = None,
    scale_eval_log: bool = True,
    maximization: bool = False,
) -> pl.DataFrame:
    """Generate empirical cumulative distribution function (ECDF) data based on EAF calculations.

    Legacy (0.0.8): the continuum ECDF on a budget grid, with bounds taken from the
    aligned data and shared by every function, each run weighted equally. Use
    ``get_ecdf`` (the target ECDF), or ``get_ecdf(..., continuous=True)`` for the
    continuum ECDF with bounds fixed per pair (``docs/definitions/ecdf.md``, EC-6).

    Args:
        data (pl.DataFrame): The DataFrame containing the full performance trajectory data.
        fval_var (str, optional): Which column contains the performance measure values. Defaults to "raw_y".
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        by (Sequence[str], optional): The columns to group by: runs columns or
            descriptors. Defaults to ["algorithm_name"].
        f_min (int, optional): Minimum value for function value scaling. If None, uses minimum from data. Defaults to None.
        f_max (int, optional): Maximum value for function value scaling. If None, uses maximum from data. Defaults to None.
        scale_f_log (bool, optional): Whether to use logarithmic scaling for function values. Defaults to True.
        eval_values (Iterable[int], optional): Specific evaluation values to use. If None, generated from eval_min/max. Defaults to None.
        eval_min (int, optional): Minimum evaluation value. If None, uses minimum from data. Defaults to None.
        eval_max (int, optional): Maximum evaluation value. If None, uses maximum from data. Defaults to None.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        maximization (bool, optional): Whether the performance measure is being maximized. Defaults to False.

    Returns:
        pl.DataFrame: A DataFrame containing the ECDF data with aligned evaluation
            points.
    """
    if eval_values is None:
        eval_values = seq_rt(data[eval_var], 50, scale_eval_log, eval_min, eval_max)
    data_aligned = align_data(
        data.cast({eval_var: pl.Int64}),
        eval_values,
        group_cols=["run_id", *by],
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
        eval_var=eval_var,
    )
    dt_ecdf = transform_fval(
        data_aligned,
        fval_var=fval_var,
        maximization=maximization,
        lb=f_min,
        ub=f_max,
        scale_log=scale_f_log,
    )
    keys = [eval_var, *by]
    # polars 2 raises on mean() of non-numeric columns, polars 1 returned null.
    dt_ecdf = (
        dt_ecdf.group_by(keys)
        .agg(
            [
                pl.col(c).mean()
                if dt.is_numeric() or dt == pl.Boolean
                else pl.lit(None, dtype=dt).alias(c)
                for c, dt in dt_ecdf.schema.items()
                if c not in keys and c != "run_id"
            ]
        )
        .sort(eval_var)
    )
    return dt_ecdf
