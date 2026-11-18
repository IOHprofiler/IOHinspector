"""Empirical attainment functions (``docs/definitions/eaf.md``, EA-1 to EA-5):
the single-objective EAF and EAF differences of the groups of ``by``, precise or
discretised, and the multi-objective EAF of 0.0.8."""

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import polars as pl
from moocore import eaf, eafdiff

from .._upgrade import upgrade_hints
from ..align import align_data
from ..targets import PAIR, get_ecdf_targets, seq_fv, seq_rt, spaced
from .ecdf import ALL, _one_group, _quality
from .utils import transform_fval

#: IOHanalyzer's 11 attainment levels, in percent (EA-1).
PERCENTILES = tuple(range(0, 101, 10))
#: The value column of an EAF over several (function, dimension) pairs (EA-3).
QUALITY = "quality"
#: The keyword-only options of ``get_eaf`` and ``get_eaf_difference``.
OPTIONS = (
    "eval_values",
    "f_values",
    "scale_eval_log",
    "scale_f_log",
    "targets",
    "maximization",
    "fval_var",
    "eval_var",
)
#: The columns of an EAF difference rectangle (``moocore.eafdiff``).
RECTANGLE = ["x_min", "y_min", "x_max", "y_max", "eaf_diff"]


@upgrade_hints
def get_discritized_eaf_single_objective(
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    eval_values=None,
    eval_min=None,
    eval_max=None,
    eval_targets=10,
    scale_eval_log: bool = True,
    f_min=1e-8,
    f_max=1e2,
    scale_f_log: bool = True,
    f_targets=101,
    maximization: bool = False,
) -> pl.DataFrame:
    """Generate discretized EAF data for single-objective optimization problems.

    Args:
        data (pl.DataFrame): The data object containing optimization trajectory data.
        fval_var (str, optional): Which column contains the function values. Defaults to "raw_y".
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        eval_values (array-like, optional): Specific evaluation values to use. If None, generated from eval_min/max.
        eval_min (int, optional): Minimum evaluation value. If None, uses minimum from data.
        eval_max (int, optional): Maximum evaluation value. If None, uses maximum from data.
        eval_targets (int, optional): Number of evaluation targets to generate. Defaults to 10.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        f_min (float, optional): Minimum function value for scaling. Defaults to 1e-8.
        f_max (float, optional): Maximum function value for scaling. Defaults to 1e2.
        scale_f_log (bool, optional): Whether to use logarithmic scaling for function values. Defaults to True.
        f_targets (int, optional): Number of function value targets to generate. Defaults to 101.
        maximization (bool, optional): Whether larger function values are better.

    Returns:
        pl.DataFrame: The discretized EAF: an ``eaf_target`` column and one column per
            evaluation value (named by its value), sorted by ``eaf_target``.
    """

    if eval_values is None:
        eval_values = seq_rt(
            data[eval_var],
            eval_targets,
            log=scale_eval_log,
            lower=eval_min,
            upper=eval_max,
        )

    dt_aligned = align_data(
        data,
        eval_values,
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
        eval_var=eval_var,
    )
    dt_aligned = transform_fval(
        dt_aligned,
        lb=f_min,
        ub=f_max,
        scale_log=scale_f_log,
        maximization=maximization,
        fval_var=fval_var,
    )
    targets = pl.DataFrame({"eaf_target": np.linspace(0, 1, f_targets)})
    merged = targets.join(dt_aligned.select(eval_var, "eaf"), how="cross")
    merged = merged.with_columns(ps=pl.col("eaf_target") <= pl.col("eaf"))
    return (
        merged.sort(eval_var)
        .pivot(on=eval_var, index="eaf_target", values="ps", aggregate_function="mean")
        .sort("eaf_target")
    )


@upgrade_hints
def get_eaf_data(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    eval_min: int = None,
    eval_max: int = None,
    scale_eval_log: bool = True,
    maximization: bool = False,
) -> pl.DataFrame:
    """Generate aligned EAF data for visualization and analysis.

    Args:
        data (pl.DataFrame): The data object containing optimization trajectory data.
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        eval_min (int, optional): Minimum evaluation value. If None, uses minimum from data.
        eval_max (int, optional): Maximum evaluation value. If None, uses maximum from data.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        maximization (bool, optional): Whether larger function values are better.

    Returns:
        pl.DataFrame: A DataFrame with aligned EAF data; null before a run's first
            logged evaluation (AL-5).
    """
    evals = seq_rt(
        data[eval_var], 50, log=scale_eval_log, lower=eval_min, upper=eval_max
    )
    return align_data(
        data, evals, x_col=eval_var, maximization=maximization, eval_var=eval_var
    )


def _points(data: pl.DataFrame, obj1_var: str, obj2_var: str) -> np.ndarray:
    """The objective values and a run number (1, 2, ...) per row, for moocore, with
    the rows of each run together: ``moocore.eaf`` reads a set from consecutive rows."""
    run = pl.col("run_id").rank("dense").alias("run")
    data = data.sort("run_id", maintain_order=True)
    return data.select(obj1_var, obj2_var, run).to_numpy()


def _eafdiff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``moocore.eafdiff(x, y, rectangles=True)``; no rows where the EAFs are equal
    everywhere (moocore fails on that)."""
    try:
        return eafdiff(x, y, rectangles=True)
    except ValueError as error:
        if "size 0" not in str(error):
            raise
        return np.empty((0, 5))


@upgrade_hints
def get_eaf_pareto_data(
    data: pl.DataFrame,
    obj1_var: str,
    obj2_var: str,
) -> pl.DataFrame:
    """Generate EAF data for multi-objective optimization problems using Pareto fronts.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization data.
        obj1_var (str): Name of the column containing first objective values.
        obj2_var (str): Name of the column containing second objective values.

    Returns:
        pl.DataFrame: A DataFrame with EAF data including objective values and EAF
            percentiles.
    """
    points = _points(data, obj1_var, obj2_var)
    eaf_data = eaf(points[:, :-1], points[:, -1])
    eaf_data_df = pl.DataFrame(
        eaf_data, schema=[obj1_var, obj2_var, "eaf"], orient="row"
    )
    # scale EAF values from percentages to proportions
    return eaf_data_df.with_columns(pl.col("eaf").cast(pl.Float64) / 100.0)


@upgrade_hints
def get_eaf_diff_data(
    data1: pl.DataFrame,
    data2: pl.DataFrame,
    obj1_var: str,
    obj2_var: str,
) -> pl.DataFrame:
    """Calculate EAF difference data between two multi-objective optimization datasets.

    Args:
        data1 (pl.DataFrame): First dataset containing multi-objective optimization data.
        data2 (pl.DataFrame): Second dataset containing multi-objective optimization data.
        obj1_var (str): Name of the column containing first objective values.
        obj2_var (str): Name of the column containing second objective values.

    Returns:
        pl.DataFrame: A DataFrame with EAF difference rectangles and difference values.
    """
    x = _points(data1, obj1_var, obj2_var)
    y = _points(data2, obj1_var, obj2_var)
    return pl.DataFrame(_eafdiff(x, y), schema=RECTANGLE, orient="row")


def _join(left: pl.DataFrame, right: pl.DataFrame, on: list[str]) -> pl.DataFrame:
    if on:
        return left.join(right, on=on, nulls_equal=True)
    return left.join(right, how="cross")


def _each(
    frame: pl.DataFrame, keys: list[str], fn: Callable[[pl.DataFrame], pl.DataFrame]
) -> pl.DataFrame:
    """``fn`` of each group of ``keys`` in ``frame``, with the group's ``keys`` in front."""
    if not keys:
        return fn(frame)
    parts = frame.partition_by(keys, maintain_order=True)
    return pl.concat(
        _join(p.select(keys).head(1), fn(p).drop(keys, strict=False), []) for p in parts
    )


def _traces(  # pylint: disable=R0913
    data: pl.DataFrame,
    by: list[str],
    *,
    targets: pl.DataFrame | None = None,
    scale_f_log: bool = True,
    maximization: bool = False,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
) -> tuple[pl.DataFrame, str, bool, list[str]]:
    """Each run's attainment set (EA-1): its first row and every row where its
    best-so-far value improves, with ``run_id``, ``by``, the pair columns,
    ``eval_var`` and the value.

    The value is f when every group of ``by`` lies in one (function, dimension)
    pair; otherwise it is the normalised quality q of each pair between the pair's
    smallest and largest target (EA-3, the q of the continuum ECDF, EC-5).
    Returns the traces, the value column, whether larger values are better, and
    the columns within which values compare: the pair for f, none for q."""
    keep = list(dict.fromkeys(["run_id", *by, *PAIR, eval_var]))
    one_pair = data.select(pl.struct(PAIR).n_unique().over(by).max()).item() == 1
    if one_pair:
        f = pl.col(fval_var)
        best = (f.cum_max() if maximization else f.cum_min()).over("run_id")
        runs = data.select(*keep, best.alias(fval_var))
        runs = runs.filter(pl.col(fval_var).is_first_distinct().over("run_id"))
        return runs, fval_var, maximization, PAIR
    if targets is None:
        kind = "log-linear" if scale_f_log else "linear"
        targets = get_ecdf_targets(data, kind, fval_var=fval_var)
    targets = targets.cast({c: data.schema[c] for c in PAIR})
    data = data.join(targets.select(PAIR), on=PAIR, how="semi", maintain_order="left")
    # q changes only where it differs from 0 before; each run starts at q = 0
    first = data.unique("run_id", keep="first", maintain_order=True)
    steps = _quality(data, targets, maximization, scale_f_log, fval_var)
    runs = pl.concat(
        [
            first.select(*keep, pl.lit(0.0).alias(QUALITY)),
            steps.select(*keep, pl.col("q").alias(QUALITY)),
        ]
    )
    return runs, QUALITY, True, []


def _sets(
    runs: pl.DataFrame, value: str, larger: bool, eval_var: str, copies: int = 1
) -> np.ndarray:
    """The traces for moocore, which minimises: rows (evaluations, value, set),
    the value negated when larger is better, the sets numbered 1..n and each run's
    rows together, all of it ``copies`` times (sets n+1..2n, ...)."""
    sign = -1.0 if larger else 1.0
    runs = runs.sort("run_id", maintain_order=True)
    points = runs.select(
        pl.col(eval_var).cast(pl.Float64),
        pl.col(value).cast(pl.Float64) * sign,
        pl.col("run_id").rank("dense").cast(pl.Float64),
    ).to_numpy()
    n = points[:, 2].max()
    return np.vstack([points + [0.0, 0.0, k * n] for k in range(copies)])


def _surfaces(runs, value, larger, eval_var, percentiles) -> pl.DataFrame:
    """The attainment surfaces of one group at ``percentiles`` (EA-1, precise)."""
    sets = _sets(runs, value, larger, eval_var)
    surfaces = eaf(sets[:, :2], sets[:, 2], percentiles=list(percentiles))
    surfaces[:, 1] *= -1.0 if larger else 1.0
    return pl.DataFrame(surfaces, schema=[eval_var, value, "percentile"], orient="row")


def _rectangles(runs, by, value, larger, eval_var) -> pl.DataFrame:
    """The rectangles where the EAF difference of the two groups of ``by`` in
    ``runs`` is constant (EA-2, precise), once for each group: its EAF minus the
    other's. moocore compares equally many runs, so each group's runs are repeated
    up to the least common multiple of the two run counts."""
    groups = runs.sort(by, nulls_last=True).partition_by(by, maintain_order=True)
    if len(groups) != 2:
        raise ValueError(
            f"the precise EAF difference compares two groups of by, not {len(groups)};"
            " use mode='discretised' to compare each with the best of the others"
        )
    n = [g["run_id"].n_unique() for g in groups]
    m = math.lcm(*n)
    x, y = (_sets(g, value, larger, eval_var, m // k) for g, k in zip(groups, n))
    rect = _eafdiff(x, y)
    rect[:, 4] /= m  # moocore counts the difference in runs of m
    if larger:
        rect[:, [1, 3]] *= -1.0
    rect = pl.DataFrame(rect, schema=RECTANGLE, orient="row").with_columns(
        y_min=pl.min_horizontal("y_min", "y_max"),
        y_max=pl.max_horizontal("y_min", "y_max"),
    )
    other = rect.with_columns(-pl.col("eaf_diff"))
    return pl.concat(
        _join(g.select(by).head(1), r, []) for g, r in zip(groups, [rect, other])
    )


def _grid(  # pylint: disable=R0913
    data: pl.DataFrame,
    scope: list[str],
    value: str,
    *,
    eval_values: int | Iterable[float] = 50,
    f_values: int | Iterable[float] = 50,
    scale_eval_log: bool = True,
    scale_f_log: bool = True,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
) -> pl.DataFrame:
    """The budgets x values of the discretised EAF in each ``scope`` (EA-5): given,
    or that many log-spaced (linear without ``scale_*_log``) over the scope's
    logged evaluations and values; for the quality q, linear in [0, 1]."""

    def grid(part: pl.DataFrame) -> pl.DataFrame:
        b, v = eval_values, f_values
        if isinstance(b, int):
            b = seq_rt(part[eval_var], b, log=scale_eval_log)
        if isinstance(v, int) and value == QUALITY:
            v = spaced(0.0, 1.0, v)
        elif isinstance(v, int):
            v = seq_fv(part[fval_var], v, log=scale_f_log)
        budgets = pl.Series(eval_var, list(b)).cast(data.schema[eval_var])
        values = pl.Series(value, list(v), pl.Float64)
        return budgets.to_frame().join(values.to_frame(), how="cross")

    return _each(data, scope, grid)


def _attainment(runs, grid, by, scope, value, larger, eval_var) -> pl.DataFrame:
    """alpha(b, y) of each group of ``by`` on the ``grid`` of its scope: the
    fraction of its runs whose best-so-far value at budget b is y or better;
    before its first row a run has attained nothing (AL-5). With the scope columns."""
    keys = list(dict.fromkeys([*by, *scope]))
    budgets = grid.select(*scope, eval_var).unique()
    at = _join(runs.select("run_id", *keys).unique("run_id"), budgets, scope)
    trace = runs.select("run_id", eval_var, pl.col(value).alias("_v"))
    at = at.sort(eval_var).join_asof(
        trace.sort(eval_var, maintain_order=True),  # the last of a tie: the best
        on=eval_var,
        by="run_id",
        check_sortedness=False,
    )
    v, y = pl.col("_v"), pl.col(value)
    hit = (v >= y) if larger else (v <= y)
    cells = _join(at, grid, [*scope, eval_var])
    alpha = cells.group_by([*keys, eval_var, value]).agg(
        eaf=hit.fill_null(False).mean()
    )
    return alpha.sort([*keys, eval_var, value], nulls_last=True)


def _prepare(data: pl.DataFrame, by: Sequence[str], mode: str, options: dict):
    """``data`` with the pair columns (one pair when absent), ``by`` as a list, the
    traces of ``_traces`` and the options of ``_grid``."""
    if mode not in ("precise", "discretised"):
        raise ValueError(f"mode {mode!r}: precise or discretised")
    unknown = set(options) - set(OPTIONS)
    if unknown:
        raise TypeError(f"unknown options {sorted(unknown)}; the options are {OPTIONS}")
    data = data.with_columns(pl.lit(0).alias(c) for c in PAIR if c not in data.columns)
    data, by = _one_group(data, by)
    grid = ("eval_values", "f_values", "scale_eval_log")
    traced = _traces(data, by, **{k: v for k, v in options.items() if k not in grid})
    grid_options = {
        k: v for k, v in options.items() if k not in ("targets", "maximization")
    }
    return data, by, traced, grid_options


def _finish(result: pl.DataFrame, data: pl.DataFrame, eval_var: str) -> pl.DataFrame:
    result = result.with_columns(pl.col(eval_var).cast(data.schema[eval_var]))
    return result.drop(ALL, strict=False)


@upgrade_hints
def get_eaf(
    data: pl.DataFrame,
    by: Sequence[str] = ("algorithm_name",),
    *,
    mode: str = "precise",
    percentiles: Iterable[float] = PERCENTILES,
    **options,
) -> pl.DataFrame:
    """The single-objective EAF of each group of ``by`` (``docs/definitions/eaf.md``).

    alpha(b, y) is the fraction of the group's runs whose best-so-far value within b
    evaluations is y or better (EA-1): (evaluations, f) as two objectives, each
    run's best-so-far trace its attainment set.

    Args:
        data: the runs' logged rows in evaluation order (``DataManager.load``): at
            least ``run_id``, ``by``, ``eval_var`` and ``fval_var``, and
            ``function_id`` and ``dimension`` when there are several pairs.
        by: the columns that make a group (D17). When a group spans several
            (function, dimension) pairs, f is replaced by the normalised quality q
            of each pair (EA-3) in a column ``quality``: 0 at the pair's largest
            target and 1 at its smallest (the reverse when maximising), log10 with
            ``scale_f_log``, so larger is better, as for the continuum ECDF.
        mode: ``"precise"``: the attainment surfaces at ``percentiles``, exact at
            every budget (``moocore.eaf``). ``"discretised"``: alpha on a grid of
            budgets x values (EA-5).
        percentiles: the levels of the precise surfaces, in percent; by default
            IOHanalyzer's 0, 10, ..., 100. The surface at p is attained by at least
            ceil(p n / 100) of the n runs, and by at least one (as ``eaf::eafs``).
        **options: keyword-only:

            - ``eval_values`` (50): the budgets of the discretised EAF, or how many,
              log-spaced over each pair's logged evaluations (IOHanalyzer's grid).
            - ``f_values`` (50): its values, or how many, log-spaced over each
              pair's logged values; for q, linear in [0, 1].
            - ``scale_eval_log`` (True), ``scale_f_log`` (True): log or linear
              spacing of the default grid, and the scale of q.
            - ``targets``: for q, the bounds of each pair: its smallest and
              largest ``target`` (a frame as ``targets.get_ecdf_targets`` gives).
              By default those of ``get_ecdf_targets(data)``, as for the continuum
              ECDF: the observed range of each pair.
            - ``maximization`` (False): larger values are better.
            - ``fval_var`` (``"raw_y"``), ``eval_var`` (``"evaluations"``).

    Returns:
        pl.DataFrame: precise: ``by``, ``eval_var``, the value (``fval_var`` or
            ``quality``) and ``percentile``, the points of each surface;
            discretised: ``by``, ``eval_var``, the value and ``eaf`` (alpha).
    """
    data, by, (runs, value, larger, scope), grid_options = _prepare(
        data, by, mode, options
    )
    eval_var = options.get("eval_var", "evaluations")
    if mode == "precise":
        result = _each(
            runs,
            by,
            lambda g: _surfaces(g, value, larger, eval_var, percentiles),
        )
    else:
        grid = _grid(data, scope, value, **grid_options)
        result = _attainment(runs, grid, by, scope, value, larger, eval_var)
        result = result.select(*by, eval_var, value, "eaf")
    return _finish(result, data, eval_var)


@upgrade_hints
def get_eaf_difference(
    data: pl.DataFrame,
    by: Sequence[str] = ("algorithm_name",),
    *,
    mode: str = "precise",
    **options,
) -> pl.DataFrame:
    """The EAF difference of the groups of ``by`` (``docs/definitions/eaf.md``,
    EA-2): for each group a, Delta_a = alpha_a - max over the other groups c of
    alpha_c, which for two groups A and B is alpha_A - alpha_B (and its negative
    for B). Positive where a attains more often.

    Args:
        data, by: as in ``get_eaf``; the groups are compared within each
            (function, dimension) pair, or on q when a group spans several.
        mode: ``"precise"``: the rectangles where Delta is constant
            (``moocore.eafdiff``, IOHanalyzer's ``EAF_Difference``), for exactly
            two groups. ``"discretised"``: Delta on the grid of budgets x values
            of ``get_eaf`` (IOHanalyzer's ``EAF_diff_Approximate``), for two or
            more groups.
        **options: the options of ``get_eaf``.

    Returns:
        pl.DataFrame: precise: ``by``, ``x_min``, ``y_min``, ``x_max``, ``y_max``
            (evaluations and values) and ``eaf_diff``; discretised: ``by``,
            ``eval_var``, the value (``fval_var`` or ``quality``) and ``eaf_diff``.
    """
    data, by, (runs, value, larger, scope), grid_options = _prepare(
        data, by, mode, options
    )
    eval_var = options.get("eval_var", "evaluations")
    if mode == "precise":
        result = _each(
            runs, scope, lambda s: _rectangles(s, by, value, larger, eval_var)
        )
        return result.select(*by, *RECTANGLE).drop(ALL, strict=False)
    groups = pl.struct(by).n_unique()
    if runs.select((groups.over(scope) if scope else groups).min()).item() < 2:
        raise ValueError("the EAF difference needs at least two groups of by")
    grid = _grid(data, scope, value, **grid_options)
    alpha = _attainment(runs, grid, by, scope, value, larger, eval_var)
    cell = [*scope, eval_var, value]
    best, second = pl.col("eaf").max(), pl.col("eaf").top_k(2).min()
    others = pl.when(pl.col("eaf") < best.over(cell)).then(best.over(cell))
    others = others.otherwise(second.over(cell))
    result = alpha.select(*by, eval_var, value, eaf_diff=pl.col("eaf") - others)
    return _finish(result, data, eval_var)
