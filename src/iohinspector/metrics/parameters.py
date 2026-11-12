"""Parameters, positions and constraint penalties (``docs/definitions/parameters.md``).

The functions take every logged row of the runs, not a best-so-far load:
``DataManager.load(monotonic=False)``. Parameters are any extra trace column, such
as a step size; positions are ``x0`` ... (``positions.parquet``, joined by ``load``).
"""

import re
from collections.abc import Callable, Iterable, Sequence

import polars as pl

from ..align import align_data
from ..targets import seq_fv, seq_rt
from .utils import describe


def get_parameter_sample(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    parameters: Sequence[str],
    x_values: Iterable[float] | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    target: bool = False,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    maximization: bool = False,
) -> pl.DataFrame:
    """Each run's parameter values at budgets, or at targets (PA-1).

    At budget b: the value on the last logged row with evaluations <= b, carried
    forward after the run stopped; null before the run's first row. At target t
    (``target=True``): the value on the first row that reaches t (f <= t, or f >= t
    under maximisation); null when the run never reaches it. Both are lookups on
    the logged rows (``align_data``); no alignment matrix is built. The runtime
    and value samples are the lookups of ``eval_var`` at targets (the hitting
    time, ``inf`` when never reached) and of ``fval_var`` at budgets (the
    best-so-far value).

    Args:
        data: every logged row, with ``run_id``, ``eval_var``, ``fval_var``, ``by``
            and the ``parameters``.
        parameters: the columns to look up.
        x_values: the budgets, or the targets; when None 50 log-spaced ones over
            the data (``seq_rt``, ``seq_fv``).
        by: the columns that make a run set, kept in the result.
        target: look up at targets instead of budgets.
        eval_var: the column of evaluation numbers.
        fval_var: the column of function values.
        maximization: whether larger function values are better.

    Returns:
        pl.DataFrame: one row per run and budget (``eval_var``) or target
            (``fval_var``): ``run_id``, ``by``, the budget or target and the
            ``parameters``.
    """
    keys = list(dict.fromkeys(["run_id", *by]))
    data = data.select(*dict.fromkeys([*keys, eval_var, fval_var, *parameters]))
    x_col, y_col = (fval_var, eval_var) if target else (eval_var, fval_var)
    if x_values is None:
        x_values = (seq_fv if target else seq_rt)(data[x_col], 50, True)
    aligned = align_data(
        data,
        x_values,
        group_cols=keys,
        x_col=x_col,
        y_col=y_col,
        maximization=maximization,
        eval_var=eval_var,
    )
    sample = aligned.select(*keys, x_col, *parameters)
    return sample.sort(*keys[1:], "run_id", x_col)  # by first


def get_parameter_summary(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    parameters: Sequence[str],
    x_values: Iterable[float] | None = None,
    by: Sequence[str] = ("algorithm_name",),
    *,
    target: bool = False,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    maximization: bool = False,
) -> pl.DataFrame:
    """Summary of each parameter per run set (a group of ``by``) at budgets or
    targets, over the runs with a defined value (``get_parameter_sample``; PA-2).
    The arguments are those of ``get_parameter_sample``.

    Returns:
        pl.DataFrame: per run set, budget (``eval_var``) or target (``fval_var``) and
            ``parameter``: ``n_runs``, ``n_defined`` (the runs with a value),
            ``mean``, ``median``, ``sd`` (denominator n - 1) and the type 7
            quantiles ``q2`` ... ``q98``.
    """
    sample = get_parameter_sample(
        data,
        parameters,
        x_values,
        by,
        target=target,
        eval_var=eval_var,
        fval_var=fval_var,
        maximization=maximization,
    )
    x_col = fval_var if target else eval_var
    keys = list(dict.fromkeys([*by, x_col]))
    long = sample.with_columns(pl.col(parameters).cast(pl.Float64)).unpivot(
        parameters,
        index=list(dict.fromkeys(["run_id", *keys])),
        variable_name="parameter",
    )
    summaries = long.group_by(*keys, "parameter").agg(describe(pl.col("value")))
    return summaries.sort(*keys, "parameter")


def get_logged_rows(
    data: pl.DataFrame,
    columns: Sequence[str] | None = None,
    by: Sequence[str] = ("algorithm_name",),
    eval_var: str = "evaluations",
) -> pl.DataFrame:
    """The logged rows of ``columns``, unaligned: the rows where all of them have a
    value. For a correlation view, two columns matched on (run, evaluation), such
    as a parameter and ``raw_y`` (PA-3; no coefficient is computed). For the
    positions view, the default: the positions ``x0`` ... of each run (add
    ``"instance"`` to ``by`` to show them per instance).

    Returns:
        pl.DataFrame: ``run_id``, ``by``, ``eval_var`` and ``columns``, in data order.
    """
    if columns is None:
        columns = [c for c in data.columns if re.fullmatch(r"x\d+", c)]
        columns.sort(key=lambda c: int(c[1:]))
    keys = dict.fromkeys(["run_id", *by, eval_var, *columns])
    return data.select(*keys).drop_nulls(columns)


def apply_penalty(  # pylint: disable=too-many-arguments
    data: pl.DataFrame,
    method: str = "relative",
    *,
    factor: float = 1.0,
    exponent: float = 1.0,
    feasible: str | Callable[[pl.Expr], pl.Expr] = "zero",
    maximization: bool = False,
    violation: str = "violation",
    fval_var: str = "raw_y",
) -> pl.DataFrame:
    """``data`` with each function value f replaced by its penalised value g(f, v)
    (PA-4), so that best-so-far values, hitting times and every metric computed on
    the result include the penalty. Apply it to every logged row
    (``DataManager.load(monotonic=False)``).

    A row is infeasible when ``feasible`` says so: ``"zero"`` (v = 0 is feasible,
    so negative violations are infeasible; IOHexperimenter's rule), ``"nonpositive"``
    (v <= 0 is feasible; IOHanalyzer's rule) or a function of the violation
    expression, such as ``lambda v: v <= capacity`` for load-type violations (PA-6).
    A penalty always makes a value worse: it is added when minimising and
    subtracted when maximising. On infeasible rows, with c = ``factor`` and
    p = ``exponent``:

    - ``"ignore"``: f;
    - ``"relative"``: f + c * |v| ** p;
    - ``"absolute"``: f + c;
    - ``"value"``: c.

    Data whose logged values already contain the penalty (IOHexperimenter's
    ``SOFT``, ``HARD`` and ``OVERRIDE`` constraints) need ``"ignore"``.
    """
    if data[violation].null_count():
        raise ValueError(f"every row needs a {violation!r} value")
    v, f = pl.col(violation), pl.col(fval_var)
    rules = {"zero": v == 0, "nonpositive": v <= 0}
    if not callable(feasible) and feasible not in rules:
        raise ValueError(f"feasible is {sorted(rules)} or a callable, not {feasible!r}")
    sign = -1 if maximization else 1
    methods = {
        "ignore": f,
        "relative": f + sign * factor * v.abs() ** exponent,
        "absolute": f + sign * factor,
        "value": pl.lit(factor, pl.Float64),
    }
    if method not in methods:
        raise ValueError(f"method is one of {sorted(methods)}, not {method!r}")
    ok = feasible(v) if callable(feasible) else rules[feasible]
    penalised = pl.when(ok).then(f).otherwise(methods[method])
    return data.with_columns(penalised.alias(fval_var))
