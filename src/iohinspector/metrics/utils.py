import string
import warnings
from collections.abc import Callable, Sequence
from typing import Iterable, Optional, Union, Dict

import numpy as np
import polars as pl

from moocore import (
    filter_dominated,
)

from ..targets import seq_rt, spaced


#: The probabilities of the quantile columns ``q2`` ... ``q98``, in percent: the
#: nine of IOHanalyzer (``docs/definitions/runtime.md`` RT-5, ``fixed-budget.md`` FB-2).
QUANTILES = (2, 5, 10, 25, 50, 75, 90, 95, 98)


def describe(v: pl.Expr) -> list[pl.Expr]:
    """``n_runs``, ``n_defined`` (non-null), ``mean``, ``median``, ``sd`` and the
    type 7 quantiles ``q2`` ... ``q98`` of ``v`` per group (FB-2, PA-2)."""
    return [
        pl.len().alias("n_runs"),
        v.count().alias("n_defined"),
        v.mean().alias("mean"),
        v.median().alias("median"),
        v.std().alias("sd"),
        *(v.quantile(pct / 100, "linear").alias(f"q{pct}") for pct in QUANTILES),
    ]


def custom(op: Callable[[pl.Series], float], v: pl.Expr) -> pl.Expr:
    """The statistic ``op`` of ``v`` per group, named after ``op``."""
    return v.map_batches(op, return_dtype=pl.Float64, returns_scalar=True).alias(
        op.__name__
    )


def runs_of(
    data: pl.DataFrame, by: Sequence[str] = (), eval_var: str = "evaluations"
) -> pl.DataFrame:
    """One row per run: ``run_id``, the ``by`` columns, ``run_set`` (an integer key
    of the run's group of ``by``) and B_i as ``evals_used``: the evaluations the run
    used by the run metadata (``DataManager.load(include_columns=["evals_used"])``),
    or else its last logged evaluation, which is less for a source that does not log
    the final evaluation (``docs/definitions/runtime.md``).

    Metrics align and group on ``run_id`` and ``run_set`` and join the ``by``
    columns afterwards (``name_run_sets``): several times faster than on ``by``."""
    by = list(dict.fromkeys(by))
    used = "evals_used" if "evals_used" in data.columns else eval_var
    runs = data.group_by("run_id").agg(
        pl.col([c for c in by if c != "run_id"]).first(),
        pl.col(used).max().cast(pl.Int64).alias("evals_used"),
    )
    run_set = pl.struct(by).rank("dense") if by else pl.lit(1, pl.UInt32)
    return runs.with_columns(run_set=run_set)


def name_run_sets(
    summary: pl.DataFrame, runs: pl.DataFrame, by: Sequence[str], *x: str
) -> pl.DataFrame:
    """``summary``, keyed by ``run_set`` (and ``x``), with the ``by`` columns of
    ``runs`` (``runs_of``) in place of ``run_set``, in front and sorted."""
    keys = list(dict.fromkeys([*by, *x]))
    names = runs.select("run_set", *dict.fromkeys(by)).unique("run_set")
    summary = summary.join(names, on="run_set").drop("run_set")
    return summary.select(*keys, pl.exclude(keys)).sort(keys)


def get_sequence(
    min: float,
    max: float,
    len: float,
    scale_log: bool = False,
    cast_to_int: bool = False,
) -> np.ndarray:
    """Create sequence of points, used for subselecting targets / budgets for alignment and data processing.

    Args:
        min (float): Starting point of the range.
        max (float): Final point of the range.
        len (float): Number of steps in the sequence.
        scale_log (bool, optional): Whether values should be scaled logarithmically. Defaults to False.
        cast_to_int (bool, optional): Whether the values should be rounded to integers
            (e.g. in case of budget, ``targets.seq_rt``) or not. Defaults to False.

    Returns:
        np.ndarray: Array of evenly spaced values between min and max.
    """
    if cast_to_int:
        return seq_rt([min, max], len, log=scale_log)
    return spaced(min, max, len, scale_log)


def group_label(by: Sequence[str], template: str | None = None) -> pl.Expr:
    """The label of each row's group of ``by`` (D17), from ``template``: text with
    ``{name}`` placeholders for ``by`` columns, such as ``"{algorithm_name}
    (sigma {sigma})"``. The default joins the values with ``", "``; a missing value
    reads ``null``. The expression is named ``label``.
    """
    if template is None:
        template = ", ".join(f"{{{name}}}" for name in by)
    parts = []
    for text, name, spec, conversion in string.Formatter().parse(template):
        if spec or conversion:
            raise ValueError(f"label template {template!r}: only {{name}} placeholders")
        parts.append(pl.lit(text))
        if name is not None:
            parts.append(pl.col(name).cast(pl.String).fill_null("null"))
    return pl.concat_str(parts).alias("label")


def normalize_objectives(
    data: pl.DataFrame,
    obj_vars: Iterable[str] = ["raw_y"],
    bounds: Optional[Dict[str, tuple[Optional[float], Optional[float]]]] = None,
    log_scale: Union[bool, Dict[str, bool]] = False,
    maximize: Union[bool, Dict[str, bool]] = False,
    only_nondominated: bool = False,
    prefix: str = "ert",
    keep_original: bool = True,
) -> pl.DataFrame:
    """Normalize multiple objective columns in a dataframe using min-max normalization.

    Args:
        data (pl.DataFrame): Input dataframe containing the objective columns.
        obj_vars (Iterable[str], optional): Which columns contain the objective values to normalize. Defaults to ["raw_y"].
        bounds (Optional[Dict[str, tuple[Optional[float], Optional[float]]]], optional): Optional manual bounds per column as (lower_bound, upper_bound). Defaults to None.
        log_scale (Union[bool, Dict[str, bool]], optional): Whether to apply log10 scaling. Can be a single bool or a dict per column. Defaults to False.
        maximize (Union[bool, Dict[str, bool]], optional): Whether to treat objective as maximization. Can be a single bool or dict per column. Defaults to False.
        only_nondominated (bool, optional): Whether to only consider non-dominated objectives in computing bounds. Defaults to False.
        prefix (str, optional): Prefix for normalized column names. Defaults to "ert".
        keep_original (bool, optional): Whether to keep original objective column names. Defaults to True.

    Returns:
        pl.DataFrame: The original dataframe with new normalized objective columns added.
    """
    result = data.clone()
    n_objectives = len(obj_vars)

    ndpoints = None
    if only_nondominated and len(obj_vars) > 1:
        obj_vals = np.array(result[obj_vars])
        ndpoints = filter_dominated(obj_vals)

    for i, col in enumerate(obj_vars):
        # Determine log scaling
        use_log = log_scale[col] if isinstance(log_scale, dict) else log_scale
        is_max = maximize[col] if isinstance(maximize, dict) else maximize

        # Get bounds
        lb, ub = None, None
        if bounds and col in bounds:
            lb, ub = bounds[col]
        if lb is None:
            lb = result[col].min() if ndpoints is None else ndpoints[:, i].min()
        if ub is None:
            ub = result[col].max() if ndpoints is None else ndpoints[:, i].max()
        # Log scale if needed
        if use_log:
            if lb <= 0:
                warnings.warn(
                    f"Lower bound for column '{col}' <= 0; resetting to 1e-8 for log-scaling."
                )
                lb = 1e-8
            lb, ub = np.log10(lb), np.log10(ub)
            norm_expr = ((pl.col(col).log10() - lb) / (ub - lb)).clip(0, 1)
        else:
            norm_expr = ((pl.col(col) - lb) / (ub - lb)).clip(0, 1)

        # Reverse if minimization
        if not is_max:
            norm_expr = 1 - norm_expr
        # Add normalized column with appropriate name
        if n_objectives > 1:
            if keep_original:
                norm_expr = norm_expr.alias(f"{prefix}_{col}")
            else:
                idx = list(obj_vars).index(col) + 1
                norm_expr = norm_expr.alias(f"{prefix}{idx}")
        else:
            # If only one objective, use the prefix directly
            norm_expr = norm_expr.alias(prefix)
        result = result.with_columns(norm_expr)

    return result


def add_normalized_objectives(
    data: pl.DataFrame,
    obj_vars: Iterable[str],
    max_obj: Optional[pl.DataFrame] = None,
    min_obj: Optional[pl.DataFrame] = None,
    only_nondominated: bool = False,
) -> pl.DataFrame:
    """Add new normalized columns to provided dataframe based on the provided objective columns.

    Args:
        data (pl.DataFrame): The original dataframe containing objective columns.
        obj_vars (Iterable[str]): Which columns contain the objective values to normalize.
        max_obj (Optional[pl.DataFrame], optional): If provided, these values will be used as the maxima instead of the values found in `data`. Defaults to None.
        min_obj (Optional[pl.DataFrame], optional): If provided, these values will be used as the minima instead of the values found in `data`. Defaults to None.
        only_nondominated (bool, optional): Whether to only consider non-dominated points for the normalization bounds. Defaults to False.)
    Returns:
        pl.DataFrame: The original `data` DataFrame with a new column 'objI' added for each objective, for I=1...len(obj_vars).
    """

    return normalize_objectives(
        data,
        obj_vars=obj_vars,
        bounds={
            col: (
                min_obj[col][0] if min_obj is not None else None,
                max_obj[col][0] if max_obj is not None else None,
            )
            for col in obj_vars
        },
        maximize=True,
        only_nondominated=only_nondominated,
        prefix="obj",
        keep_original=False,
    )


def transform_fval(
    data: pl.DataFrame,
    lb: float = 1e-8,
    ub: float = 1e8,
    scale_log: bool = True,
    maximization: bool = False,
    fval_var: str = "raw_y",
) -> pl.DataFrame:
    """Helper function to transform function values using min-max normalization based on provided bounds and scaling.

    Args:
        data (pl.DataFrame): Input dataframe containing function values.
        lb (float, optional): Lower bound for normalization. Defaults to 1e-8.
        ub (float, optional): Upper bound for normalization. Defaults to 1e8.
        scale_log (bool, optional): Whether to apply logarithmic scaling. Defaults to True.
        maximization (bool, optional): Whether the problem is a maximization problem. Defaults to False.
        fval_var (str, optional): Which column contains the function values to transform. Defaults to "raw_y".

    Returns:
        pl.DataFrame: The original dataframe with normalized function values in a new
            'eaf' column. A run with no value yet (null, AL-5) has reached nothing: 0.
    """
    bounds = {fval_var: (lb, ub)}
    res = normalize_objectives(
        data,
        obj_vars=[fval_var],
        bounds=bounds,
        log_scale=scale_log,
        maximize=maximization,
        prefix="eaf",
    )
    return res.with_columns(pl.col("eaf").fill_null(0.0))
