import numpy as np
import polars as pl
import warnings
from typing import Iterable, Optional, Union, Dict

from moocore import (
    filter_dominated,
)

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
        cast_to_int (bool, optional): Whether the values should be casted to integers (e.g. in case of budget) or not. Defaults to False.

    Returns:
        np.ndarray: Array of evenly spaced values between min and max.
    """
    transform = lambda x: x
    if scale_log:
        assert min > 0
        min = np.log10(min)
        max = np.log10(max)
        transform = lambda x: 10**x
    if len == 1:
        values =np.array([min])
    else:
        if(max == min):
            values = np.ones(len) * min
        else:
            values = np.arange(
                min,
                max + (max - min) / (2 * (len - 1)),
                (max - min) / (len - 1),
                dtype=float,
            )
            
    values = transform(values)
    if cast_to_int:
        return np.unique(np.array(values, dtype=int))
    return np.unique(values)




def normalize_objectives(
    data: pl.DataFrame,
    obj_vars: Iterable[str] = ["raw_y"],
    bounds: Optional[Dict[str, tuple[Optional[float], Optional[float]]]] = None,
    log_scale: Union[bool, Dict[str, bool]] = False,
    maximize: Union[bool, Dict[str, bool]] = False,
    only_nondominated: bool = False,
    prefix: str = "ert",
    keep_original: bool = True
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
            lb = result[col].min() if ndpoints is None else ndpoints[:,i].min()
        if ub is None:
            ub = result[col].max() if ndpoints is None else ndpoints[:,i].max()
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
            col: (min_obj[col][0] if min_obj is not None else None,
                  max_obj[col][0] if max_obj is not None else None)
            for col in obj_vars
        },
        maximize=True,
        only_nondominated=only_nondominated,
        prefix="obj",
        keep_original=False
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
        pl.DataFrame: The original dataframe with normalized function values in a new 'eaf' column.
    """
    bounds = {fval_var: (lb, ub)}
    res = normalize_objectives(
        data,
        obj_vars=[fval_var],
        bounds=bounds,
        log_scale=scale_log,
        maximize=maximization,
        prefix="eaf"
    )
    return res