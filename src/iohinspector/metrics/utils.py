import numpy as np
import polars as pl
import warnings
from typing import Iterable, Optional, Union, Dict

def get_sequence(
    min: float,
    max: float,
    len: float,
    scale_log: bool = False,
    cast_to_int: bool = False,
) -> np.ndarray:
    """Create sequence of points, used for subselecting targets / budgets for allignment and data processing

    Args:
        min (float): Starting point of the range
        max (float): Final point of the range
        len (float): Number of steps
        scale_log (bool): Whether values should be scaled logarithmically. Defaults to False
        version (str, optional): Whether the value should be casted to integers (e.g. in case of budget) or not. Defaults to False.

    Returns:
        np.ndarray: Array of evenly spaced values
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
    obj_cols: Iterable[str] = ["raw_y"],
    bounds: Optional[Dict[str, tuple[Optional[float], Optional[float]]]] = None,
    log_scale: Union[bool, Dict[str, bool]] = False,
    maximize: Union[bool, Dict[str, bool]] = False,
    prefix: str = "ert",
    keep_original: bool = True
) -> pl.DataFrame:
    """
    Normalize multiple objective columns in a dataframe.

    Args:
        data (pl.DataFrame): Input dataframe.
        obj_cols (Iterable[str]): Columns to normalize.
        bounds (Optional[Dict[str, tuple(lb, ub)]]): Optional manual bounds per column.
        log_scale (Union[bool, Dict[str, bool]]): Whether to apply log10 scaling. Can be a single bool or a dict per column.
        maximize (Union[bool, Dict[str, bool]]): Whether to treat objective as maximization. Can be a single bool or dict.
        prefix (str): Prefix for normalized column names.
        keep_original (bool): Whether to keep original objective columns names.
    Returns:
        pl.DataFrame: The original dataframe with new normalized objective columns added.
    """
    result = data.clone()
    n_objectives = len(obj_cols)
    for col in obj_cols:
        # Determine log scaling
        use_log = log_scale[col] if isinstance(log_scale, dict) else log_scale
        is_max = maximize[col] if isinstance(maximize, dict) else maximize

        # Get bounds
        lb, ub = None, None
        if bounds and col in bounds:
            lb, ub = bounds[col]
        if lb is None:
            lb = result[col].min()
        if ub is None:
            ub = result[col].max()
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
                idx = list(obj_cols).index(col) + 1
                norm_expr = norm_expr.alias(f"{prefix}{idx}")
        else:
            # If only one objective, use the prefix directly
            norm_expr = norm_expr.alias(prefix)
        result = result.with_columns(norm_expr)

    return result


def add_normalized_objectives(
    data: pl.DataFrame, 
    obj_cols: Iterable[str], 
    max_vals: Optional[pl.DataFrame] = None, 
    min_vals: Optional[pl.DataFrame] = None
):
    """Add new normalized columns to provided dataframe based on the provided objective columns

    Args:
        data (pl.DataFrame): The original dataframe
        obj_cols (Iterable[str]): The names of each objective column
        max_vals (Optional[pl.DataFrame]): If provided, these values will be used as the maxima instead of the values found in `data`
        min_vals (Optional[pl.DataFrame]): If provided, these values will be used as the minima instead of the values found in `data`

    Returns:
        _type_: The original `data` DataFrame with a new column 'objI' added for each objective, for I=1...len(obj_cols)
    """

    return normalize_objectives(
        data,
        obj_cols=obj_cols,
        bounds={
            col: (min_vals[col][0] if min_vals is not None else None,
                  max_vals[col][0] if max_vals is not None else None)
            for col in obj_cols
        },
        maximize=True, 
        prefix="obj",
        keep_original=False
    )


def transform_fval(
    data: pl.DataFrame,
    lb: float = 1e-8,
    ub: float = 1e8,
    scale_log: bool = True,
    maximization: bool = False,
    fval_col: str = "raw_y",
):
    """
    Helper function to transform function values (min-max normalization based on provided bounds and scaling)
    """
    bounds = {fval_col: (lb, ub)}
    res = normalize_objectives(
        data,
        obj_cols=[fval_col],
        bounds=bounds,
        log_scale=scale_log,
        maximize=maximization,
        prefix="eaf"
    )
    return res