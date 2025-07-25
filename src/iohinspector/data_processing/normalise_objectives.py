import polars as pl
import numpy as np
import warnings
from typing import Iterable, Optional, Union, Dict


def normalize_objectives(
    data: pl.DataFrame,
    obj_cols: Iterable[str] = ["raw_y"],
    bounds: Optional[Dict[str, tuple[Optional[float], Optional[float]]]] = None,
    log_scale: Union[bool, Dict[str, bool]] = False,
    maximize: Union[bool, Dict[str, bool]] = False,
    prefix: str = "ert"
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
            norm_expr = norm_expr.alias(f"{prefix}_{col}")
        else:
            # If only one objective, use the prefix directly
            norm_expr = norm_expr.alias(prefix)
        result = result.with_columns(norm_expr)

    return result