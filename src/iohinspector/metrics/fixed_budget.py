import polars as pl
import pandas as pd
from typing import Iterable, Callable
from .utils import get_sequence
from ..align import align_data

def aggregate_convergence(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    free_vars: Iterable[str] = ["algorithm_name"],
    eval_min: int = None,
    eval_max: int = None,
    custom_op: Callable[[pl.Series], float] = None,
    maximization: bool = False,
    return_as_pandas: bool = True,
) -> pl.DataFrame | pd.DataFrame:
    """Aggregate performance data from a fixed-budget perspective with multiple statistics.

    Args:
        data (pl.DataFrame): The data object containing evaluation and performance data.
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        fval_var (str, optional): Which column contains the function values. Defaults to "raw_y".
        free_vars (Iterable[str], optional): Which columns to NOT aggregate over. Defaults to ["algorithm_name"].
        eval_min (int, optional): Minimum evaluation value to include. If None, uses minimum from data. Defaults to None.
        eval_max (int, optional): Maximum evaluation value to include. If None, uses maximum from data. Defaults to None.
        custom_op (Callable[[pl.Series], float], optional): Custom aggregation function to apply per group. Defaults to None.
        maximization (bool, optional): Whether the objective is being maximized. Defaults to False.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pl.DataFrame or pd.DataFrame: A DataFrame with aggregated performance statistics (mean, min, max, median, std, geometric_mean).
    """
    if(data.is_empty()):
        raise ValueError("Data is empty, cannot aggregate convergence.")

    # Getting alligned data (to check if e.g. limits should be args for this function)
    if eval_min is None:
        eval_min = data[eval_var].min()
    if eval_max is None:
        eval_max = data[eval_var].max()
    x_values = get_sequence(eval_min, eval_max, 50, scale_log=True, cast_to_int=True)
    group_variables = free_vars + [eval_var]
    data_aligned = align_data(
        data.cast({eval_var: pl.Int64}),
        x_values,
        group_cols=["data_id"] + free_vars,
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
        silence_warning=True
    )
    aggregations = [
        pl.mean(fval_var).alias("mean"),
        pl.min(fval_var).alias("min"),
        pl.max(fval_var).alias("max"),
        pl.median(fval_var).alias("median"),
        pl.std(fval_var).alias("std"),
        pl.col(fval_var).log().mean().exp().alias("geometric_mean")
    ]

    if custom_op is not None:
        aggregations.append(
            pl.col(fval_var).map_batches(
                lambda s: custom_op(s), return_dtype=pl.Float64, returns_scalar=True
            ).alias(custom_op.__name__)
    )
        
    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    if return_as_pandas:
        return dt_plot.sort(eval_var).to_pandas()
    return dt_plot.sort(eval_var)