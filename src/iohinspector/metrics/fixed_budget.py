import polars as pl
from typing import Iterable, Callable
from .utils import get_sequence, geometric_mean
from ..align import align_data

def aggregate_convergence(
    data: pl.DataFrame,
    evaluation_variable: str = "evaluations",
    fval_variable: str = "raw_y",
    free_variables: Iterable[str] = ["algorithm_name"],
    x_min: int = None,
    x_max: int = None,
    custom_op: Callable[[pl.Series], float] = None,
    maximization: bool = False,
    return_as_pandas: bool = True,
):
    """Function to aggregate performance on a fixed-budget perspective

    Args:
        data (pl.DataFrame): The data object to use for getting the performance. Note that the fval, evaluation and free variables as defined in
        this object determine the axes of the final performance (most data will have 'raw_y', 'evaluations' and ['algId'] as defaults)
        evaluation_variable (str, optional): Column name for evaluation number. Defaults to "evaluations".
        fval_variable (str, optional): Column name for function value. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Column name for free variables (variables over which performance should not be aggregated). Defaults to ["algorithm_name"].
        x_min (int, optional): Minimum evaulation value to use. Defaults to None (minimum present in data).
        x_max (int, optional): Maximum evaulation value to use. Defaults to None (maximum present in data).
        custom_op (Callable[[pl.Series], float], optional): Custom aggregation method for performance values. Defaults to None.
        maximization (bool, optional): Whether performance metric is being maximized or not. Defaults to False.
        return_as_pandas (bool, optional): Whether the data should be returned as Pandas (True) or Polars (False) object. Defaults to True.

    Returns:
        DataFrame: Depending on 'return_as_pandas', a pandas or polars DataFrame with the aggregated performance values
    """
    if(data.is_empty()):
        raise ValueError("Data is empty, cannot aggregate convergence.")

    # Getting alligned data (to check if e.g. limits should be args for this function)
    if x_min is None:
        x_min = data[evaluation_variable].min()
    if x_max is None:
        x_max = data[evaluation_variable].max()
    x_values = get_sequence(x_min, x_max, 50, scale_log=True, cast_to_int=True)
    group_variables = free_variables + [evaluation_variable]
    data_aligned = align_data(
        data.cast({evaluation_variable: pl.Int64}),
        x_values,
        group_cols=["data_id"] + free_variables,
        x_col=evaluation_variable,
        y_col=fval_variable,
        maximization=maximization,
    )
    aggregations = [
        pl.mean(fval_variable).alias("mean"),
        pl.min(fval_variable).alias("min"),
        pl.max(fval_variable).alias("max"),
        pl.median(fval_variable).alias("median"),
        pl.std(fval_variable).alias("std"),
        pl.col(fval_variable).log().mean().exp().alias("geometric_mean")
    ]

    if custom_op is not None:
        aggregations.append(
            pl.col(fval_variable).map_batches(
                lambda s: custom_op(s), return_dtype=pl.Float64, returns_scalar=True
            ).alias(custom_op.__name__)
    )
        
    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    if return_as_pandas:
        return dt_plot.sort(evaluation_variable).to_pandas()
    return dt_plot.sort(evaluation_variable)