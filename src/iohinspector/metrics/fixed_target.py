import polars as pl
from typing import Iterable, Callable
from .utils import get_sequence, geometric_mean
from ..align import align_data

def aggregate_running_time(
    data: pl.DataFrame,
    evaluation_variable: str = "evaluations",
    fval_variable: str = "raw_y",
    free_variables: Iterable[str] = ["algorithm_name"],
    f_min: float = None,
    f_max: float = None,
    scale_flog: bool = True,
    max_budget: int = None,
    maximization: bool = False,
    custom_op: Callable[[pl.Series], float] = None,
    return_as_pandas: bool = True,
):
    """Function to aggregate performance on a fixed-target perspective

    Args:
        data (pl.DataFrame): The data object to use for getting the performance. Note that the fval, evaluation and free variables as defined in
        this object determine the axes of the final performance (most data will have 'raw_y', 'evaluations' and ['algId'] as defaults)
        evaluation_variable (str, optional): Column name for evaluation number. Defaults to "evaluations".
        fval_variable (str, optional): Column name for function value. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Column name for free variables (variables over which performance should not be aggregated). Defaults to ["algorithm_name"].
        f_min (float, optional): Minimum function value to use. Defaults to None (minimum present in data).
        f_max (float, optional): Maximum function value to use. Defaults to None (maximum present in data).
        scale_flog (bool): Whether or not function values should be scaled logarithmically for the x-axis. Defaults to True.
        max_budget: If present, what budget value should be the maximum considered. Defaults to None.
        custom_op (Callable[[pl.Series], float], optional): Custom aggregation method for performance values. Defaults to None.
        maximization (bool, optional): Whether performance metric is being maximized or not. Defaults to False.
        return_as_pandas (bool, optional): Whether the data should be returned as Pandas (True) or Polars (False) object. Defaults to True.

    Returns:
        DataFrame: Depending on 'return_as_pandas', a pandas or polars DataFrame with the aggregated performance values
    """

    # Getting alligned data (to check if e.g. limits should be args for this function)
    if f_min is None:
        f_min = data[fval_variable].min()
    if f_max is None:
        f_max = data[fval_variable].max()
    f_values = get_sequence(f_min, f_max, 50, scale_log=scale_flog)
    group_variables = free_variables + [fval_variable]
    data_aligned = align_data(
        data,
        f_values,
        group_cols=["data_id"] + free_variables,
        x_col=fval_variable,
        y_col=evaluation_variable,
        maximization=maximization,
    )

    if max_budget is None:
        max_budget = data[evaluation_variable].max()

    aggregations = [
        pl.col(evaluation_variable).mean().alias("mean"),
        pl.col(evaluation_variable).min().alias("min"),
        pl.col(evaluation_variable).max().alias("max"),
        pl.col(evaluation_variable).median().alias("median"),
        pl.col(evaluation_variable).std().alias("std"),
        pl.col(evaluation_variable).is_finite().mean().alias("success_ratio"),
        pl.col(evaluation_variable).is_finite().sum().alias("success_count"),
        (
            pl.when(pl.col(evaluation_variable).is_finite())
            .then(pl.col(evaluation_variable))
            .otherwise(max_budget)
            .sum()
            /pl.col(evaluation_variable).is_finite().sum()
        ).alias("ERT"),
        (
            pl.when(pl.col(evaluation_variable).is_finite())
            .then(pl.col(evaluation_variable))
            .otherwise(10 * max_budget)
            .sum()
            / pl.col(evaluation_variable).count()
        ).alias("PAR-10"),
    ]

    if custom_op is not None:
        aggregations.append(
            pl.col(evaluation_variable)
            .map_batches(lambda s: custom_op(s), return_dtype=pl.Float64, returns_scalar=True)
            .alias(custom_op.__name__)
        )
    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    if return_as_pandas:
        return dt_plot.sort(fval_variable).to_pandas()
    return dt_plot.sort(fval_variable)