import polars as pl
from typing import Callable, Sequence
from .utils import get_sequence
from .._upgrade import upgrade_hints
from ..align import align_data


@upgrade_hints
def aggregate_convergence(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    by: Sequence[str] = ("algorithm_name",),
    eval_min: int = None,
    eval_max: int = None,
    custom_op: Callable[[pl.Series], float] = None,
    maximization: bool = False,
) -> pl.DataFrame:
    """Aggregate performance data from a fixed-budget perspective with multiple statistics.

    Args:
        data (pl.DataFrame): The data object containing evaluation and performance data.
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        fval_var (str, optional): Which column contains the function values. Defaults to "raw_y".
        by (Sequence[str], optional): The columns to group by: runs columns or
            descriptors. Defaults to ["algorithm_name"].
        eval_min (int, optional): Minimum evaluation value to include. If None, uses minimum from data. Defaults to None.
        eval_max (int, optional): Maximum evaluation value to include. If None, uses maximum from data. Defaults to None.
        custom_op (Callable[[pl.Series], float], optional): Custom aggregation function to apply per group. Defaults to None.
        maximization (bool, optional): Whether the objective is being maximized. Defaults to False.

    Returns:
        pl.DataFrame: A DataFrame with aggregated performance statistics (mean, min,
            max, median, std, geometric_mean) over the runs that have a value at the
            budget, ``n_runs`` and ``n_defined`` (the runs that have a value; a run has
            none before its first logged evaluation, AL-5).
    """
    if data.is_empty():
        raise ValueError("Data is empty, cannot aggregate convergence.")

    # Getting alligned data (to check if e.g. limits should be args for this function)
    if eval_min is None:
        eval_min = data[eval_var].min()
    if eval_max is None:
        eval_max = data[eval_var].max()
    x_values = get_sequence(eval_min, eval_max, 50, scale_log=True, cast_to_int=True)
    group_variables = [*by, eval_var]
    data_aligned = align_data(
        data.cast({eval_var: pl.Int64}),
        x_values,
        group_cols=["run_id", *by],
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
    )
    aggregations = [
        pl.mean(fval_var).alias("mean"),
        pl.min(fval_var).alias("min"),
        pl.max(fval_var).alias("max"),
        pl.median(fval_var).alias("median"),
        pl.std(fval_var).alias("std"),
        pl.col(fval_var).log().mean().exp().alias("geometric_mean"),
        pl.len().alias("n_runs"),
        pl.col(fval_var).count().alias("n_defined"),
    ]

    if custom_op is not None:
        aggregations.append(
            pl.col(fval_var)
            .map_batches(
                lambda s: custom_op(s), return_dtype=pl.Float64, returns_scalar=True
            )
            .alias(custom_op.__name__)
        )

    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    return dt_plot.sort(eval_var)
