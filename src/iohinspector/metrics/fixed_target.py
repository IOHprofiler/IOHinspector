import polars as pl
from typing import Callable, Sequence
from .utils import get_sequence
from .._upgrade import upgrade_hints
from ..align import align_data


@upgrade_hints
def aggregate_running_time(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    by: Sequence[str] = ("algorithm_name",),
    f_min: float = None,
    f_max: float = None,
    scale_f_log: bool = True,
    eval_max: int = None,
    maximization: bool = False,
    custom_op: Callable[[pl.Series], float] = None,
) -> pl.DataFrame:
    """Aggregate performance data from a fixed-target perspective with running time statistics.

    Args:
        data (pl.DataFrame): The data object containing performance and evaluation data.
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        fval_var (str, optional): Which column contains the function values. Defaults to "raw_y".
        by (Sequence[str], optional): The columns to group by: runs columns or
            descriptors. Defaults to ["algorithm_name"].
        f_min (float, optional): Minimum function value to use. If None, uses minimum from data. Defaults to None.
        f_max (float, optional): Maximum function value to use. If None, uses maximum from data. Defaults to None.
        scale_f_log (bool, optional): Whether to use logarithmic scaling for function values. Defaults to True.
        eval_max (int, optional): Maximum evaluation value to consider. If None, uses maximum from data. Defaults to None.
        maximization (bool, optional): Whether the performance metric is being maximized. Defaults to False.
        custom_op (Callable[[pl.Series], float], optional): Custom aggregation function to apply per group. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with aggregated running time statistics (mean, min,
            max, median, std, success_ratio, ERT, PAR-10).
    """

    # Getting alligned data (to check if e.g. limits should be args for this function)
    if f_min is None:
        f_min = data[fval_var].min()
    if f_max is None:
        f_max = data[fval_var].max()
    f_values = get_sequence(f_min, f_max, 50, scale_log=scale_f_log)
    group_variables = [*by, fval_var]
    data_aligned = align_data(
        data,
        f_values,
        group_cols=["run_id", *by],
        x_col=fval_var,
        y_col=eval_var,
        maximization=maximization,
        silence_warning=True,
    )

    if eval_max is None:
        eval_max = data[eval_var].max()

    aggregations = [
        pl.col(eval_var).mean().alias("mean"),
        pl.col(eval_var).min().alias("min"),
        pl.col(eval_var).max().alias("max"),
        pl.col(eval_var).median().alias("median"),
        pl.col(eval_var).std().alias("std"),
        pl.col(eval_var).is_finite().mean().alias("success_ratio"),
        pl.col(eval_var).is_finite().sum().alias("success_count"),
        (
            pl.when(pl.col(eval_var).is_finite())
            .then(pl.col(eval_var))
            .otherwise(eval_max)
            .sum()
            / pl.col(eval_var).is_finite().sum()
        ).alias("ERT"),
        (
            pl.when(pl.col(eval_var).is_finite())
            .then(pl.col(eval_var))
            .otherwise(10 * eval_max)
            .sum()
            / pl.col(eval_var).count()
        ).alias("PAR-10"),
    ]

    if custom_op is not None:
        aggregations.append(
            pl.col(eval_var)
            .map_batches(
                lambda s: custom_op(s), return_dtype=pl.Float64, returns_scalar=True
            )
            .alias(custom_op.__name__)
        )
    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    return dt_plot.sort(fval_var)
