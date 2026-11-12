import numpy as np
import polars as pl
from moocore import eaf, eafdiff

from .._upgrade import upgrade_hints
from ..align import align_data
from ..targets import seq_rt
from .utils import transform_fval


@upgrade_hints
def get_discritized_eaf_single_objective(
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    eval_values=None,
    eval_min=None,
    eval_max=None,
    eval_targets=10,
    scale_eval_log: bool = True,
    f_min=1e-8,
    f_max=1e2,
    scale_f_log: bool = True,
    f_targets=101,
    maximization: bool = False,
) -> pl.DataFrame:
    """Generate discretized EAF data for single-objective optimization problems.

    Args:
        data (pl.DataFrame): The data object containing optimization trajectory data.
        fval_var (str, optional): Which column contains the function values. Defaults to "raw_y".
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        eval_values (array-like, optional): Specific evaluation values to use. If None, generated from eval_min/max.
        eval_min (int, optional): Minimum evaluation value. If None, uses minimum from data.
        eval_max (int, optional): Maximum evaluation value. If None, uses maximum from data.
        eval_targets (int, optional): Number of evaluation targets to generate. Defaults to 10.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        f_min (float, optional): Minimum function value for scaling. Defaults to 1e-8.
        f_max (float, optional): Maximum function value for scaling. Defaults to 1e2.
        scale_f_log (bool, optional): Whether to use logarithmic scaling for function values. Defaults to True.
        f_targets (int, optional): Number of function value targets to generate. Defaults to 101.
        maximization (bool, optional): Whether larger function values are better.

    Returns:
        pl.DataFrame: The discretized EAF: an ``eaf_target`` column and one column per
            evaluation value (named by its value), sorted by ``eaf_target``.
    """

    if eval_values is None:
        eval_values = seq_rt(
            data[eval_var], eval_targets, scale_eval_log, eval_min, eval_max
        )

    dt_aligned = align_data(
        data, eval_values, x_col=eval_var, y_col=fval_var, maximization=maximization
    )
    dt_aligned = transform_fval(
        dt_aligned,
        lb=f_min,
        ub=f_max,
        scale_log=scale_f_log,
        maximization=maximization,
        fval_var=fval_var,
    )
    targets = pl.DataFrame({"eaf_target": np.linspace(0, 1, f_targets)})
    merged = targets.join(dt_aligned.select(eval_var, "eaf"), how="cross")
    merged = merged.with_columns(ps=pl.col("eaf_target") <= pl.col("eaf"))
    return (
        merged.sort(eval_var)
        .pivot(on=eval_var, index="eaf_target", values="ps", aggregate_function="mean")
        .sort("eaf_target")
    )


@upgrade_hints
def get_eaf_data(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    eval_min: int = None,
    eval_max: int = None,
    scale_eval_log: bool = True,
    maximization: bool = False,
) -> pl.DataFrame:
    """Generate aligned EAF data for visualization and analysis.

    Args:
        data (pl.DataFrame): The data object containing optimization trajectory data.
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        eval_min (int, optional): Minimum evaluation value. If None, uses minimum from data.
        eval_max (int, optional): Maximum evaluation value. If None, uses maximum from data.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        maximization (bool, optional): Whether larger function values are better.

    Returns:
        pl.DataFrame: A DataFrame with aligned EAF data; null before a run's first
            logged evaluation (AL-5).
    """
    evals = seq_rt(data[eval_var], 50, scale_eval_log, eval_min, eval_max)
    return align_data(data, evals, maximization=maximization)


def _points(data: pl.DataFrame, obj1_var: str, obj2_var: str) -> np.ndarray:
    """The objective values and a run number (1, 2, ...) per row, for moocore."""
    run = pl.col("run_id").rank("dense").alias("run")
    return data.select(obj1_var, obj2_var, run).to_numpy()


@upgrade_hints
def get_eaf_pareto_data(
    data: pl.DataFrame,
    obj1_var: str,
    obj2_var: str,
) -> pl.DataFrame:
    """Generate EAF data for multi-objective optimization problems using Pareto fronts.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization data.
        obj1_var (str): Name of the column containing first objective values.
        obj2_var (str): Name of the column containing second objective values.

    Returns:
        pl.DataFrame: A DataFrame with EAF data including objective values and EAF
            percentiles.
    """
    points = _points(data, obj1_var, obj2_var)
    eaf_data = eaf(points[:, :-1], points[:, -1])
    eaf_data_df = pl.DataFrame(
        eaf_data, schema=[obj1_var, obj2_var, "eaf"], orient="row"
    )
    # scale EAF values from percentages to proportions
    return eaf_data_df.with_columns(pl.col("eaf").cast(pl.Float64) / 100.0)


@upgrade_hints
def get_eaf_diff_data(
    data1: pl.DataFrame,
    data2: pl.DataFrame,
    obj1_var: str,
    obj2_var: str,
) -> pl.DataFrame:
    """Calculate EAF difference data between two multi-objective optimization datasets.

    Args:
        data1 (pl.DataFrame): First dataset containing multi-objective optimization data.
        data2 (pl.DataFrame): Second dataset containing multi-objective optimization data.
        obj1_var (str): Name of the column containing first objective values.
        obj2_var (str): Name of the column containing second objective values.

    Returns:
        pl.DataFrame: A DataFrame with EAF difference rectangles and difference values.
    """
    x = _points(data1, obj1_var, obj2_var)
    y = _points(data2, obj1_var, obj2_var)
    if np.array_equal(
        np.sort(x.view(np.void), axis=0), np.sort(y.view(np.void), axis=0)
    ):
        cols = ["x_min", "y_min", "x_max", "y_max", "eaf_diff"]
        return pl.DataFrame({c: [] for c in cols})
    eaf_diff_rect = eafdiff(x, y, rectangles=True)
    return pl.DataFrame(
        eaf_diff_rect, schema=["x_min", "y_min", "x_max", "y_max", "eaf_diff"]
    )
