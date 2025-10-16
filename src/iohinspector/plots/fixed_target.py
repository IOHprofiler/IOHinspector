import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
from typing import Iterable
from iohinspector.metrics.fixed_target import aggregate_running_time


def plot_single_function_fixed_target(
    data: pl.DataFrame,
    evaluation_variable: str = "evaluations",
    fval_variable: str = "raw_y",
    free_variables: Iterable[str] = ["algorithm_name"],
    f_min: float = None,
    f_max: float = None,
    max_budget: int = None,
    maximization: bool = False,
    measures: Iterable[str] = ["ERT"],
    scale_xlog: bool = True,
    scale_ylog: bool = True,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str = None,
):
    """Create a fixed-target plot for a given set of performance data.

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should        be generated from a DataManager.
        evaluation_variable (str, optional): Column in 'data' which corresponds to the number of evaluations. Defaults to "evaluations".
        fval_variable (str, optional): Column in 'data' which corresponds to the performance measure. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Columns in 'data' which correspond to the variables which will be used to distinguish between lines in the plot. Defaults to ["algorithm_name"].
        f_min (float, optional): Minimum value to use for the 'fval_variable', if not present the min of that column will be used. Defaults to None.
        f_max (float, optional): Maximum value to use for the 'fval_variable', if not present the max of that column will be used. Defaults to None.
        max_budget (int, optional): Maximum value to use for the 'evaluation_variable', if not present the max of that column will be used. Defaults to None.
        maximization (bool, optional): Boolean indicating whether the 'fval_variable' is being maximized. Defaults to False.
        measures (Iterable[str], optional): List of measures which should be used in the plot. Valid options are 'ERT', 'mean', 'PAR-10', 'min', 'max'. Defaults to ['ERT'].
        scale_xlog (bool, optional): Should the x-axis be log-scaled. Defaults to True.
        scale_ylog (bool, optional): Should the y-axis be log-scaled. Defaults to True.
        ax (matplotlib.axes._axes.Axes, optional): Existing matplotlib axis object to draw the plot on.
        file_name (str, optional): Where should the resulting plot be stored. Defaults to None. If existing axis is provided, this functionality is disabled.

    Returns:
        pd.DataFrame: The final dataframe which was used to create the plot
    """
    dt_agg = aggregate_running_time(
        data,
        evaluation_variable=evaluation_variable,
        fval_variable=fval_variable,
        free_variables=free_variables,
        f_min=f_min,
        f_max=f_max,
        scale_flog=scale_xlog,
        max_budget=max_budget,
        maximization=maximization,
    )

    dt_molt = dt_agg.melt(id_vars=[fval_variable] + free_variables)
    dt_plot = dt_molt[dt_molt["variable"].isin(measures)].sort_values(free_variables)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    
    sbs.lineplot(
        dt_plot,
        x=fval_variable,
        y="value",
        style="variable",
        hue=dt_plot[free_variables].apply(tuple, axis=1),
        ax=ax,
    )
    if scale_xlog:
        ax.set_xscale("log")
    if scale_ylog:
        ax.set_yscale("log")

    if not maximization:
        ax.set_xlim(ax.get_xlim()[::-1])

    if ax is None and file_name:
        fig.tight_layout()
        fig.savefig(file_name)

    return dt_plot


def plot_multi_function_fixed_target():
    # either just loop over function column(s), or more advanced
    raise NotImplementedError