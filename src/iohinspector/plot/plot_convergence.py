import polars as pl
from typing import Iterable
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
from iohinspector.data_processing import aggegate_convergence


def single_function_fixedbudget(
    data: pl.DataFrame,
    evaluation_variable: str = "evaluations",
    fval_variable: str = "raw_y",
    free_variables: Iterable[str] = ["algorithm_name"],
    x_min: float = None,
    x_max: float = None,
    maximization: bool = False,
    measures: Iterable[str] = ["geometric_mean"],
    scale_xlog: bool = True,
    scale_ylog: bool = True,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str = None,
):
    """Create a fixed-budget plot for a given set of performance data.

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should        be generated from a DataManager.
        evaluation_variable (str, optional): Column in 'data' which corresponds to the number of evaluations. Defaults to "evaluations".
        fval_variable (str, optional): Column in 'data' which corresponds to the performance measure. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Columns in 'data' which correspond to the variables which will be used to distinguish between lines in the plot. Defaults to ["algorithm_name"].
        x_min (float, optional): Minimum value to use for the 'evaluation_variable', if not present the min of that column will be used. Defaults to None.
        x_max (float, optional): Maximum value to use for the 'evaluation_variable', if not present the max of that column will be used. Defaults to None.
        maximization (bool, optional): Boolean indicating whether the 'fval_variable' is being maximized. Defaults to False.
        measures (Iterable[str], optional): List of measures which should be used in the plot. Valid options are 'geometric_mean', 'mean', 'median', 'min', 'max'. Defaults to ['geometric_mean'].
        scale_xlog (bool, optional): Should the x-axis be log-scaled. Defaults to True.
        scale_ylog (bool, optional): Should the y-axis be log-scaled. Defaults to True.
        ax (matplotlib.axes._axes.Axes, optional): Existing matplotlib axis object to draw the plot on.
        file_name (str, optional): Where should the resulting plot be stored. Defaults to None. If existing axis is provided, this functionality is disabled.

    Returns:
        pd.DataFrame: The final dataframe which was used to create the plot
    """
    dt_agg = aggegate_convergence(
        data,
        evaluation_variable=evaluation_variable,
        fval_variable=fval_variable,
        free_variables=free_variables,
        x_min=x_min,
        x_max=x_max,
        maximization=maximization,
    )

    dt_molt = dt_agg.melt(id_vars=[evaluation_variable] + free_variables)
    dt_plot = dt_molt[dt_molt["variable"].isin(measures)].sort_values(free_variables)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    sbs.lineplot(
        dt_plot,
        x=evaluation_variable,
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
