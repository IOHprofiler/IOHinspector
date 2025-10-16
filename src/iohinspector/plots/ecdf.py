import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
import polars as pl
from typing import Iterable, Optional
from iohinspector.metrics import get_data_ecdf


def plot_ecdf(
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    free_vars: Iterable[str] = ["algorithm_name"],
    maximization: bool = False,
    x_values: Iterable[int] = None,
    x_min: int = None,
    x_max: int = None,
    scale_xlog: bool = True,
    y_min: int = None,
    y_max: int = None,
    scale_ylog: bool = True,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
):
    """Function to plot empirical cumulative distribution function (Based on EAF)

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        eval_var (str, optional): Column in 'data' which corresponds to the number of evaluations. Defaults to "evaluations".
        fval_var (str, optional): Column in 'data' which corresponds to the performance measure. Defaults to "raw_y".
        free_vars (Iterable[str], optional): Columns in 'data' which correspond to the variables which will be used to distinguish between lines in the plot. Defaults to ["algorithm_name"].
        x_min (float, optional): Minimum value to use for the 'eval_var', if not present the min of that column will be used. Defaults to None.
        x_max (float, optional): Maximum value to use for the 'eval_var', if not present the max of that column will be used. Defaults to None.
        x_values (Iterable[int], optional): List of x-values at which to plot the ECDF. If not provided, the x_min, x_max and scale_xlog arguments will be used to sample these points.
        scale_xlog (bool, optional): Should the x-axis be log-scaled. Defaults to True.
        y_min (float, optional): Minimum value to use for the 'fval_var', if not present the min of that column will be used. Defaults to None.
        y_max (float, optional): Maximum value to use for the 'fval_var', if not present the max of that column will be used. Defaults to None.
        scale_ylog (bool, optional): Should the y-values be log-scaled before normalization. Defaults to True.
        maximization (bool, optional): Boolean indicating whether the 'fval_var' is being maximized. Defaults to False.
        measures (Iterable[str], optional): List of measures which should be used in the plot. Valid options are 'geometric_mean', 'mean', 'median', 'min', 'max'. Defaults to ['geometric_mean'].
        ax (matplotlib.axes._axes.Axes, optional): Existing matplotlib axis object to draw the plot on.
        file_name (str, optional): Where should the resulting plot be stored. Defaults to None. If existing axis is provided, this functionality is disabled.

    Returns:
        pd.DataFrame: pandas dataframe of the exact data used to create the plot
    """
    dt_plot = get_data_ecdf(
        data,
        fval_var=fval_var,
        eval_var=eval_var,
        free_vars=free_vars,
        maximization=maximization,
        x_values=x_values,
        x_min=x_min,
        x_max=x_max,
        scale_xlog=scale_xlog,
        y_min=y_min,
        y_max=y_max,
        scale_ylog=scale_ylog,
        turbo=True
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    if len(free_vars) == 1:
        hue_arg = free_vars[0]
        style_arg = free_vars[0]
    else:
        style_arg = free_vars[0]
        hue_arg = dt_plot[free_vars[1:]].apply(tuple, axis=1)

    sbs.lineplot(
        dt_plot,
        x="evaluations",
        y="eaf",
        style=style_arg,
        hue=hue_arg,
        ax=ax,
    )
    if scale_xlog:
        ax.set_xscale("log")
    ax.grid()
    if ax is None and file_name:
        fig.tight_layout()
        fig.savefig(file_name)
    return dt_plot