import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
import polars as pl
from typing import Iterable
from iohinspector.metrics.fixed_budget import aggregate_convergence
from iohinspector.plots.utils import LinePlotArgs, _save_fig, _create_plot_args
import matplotlib

def plot_single_function_fixed_budget(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    free_vars: Iterable[str] = ["algorithm_name"],
    eval_min: float = None,
    eval_max: float = None,
    maximization: bool = False,
    measures: Iterable[str] = ["geometric_mean"],
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str = None,
    plot_args: dict | LinePlotArgs = None,
):
    """Create a fixed-budget convergence plot showing algorithm performance over evaluation budgets.

    Visualizes how different algorithms converge by plotting aggregate performance measures 
    (geometric mean, median, etc.) against evaluation budgets, allowing direct comparison 
    of convergence behavior across algorithms.

    Args:
        data (pl.DataFrame): Input dataframe containing optimization algorithm trajectory data.
        eval_var (str, optional): Which column contains the evaluation counts. Defaults to "evaluations".
        fval_var (str, optional): Which column contains the function/objective values. Defaults to "raw_y".
        free_vars (Iterable[str], optional): Which columns contain the grouping variables for distinguishing 
            between different lines in the plot. Defaults to ["algorithm_name"].
        eval_min (float, optional): Minimum evaluation bound for the plot. If None, uses data minimum. Defaults to None.
        eval_max (float, optional): Maximum evaluation bound for the plot. If None, uses data maximum. Defaults to None.
        maximization (bool, optional): Whether the optimization problem is maximization. Defaults to False.
        measures (Iterable[str], optional): Aggregate measures to plot. Valid options are "geometric_mean", 
            "mean", "median", "min", "max". Defaults to ["geometric_mean"].
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (str, optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | LinePlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "Fixed-Budget Plot".
            - xlabel (str): X-axis label. Defaults to eval_var value.
            - ylabel (str): Y-axis label. Defaults to fval_var value.
            - xscale (str): X-axis scale. Defaults to "log".
            - yscale (str): Y-axis scale. Defaults to "log".
            - line_colors (Sequence[str]): Colors for different lines. Defaults to seaborn palette.
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other LinePlotArgs parameters (xlim, ylim, grid, legend, fontsize, etc.).

    Returns:
        tuple[matplotlib.axes.Axes, pl.DataFrame]: The matplotlib axes object and the processed 
            (melted/filtered) dataframe used to create the plot.
    """
    dt_agg = aggregate_convergence(
        data,
        eval_var=eval_var,
        fval_var=fval_var,
        free_vars=free_vars,
        eval_min=eval_min,
        eval_max=eval_max,
        maximization=maximization,
    )
    dt_molt = dt_agg.melt(id_vars=[eval_var] + free_vars)
    dt_plot = dt_molt[dt_molt["variable"].isin(measures)].sort_values(free_vars)

    plot_args = _create_plot_args(
        LinePlotArgs(
            xlabel=eval_var,
            ylabel=fval_var,
            title="Fixed-Budget Plot",
            xscale="log",
            yscale="log",
        ),
        plot_args
    )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_args.figsize)
    else:
        fig = None
        
    sbs.lineplot(
        dt_plot,
        x=eval_var,
        y="value",
        style="variable",
        hue=dt_plot[free_vars].apply(tuple, axis=1),
        palette=plot_args.line_colors,
        ax=ax,
    )


    ax = plot_args.apply(ax=ax)

    _save_fig(fig, file_name, plot_args=plot_args)
    
    return ax, dt_plot



def plot_multi_function_fixed_budget():
    raise NotImplementedError