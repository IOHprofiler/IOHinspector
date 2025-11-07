from typing import Iterable, Optional, cast
from iohinspector.metrics.multi_objective import get_pareto_front_2d, get_indicator_over_time_data
import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
from iohinspector.plots.utils import ScatterPlotArgs, LinePlotArgs, _save_fig, _create_plot_args

def plot_paretofronts_2d(
    data: pl.DataFrame,
    obj1_var: str = "raw_y",
    obj2_var: str = "F2",
    free_var: str = "algorithm_name",
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str = None,
    plot_args: dict | ScatterPlotArgs = None
):
    """Visualize 2D Pareto fronts for multi-objective optimization algorithms.

    Creates a scatter plot showing the non-dominated solutions (Pareto fronts) achieved by 
    different algorithms in a two-objective space, allowing visual comparison of algorithm 
    performance and trade-off quality.

    Args:
        data (pl.DataFrame): Input dataframe containing multi-objective optimization trajectory data.
        obj1_var (str, optional): Which column contains the first objective values. Defaults to "raw_y".
        obj2_var (str, optional): Which column contains the second objective values. Defaults to "F2".
        free_var (str, optional): Which column contains the grouping variable for distinguishing 
            between different algorithms/categories. Defaults to "algorithm_name".
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (str, optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | ScatterPlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "Pareto Fronts".
            - xlabel (str): X-axis label. Defaults to obj1_var value.
            - ylabel (str): Y-axis label. Defaults to obj2_var value.
            - point_colors (Sequence[str]): Colors for different algorithm points. Defaults to seaborn palette.
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other ScatterPlotArgs parameters (xlim, ylim, xscale, yscale, grid, legend, fontsize, etc.).

    Returns:
        tuple[matplotlib.axes.Axes, pd.DataFrame]: The matplotlib axes object and the Pareto front 
            dataframe used to create the plot.
    """
    df = get_pareto_front_2d(
        data, obj1_var=obj1_var, obj2_var=obj2_var
    )

    plot_args = _create_plot_args(
        ScatterPlotArgs(
            xlabel= obj1_var,
            ylabel= obj2_var,
            title= "Pareto Fronts",
        ),
        plot_args
    )

    df.sort_values(free_var)
   
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_args.figsize)
    else:
        fig = None
        
    sbs.scatterplot(
        df,
        x=obj1_var,
        y=obj2_var,
        hue=free_var,
        palette= plot_args.point_colors,
        ax=ax
        )
    
    plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args=plot_args)

    return ax,df


def plot_indicator_over_time(
    data: pl.DataFrame,
    obj_vars: Iterable[str] =  ["raw_y", "F2"],
    indicator: object = None,
    free_var: str = "algorithm_name",
    eval_min: int = 1,
    eval_max: int = 50_000,
    scale_eval_log: bool = True,
    eval_steps: int = 50,
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
    plot_args: dict | LinePlotArgs = None
):
    """Plot the anytime performance of multi-objective quality indicators over evaluation budgets.

    Creates line plots showing how quality indicators (like hypervolume, IGD, etc.) evolve 
    over the course of algorithm runs, enabling comparison of convergence behavior and 
    solution quality improvement across different algorithms.

    Args:
        data (pl.DataFrame): Input dataframe containing multi-objective optimization trajectory data.
        obj_vars (Iterable[str], optional): Which columns contain the objective values for indicator calculation. 
            Defaults to ["raw_y", "F2"].
        indicator (object, optional): Quality indicator object from iohinspector.indicators module. Defaults to None.
        free_var (str, optional): Which column contains the grouping variable for distinguishing 
            between different algorithms. Defaults to "algorithm_name".
        eval_min (int, optional): Minimum evaluation bound for the time axis. Defaults to 1.
        eval_max (int, optional): Maximum evaluation bound for the time axis. Defaults to 50_000.
        scale_eval_log (bool, optional): Whether the evaluation axis should be log-scaled. Defaults to True.
        eval_steps (int, optional): Number of evaluation points to sample between eval_min and eval_max. Defaults to 50.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (Optional[str], optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | LinePlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "Anytime Performance: {indicator.var_name}".
            - xlabel (str): X-axis label. Defaults to "evaluations".
            - ylabel (str): Y-axis label. Defaults to indicator.var_name value.
            - xscale (str): X-axis scale ("log" or "linear"). Defaults to "log" if scale_eval_log=True.
            - line_colors (Sequence[str]): Colors for different algorithm lines. Defaults to seaborn palette.
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other LinePlotArgs parameters (xlim, ylim, yscale, grid, legend, fontsize, etc.).

    Returns:
        tuple[matplotlib.axes.Axes, pd.DataFrame]: The matplotlib axes object and the indicator 
            performance dataframe used to create the plot.
    """
    df = get_indicator_over_time_data(
        data,
        indicator=indicator,
        obj_vars=obj_vars,
        eval_min=eval_min,
        eval_max=eval_max,
        scale_eval_log=scale_eval_log,
        eval_steps=eval_steps,
    )

    plot_args = _create_plot_args(
        LinePlotArgs(
            xlabel= "evaluations",
            ylabel= indicator.var_name,
            title= f"Anytime Performance: {indicator.var_name}",
            xscale= "log" if scale_eval_log else "linear",
        ),
        plot_args
    )
        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_args.figsize)
    else:
        fig = None
    sbs.lineplot(
        df,
        x="evaluations",
        y=indicator.var_name,
        hue=free_var,
        palette=sbs.color_palette(n_colors=len(np.unique(data[free_var]))),
        ax=ax,
    )
   
    plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args=plot_args)
    return ax, df
