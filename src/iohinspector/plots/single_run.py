import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
import polars as pl
from typing import Iterable, Optional
import numpy as np
from iohinspector.plots.utils import HeatmapPlotArgs, _create_plot_args, _save_fig
from iohinspector.metrics.single_run import get_heatmap_single_run_data

def plot_heatmap_single_run(
    data: pl.DataFrame,
    vars: Iterable[str],
    eval_var: str = "evaluations",
    var_mins: Iterable[float] = [-5],
    var_maxs: Iterable[float] = [5],
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
    plot_args: dict | HeatmapPlotArgs = None
):
    """Create a heatmap visualization showing search space exploration patterns in a single algorithm run.

    Visualizes how an optimization algorithm explores the search space over time by showing 
    the density of evaluations across different variable dimensions and evaluation budgets,
    revealing search patterns and exploration behavior.

    Args:
        data (pl.DataFrame): Input dataframe containing trajectory data from a single algorithm run.
            Must contain data for exactly one run (unique data_id).
        vars (Iterable[str]): Which columns contain the decision/search space variables to visualize.
        eval_var (str, optional): Which column contains the evaluation counts. Defaults to "evaluations".
        var_mins (Iterable[float], optional): Minimum bounds for the search space variables. 
            Should be same length as vars. Defaults to [-5].
        var_maxs (Iterable[float], optional): Maximum bounds for the search space variables. 
            Should be same length as vars. Defaults to [5].
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (Optional[str], optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | HeatmapPlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. No default title set.
            - xlabel (str): X-axis label. Defaults to eval_var value.
            - ylabel (str): Y-axis label. Defaults to "Variables".
            - figsize (Tuple[float, float]): Figure size. Defaults to (32, 9).
            - heatmap_palette (str): Colormap for the heatmap. Defaults to "viridis".
            - All other HeatmapPlotArgs parameters (xlim, ylim, xscale, yscale, grid, legend, fontsize, etc.).

    Returns:
        tuple[matplotlib.axes.Axes, pd.DataFrame]: The matplotlib axes object and the processed 
            heatmap dataframe used to create the plot.

    Raises:
        AssertionError: If data contains multiple runs (data_id has more than one unique value).
    """
    assert data["data_id"].n_unique() == 1

    dt_plot = get_heatmap_single_run_data(
        data = data,
        vars = vars,
        eval_var=eval_var,
        var_mins=var_mins,
        var_maxs=var_maxs,
    )
    
    plot_args = _create_plot_args(
        HeatmapPlotArgs(
            figsize= (32, 9),
            xlabel= eval_var,
            ylabel= "Variables",
        ),
        plot_args
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=plot_args.figsize)
    else:
        fig = None
        
    sbs.heatmap(dt_plot, cmap=plot_args.heatmap_palette, vmin=0, vmax=1, ax=ax)

    plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args=plot_args)
    return ax, dt_plot
