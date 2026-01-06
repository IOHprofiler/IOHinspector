

import numpy as np
import polars as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import seaborn as sbs
from typing import Optional, Iterable
from iohinspector.metrics import get_eaf_data, get_eaf_pareto_data, get_eaf_diff_data
from iohinspector.plots.utils import HeatmapPlotArgs, _create_plot_args, _save_fig
from moocore import eaf, eafdiff

def plot_eaf_single_objective(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    eval_min: int = None,
    eval_max: int = None,
    scale_eval_log: bool = True,
    n_quantiles: int = 100,
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
    plot_args: dict | HeatmapPlotArgs = None
):
    """Plot the Empirical Attainment Function (EAF) for single-objective optimization against budget.

    Creates a heatmap visualization showing the probability of attaining different function values
    at different evaluation budgets across multiple algorithm runs.

    Args:
        data (pl.DataFrame): Input dataframe containing optimization algorithm trajectory data.
        eval_var (str, optional): Which column contains the evaluation counts. Defaults to "evaluations".
        fval_var (str, optional): Which column contains the function values. Defaults to "raw_y".
        eval_min (int, optional): Minimum evaluation bound for the plot. If None, uses data minimum. Defaults to None.
        eval_max (int, optional): Maximum evaluation bound for the plot. If None, uses data maximum. Defaults to None.
        scale_eval_log (bool, optional): Whether the evaluations should be log-scaled. Defaults to True.
        n_quantiles (int, optional): Number of discrete probability levels in the EAF heatmap. Defaults to 100.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (Optional[str], optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | HeatmapPlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "EAF".
            - xlabel (str): X-axis label. Defaults to eval_var value.
            - ylabel (str): Y-axis label. Defaults to fval_var value.
            - xscale (str): X-axis scale ("log" or "linear"). Defaults to "log" if scale_eval_log=True.
            - yscale (str): Y-axis scale. Defaults to "log".
            - xlim (Tuple[float, float]): X-axis limits. Defaults to (eval_min, eval_max).
            - ylim (Tuple[float, float]): Y-axis limits. Defaults to (1e-8, 1e2).
            - heatmap_palette (str): Colormap name. Defaults to "viridis_r".
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other HeatmapPlotArgs parameters.

    Returns:
        tuple[matplotlib.axes.Axes, pl.DataFrame]: The matplotlib axes object and the processed 
            dataframe used to create the plot.
    """
    df = get_eaf_data(
        data,
        eval_var=eval_var,
        eval_min=eval_min,
        eval_max=eval_max,
        scale_eval_log=scale_eval_log,
        return_as_pandas=False,
    )  
    eval_min = df[eval_var].min() 
    eval_max = df[eval_var].max()

    plot_args = _create_plot_args(
        HeatmapPlotArgs(
            xlabel= eval_var,
            ylabel= fval_var,
            title= "EAF",
            xscale= "log" if scale_eval_log else "linear",
            yscale= "log",
            xlim= (eval_min, eval_max),
            ylim= (10**-8,10**2),
            heatmap_palette= "viridis_r",
        ),
        plot_args
    )
    f_min, f_max = plot_args.ylim
    
    
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_args.figsize)
    else:
        fig = None

    quantiles = np.arange(0, 1 + 1 / ((n_quantiles - 1) * 2), 1 / (n_quantiles - 1))
    cmap = plt.get_cmap(plot_args.heatmap_palette)
    norm = plt.Normalize(
        vmin=0,
        vmax=1
        )
    colors = [cmap(norm(quant)) for quant in quantiles]
    if(not plot_args.use_background_color):
        ax.add_patch(
            Rectangle( 
                (eval_min, f_min),
                eval_max - eval_min,
                f_max - f_min,
                facecolor=cmap(norm(0)),
                zorder=0,
                )
        )   
    
    for quant, color in zip(quantiles,colors):
        poly = np.array(
            df.group_by(eval_var).quantile(quant).sort(eval_var)[eval_var, fval_var]
        )
        poly = np.append(
            poly, np.array([[max(poly[:, 0]), f_max]]), axis=0
        )
        poly = np.append(
            poly, np.array([[min(poly[:, 0]), f_max]]), axis=0
        )
        poly2 = np.repeat(poly, 2, axis=0)
        poly2[2::2, 1] = poly[:, 1][:-1]
        ax.add_patch(Polygon(poly2, facecolor=color))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)

    plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args)

    return ax, df


def plot_eaf_pareto(
    data: pl.DataFrame,
    obj1_var: str,
    obj2_var: str,
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
    plot_args: dict | HeatmapPlotArgs = None
):
    """Plot the Empirical Attainment Function (EAF) for multi-objective optimization with two objectives.

    Creates a heatmap visualization showing the probability of attaining different combinations
    of objective values across multiple algorithm runs in the Pareto front space.

    Args:
        data (pl.DataFrame): Input dataframe containing multi-objective optimization trajectory data.
        obj1_var (str): Which column contains the first objective values.
        obj2_var (str): Which column contains the second objective values.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (Optional[str], optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | HeatmapPlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "Pareto EAF".
            - xlabel (str): X-axis label. Defaults to obj1_var value.
            - ylabel (str): Y-axis label. Defaults to obj2_var value.
            - xlim (Tuple[float, float]): X-axis limits. Defaults to data range.
            - ylim (Tuple[float, float]): Y-axis limits. Defaults to data range.
            - heatmap_palette (str): Colormap name. Defaults to "viridis_r".
            - use_background_color (bool): Whether to use background color. Defaults to True.
            - background_color (str): Background color. Defaults to "white".
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other HeatmapPlotArgs parameters.

    Returns:
        tuple[matplotlib.axes.Axes, pd.DataFrame]: The matplotlib axes object and the EAF 
            dataframe used to create the plot.
    """

    eaf_data_df = get_eaf_pareto_data(data, obj1_var, obj2_var)
    
    x_max = eaf_data_df[obj1_var].max()
    x_min = eaf_data_df[obj1_var].min()
    y_max = eaf_data_df[obj2_var].max()
    y_min = eaf_data_df[obj2_var].min()
    
    min_eaf = eaf_data_df["eaf"].min()

    plot_args = _create_plot_args(
        HeatmapPlotArgs(
            xlabel= obj1_var,
            ylabel= obj2_var,
            title= "Pareto EAF",
            xlim= (x_min, x_max),
            ylim= (y_min, y_max),
            heatmap_palette= "viridis_r",
        ),
        plot_args
    )

    x_min, x_max = plot_args.xlim
    y_min, y_max = plot_args.ylim

    if ax is None:
        fig, ax = plt.subplots(figsize=plot_args.figsize)
    else:
        fig = None

    eaf_data_df = eaf_data_df.sort_values(obj1_var)
    
    cmap = plt.get_cmap(plot_args.heatmap_palette)
    norm = plt.Normalize(
        vmin=(min_eaf if plot_args.use_background_color else 0),
        vmax=1
        )
    _unique_eafs = eaf_data_df["eaf"].unique()
    colors = [cmap(norm(v)) for v in _unique_eafs]  


    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor= (plot_args.background_color if plot_args.use_background_color else cmap(norm(0))),
            zorder=0,
        )
    )

    for i, color in zip(eaf_data_df["eaf"].unique(), colors):
        poly = np.array(eaf_data_df[eaf_data_df["eaf"] == i][[obj1_var, obj2_var]])
        poly = np.append(poly, np.array([[x_max, y_max]]), axis=0)
        poly = np.append(poly, np.array([[min(poly[:, 0]), y_max]]), axis=0)
        poly2 = np.repeat(poly, 2, axis=0)
        poly2[2::2, 1] = poly[:, 1][:-1]
        ax.add_patch(Polygon(poly2, facecolor=color))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    # set a background rectangle behind the EAF polygons
    
    ax.set_facecolor("white")
    ax = plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args)

    return ax, eaf_data_df

def plot_eaf_diffs(
    data1: pl.DataFrame,
    data2: pl.DataFrame,
    obj1_var: str,
    obj2_var: str,
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
    plot_args: dict | HeatmapPlotArgs = None
):
    """Plot the Empirical Attainment Function (EAF) differences between two algorithms.

    Creates a heatmap visualization showing the statistical differences in attainment probabilities
    between two algorithms in the objective space, highlighting regions where one algorithm
    performs better than the other.

    Args:
        data1 (pl.DataFrame): Input dataframe containing trajectory data for the first algorithm.
        data2 (pl.DataFrame): Input dataframe containing trajectory data for the second algorithm.
        obj1_var (str): Which column contains the first objective values.
        obj2_var (str): Which column contains the second objective values.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (Optional[str], optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | HeatmapPlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "EAF Differences".
            - xlabel (str): X-axis label. Defaults to obj1_var value.
            - ylabel (str): Y-axis label. Defaults to obj2_var value.
            - xlim (Tuple[float, float]): X-axis limits. Defaults to data range.
            - ylim (Tuple[float, float]): Y-axis limits. Defaults to data range.
            - heatmap_palette (str): Colormap name. Defaults to "viridis".
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other HeatmapPlotArgs parameters.

    Returns:
        tuple[matplotlib.axes.Axes, pd.DataFrame]: The matplotlib axes object and the EAF 
            differences dataframe used to create the plot.

    Note:
        The plot shows regions where data1 performs better (positive differences) and regions 
        where data2 performs better (negative differences) in different colors.
    """
    # TODO: add an approximation version to speed up plotting
    eaf_diff_rect_data = get_eaf_diff_data(
        data1,
        data2,
        obj1_var,
        obj2_var,
    )
    x_min = eaf_diff_rect_data["x_min"].replace([np.inf, -np.inf], np.nan).min()
    x_max = eaf_diff_rect_data["x_max"].replace([np.inf, -np.inf], np.nan).max()
    y_min = eaf_diff_rect_data["y_min"].replace([np.inf, -np.inf], np.nan).min()
    y_max = eaf_diff_rect_data["y_max"].replace([np.inf, -np.inf], np.nan).max()

    plot_args = _create_plot_args(
        HeatmapPlotArgs(
            xlabel= obj1_var,
            ylabel= obj2_var,
            title= "EAF Differences",
            xlim= (x_min, x_max),
            ylim= (y_min, y_max),
        ),
        plot_args
    )
    eaf_min_diff = eaf_diff_rect_data["eaf_diff"].min()
    eaf_max_diff = eaf_diff_rect_data["eaf_diff"].max()
    
    color_dict = {
        k: v
        for k, v in zip(
            np.unique(eaf_diff_rect_data["eaf_diff"]),
            sbs.color_palette(plot_args.heatmap_palette, n_colors=len(np.unique(eaf_diff_rect_data["eaf_diff"]))),
        )
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=plot_args.figsize)
    else:
        fig = None

    for rect in eaf_diff_rect_data.itertuples(index=False):
        ax.add_patch(
            Rectangle(
                (rect.x_min, rect.y_min),
                rect.x_max - rect.x_min,
                rect.y_max - rect.y_min,
                facecolor=color_dict[rect.eaf_diff],
            )
        )
    sm = plt.cm.ScalarMappable(cmap=plot_args.heatmap_palette, norm=plt.Normalize(vmin=eaf_min_diff, vmax=eaf_max_diff))
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    ax = plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args)


    return ax, eaf_diff_rect_data