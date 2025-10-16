import numpy as np
import polars as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import seaborn as sbs
from typing import Optional, Iterable
from iohinspector.metrics import get_sequence
from iohinspector.align import align_data
from moocore import eaf, eafdiff

def plot_eaf_single_objective(
    data: pl.DataFrame,
    min_budget: int = None,
    max_budget: int = None,
    scale_xlog: bool = True,
    n_quantiles: int = 100,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
):
    """Plot the EAF for a single objective column agains budget. For the EAF-plot for multiple objective
    columns, see 'plot_eaf_pareto'.

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        n_quantiles (int, optional): Number of discrete levels in the EAF. Defaults to 100.
        eval_var (str, optional): The variable corresponding to evaluations. Defaults to 'evaluations'
        fval_var (str, optional): The variable corresponding to function values. Defaults to "raw_y".
        scale_xlog (bool, optional): Whether the evaluations should be log-scaled. Defaults to True.
        min_budget (Iterable[float], optional): Minimum bound for the variables. Should be of the same length as 'var_cols'. Defaults to [-5].
        max_budget (Iterable[float], optional): Maximum bound for the variables. Should be of the same length as 'var_cols'.. Defaults to [5].
        ax (matplotlib.axes._axes.Axes, optional): Axis on which to create the plot. Defaults to None.
        file_name (Optional[str], optional): If ax is not given, filename to save the plot. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe of the exact data used to create the plot
    """
    if min_budget is None:
        min_budget = data[eval_var].min()
    if max_budget is None:
        max_budget = data[eval_var].max()
    evals = get_sequence(min_budget, max_budget, 50, scale_xlog, True)
    long = align_data(data, np.array(evals, "uint64"), ["data_id"], output="long")
    quantiles = np.arange(0, 1 + 1 / ((n_quantiles - 1) * 2), 1 / (n_quantiles - 1))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    
    colors = sbs.color_palette("viridis", n_colors=len(quantiles))
    for quant, color in zip(quantiles, colors[::-1]):
        poly = np.array(
            long.group_by(eval_var).quantile(quant).sort(eval_var)[eval_var, fval_var]
        )
        poly = np.append(
            poly, np.array([[max(poly[:, 0]), long[fval_var].max()]]), axis=0
        )
        poly = np.append(
            poly, np.array([[min(poly[:, 0]), long[fval_var].max()]]), axis=0
        )
        poly2 = np.repeat(poly, 2, axis=0)
        poly2[2::2, 1] = poly[:, 1][:-1]
        ax.add_patch(Polygon(poly2, facecolor=color))

    ax.set_ylim(long[fval_var].min(), long[fval_var].max())
    ax.set_ylim(1e-8, 1)
    ax.set_xlim(min(evals), max(evals))
    ax.set_axisbelow(True)
    ax.grid(which="both", zorder=100)
    ax.set_yscale("log")
    if scale_xlog:
        ax.set_xscale("log")
    
    if file_name:
        fig.tight_layout()
        fig.savefig(file_name)
    return long


def plot_eaf_pareto(
    data: pl.DataFrame,
    x_column: str,
    y_column: str,
    min_y: float = 0,
    max_y: float = 1,
    scale_xlog: bool = False,
    scale_ylog: bool = False,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
):
    """Plot the EAF for two arbitrary data columns. For the EAF-plot for single-objective
    optimization runs, the 'plot_eaf_single_objective' provides a simpler interface.

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        x_column (str, optional): The variable corresponding to the first objective.
        y_column (str, optional): The variable corresponding to the second objective.
        min_y (float): Minimum value for the second objective.
        max_y (float): Maximum value for the second objective.
        scale_xlog (bool, optional): Whether the first objective should be log-scaled. Defaults to False.
        scale_ylog (bool, optional): Whether the second objective should be log-scaled. Defaults to False.
        ax (matplotlib.axes._axes.Axes, optional): Axis on which to create the plot. Defaults to None.
        file_name (Optional[str], optional): If ax is not given, filename to save the plot. Defaults to None.
    """
    data_to_process = np.array(data[[x_column, y_column, "data_id"]])
    eaf_data = eaf(data_to_process[:,:-1], data_to_process[:,-1] )
    eaf_data_df = pd.DataFrame(eaf_data)
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    colors = sbs.color_palette("viridis", n_colors=eaf_data_df[2].nunique())
    eaf_data_df = eaf_data_df.sort_values(0)
    min_x = np.min(eaf_data_df[0])
    max_x = np.max(eaf_data_df[0])
    if min_y is None:
        min_y = np.min(eaf_data_df[1])
    if max_y is None:
        max_y = np.max(eaf_data_df[1])
    for i, color in zip(eaf_data_df[2].unique(), colors[::-1]):
        poly = np.array(eaf_data_df[eaf_data_df[2] == i][[0, 1]])
        # poly = np.append(poly, np.array([[max(poly[:, 0]), max(poly[:, 1])]]), axis=0)
        # poly = np.append(poly, np.array([[min(poly[:, 0]), max(poly[:, 1])]]), axis=0)
        poly = np.append(poly, np.array([[max_x, max_y]]), axis=0)
        poly = np.append(poly, np.array([[min(poly[:, 0]), max_y]]), axis=0)
        poly2 = np.repeat(poly, 2, axis=0)
        poly2[2::2, 1] = poly[:, 1][:-1]
        ax.add_patch(Polygon(poly2, facecolor=color))
    # ax.add_colorbar()
    ax.set_ylim(min_y, max_y)
    ax.set_xlim(min_x, max_x)
    ax.set_axisbelow(True)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    if scale_ylog:
        ax.set_yscale("log")
    if scale_xlog:
        ax.set_xscale("log")
    ax.grid(which="both", zorder=100)
    if file_name:
        fig.tight_layout()
        fig.savefig(file_name)

def plot_eaf_diffs(
    data1: pl.DataFrame,
    data2: pl.DataFrame,
    x_column: str,
    y_column: str,
    min_y: float = 0,
    max_y: float = 1,
    scale_xlog: bool = False,
    scale_ylog: bool = False,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
):
    """Plot the EAF differences between two datasets.

    Args:
        data1 (pl.DataFrame): The DataFrame which contains the full performance trajectory for algorithm 1. Should be generated from a DataManager.
        data2 (pl.DataFrame): The DataFrame which contains the full performance trajectory for algorithm 2. Should be generated from a DataManager.
        x_column (str, optional): The variable corresponding to the first objective.
        y_column (str, optional): The variable corresponding to the second objective.
        min_y (float): Minimum value for the second objective.
        max_y (float): Maximum value for the second objective.
        scale_xlog (bool, optional): Whether the first objective should be log-scaled. Defaults to False.
        scale_ylog (bool, optional): Whether the second objective should be log-scaled. Defaults to False.
        ax (matplotlib.axes._axes.Axes, optional): Axis on which to create the plot. Defaults to None.
        file_name (Optional[str], optional): If ax is not given, filename to save the plot. Defaults to None.
    """
    # TODO: add an approximation version to speed up plotting
    x = np.array(data1[[x_column, y_column, "data_id"]])
    y = np.array(data2[[x_column, y_column, "data_id"]])
    eaf_diff_rect = eafdiff(x, y, rectangles=True)
    color_dict = {
        k: v
        for k, v in zip(
            np.unique(eaf_diff_rect[:, -1]),
            sbs.color_palette("viridis", n_colors=len(np.unique(eaf_diff_rect[:, -1]))),
        )
    }
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    for rect in eaf_diff_rect:
        ax.add_patch(
            Rectangle(
                (rect[0], rect[1]),
                rect[2] - rect[0],
                rect[3] - rect[1],
                facecolor=color_dict[rect[-1]],
            )
        )
    if min_y is None:
        min_y = np.min(x[1])
    if max_y is None:
        max_y = np.max(x[1])
    ax.set_ylim(min_y, max_y)
    ax.set_xlim((0,1000))
    if scale_ylog:
        ax.set_yscale("log")
    if scale_xlog:
        ax.set_xscale("log")
    if file_name:
        fig.tight_layout()
        fig.savefig(file_name) 
