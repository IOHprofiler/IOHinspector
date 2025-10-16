import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
import polars as pl
from typing import Iterable, Optional
import numpy as np


def plot_heatmap_single_run(
    data: pl.DataFrame,
    var_cols: Iterable[str],
    eval_col: str = "evaluations",
    scale_xlog: bool = True,
    x_mins: Iterable[float] = [-5],
    x_maxs: Iterable[float] = [5],
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
):
    """Create a heatmap showing the search space points evaluated in a single run

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        var_cols (Iterable[str]): The variables which correspond to the searchspace variable columns
        eval_col (str): The variable corresponding to evaluations. Defaults to 'evaluations'
        scale_xlog (bool, optional): Whether the evaluations should be log-scaled. Defaults to True.
        x_mins (Iterable[float], optional): Minimum bound for the variables. Should be of the same length as 'var_cols'. Defaults to [-5].
        x_maxs (Iterable[float], optional): Maximum bound for the variables. Should be of the same length as 'var_cols'.. Defaults to [5].
        ax (matplotlib.axes._axes.Axes, optional): Axis on which to create the plot. Defaults to None.
        file_name (Optional[str], optional): If ax is not given, filename to save the plot. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe of the exact data used to create the plot
    """
    assert data["data_id"].n_unique() == 1
    dt_plot = data[var_cols].transpose().to_pandas()
    dt_plot.columns = list(data[eval_col])
    x_mins_arr = np.array(x_mins)
    x_maxs_arr = np.array(x_maxs)
    dt_plot = (dt_plot.subtract(x_mins_arr, axis=0)).divide(x_maxs_arr - x_mins_arr, axis=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(32, 9))
    sbs.heatmap(dt_plot, cmap="viridis", vmin=0, vmax=1, ax=ax)
    if scale_xlog:
        ax.set_xscale("log")
        ax.set_xlim(1, len(data))

    if file_name:
        fig.tight_layout()
        fig.savefig(file_name)
    return dt_plot
