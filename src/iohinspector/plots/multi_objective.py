from typing import Iterable, Optional, cast
import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
from iohinspector.indicators import final, add_indicator
from iohinspector.metrics import get_sequence

def plot_paretofronts_2d(
    data: pl.DataFrame,
    obj_vars: Iterable[str] = ["raw_y", "F2"],
    free_var: str = "algorithm_name",
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str = None,
):
    """Very basic plot to visualize pareto fronts

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        obj_vars (Iterable[str], optional): Which variables (length should be 2) to use for plotting. Defaults to ["raw_y", "F2"].
        free_vars (Iterable[str], optional): Which varialbes should be used to distinguish between categories. Defaults to ["algorithm_name"].
        ax (matplotlib.axes._axes.Axes, optional): Existing matplotlib axis object to draw the plot on.
        file_name (str, optional): Where should the resulting plot be stored. Defaults to None. If existing axis is provided, this functionality is disabled.

    Returns:
        pd.DataFrame: pandas dataframe of the exact data used to create the plot
    """
    assert len(obj_vars) == 2

    df = add_indicator(data, final.NonDominated(), obj_vars)
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    sbs.scatterplot(df.filter(pl.col("final_nondominated") == True), x=obj_vars[0], y=obj_vars[1], hue=free_var, ax=ax)
    if file_name:
        fig.tight_layout()
        fig.savefig(file_name)
    return df


def plot_indicator_over_time(
    data: pl.DataFrame,
    obj_columns: Iterable[str] =  ["raw_y", "F2"],
    indicator: object = None,
    eval_column: str = "evaluations",
    evals_min: int = 1,
    evals_max: int = 50_000,
    nr_eval_steps: int = 50,
    eval_scale_log: bool = True,
    free_variable: str = "algorithm_name",
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
):
    """Convenience function to plot the anytime performance of a single indicator.

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        obj_columns (Iterable[str], optional): Which columns in 'data' correspond to the objectives.
        indicator (object): Indicator object from iohinspector.indicators
        eval_column (Iterable[str], optional): Which columns in 'data' correspond to the objectives. Defaults to 'evaluations'.
        evals_min (int, optional): Lower bound for eval_column. Defaults to 0.
        evals_max (int, optional): Upper bound for eval_column. Defaults to 50_000.
        nr_eval_steps (int, optional): Number of steps between lower and upper bounds of eval_column. Defaults to 50.
        free_variable (str, optional): Variable which corresponds to category to differentiate in the plot. Defaults to 'algorithm_name'.
        ax (matplotlib.axes._axes.Axes, optional): Existing matplotlib axis object to draw the plot on.
        file_name (str, optional): Where should the resulting plot be stored. Defaults to None. If existing axis is provided, this functionality is disabled.
    """

    evals = get_sequence(
        evals_min, evals_max, nr_eval_steps, cast_to_int=True, scale_log=eval_scale_log
    )
    df = add_indicator(
        data, indicator, objective_columns=obj_columns, evals=evals
    ).to_pandas()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    sbs.lineplot(
        df,
        x=eval_column,
        y=indicator.var_name,
        hue=free_variable,
        palette=sbs.color_palette(n_colors=len(np.unique(data[free_variable]))),
        ax=ax,
    )
    ax.set_xlabel(eval_column)
    ax.set_xlim(evals_min, evals_max)
    ax.set_xscale("log")
    ax.grid()
    if file_name:
        fig.tight_layout()
        fig.savefig(file_name)

    return df
