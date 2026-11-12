import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
import polars as pl
from typing import Iterable, Sequence
from .._upgrade import upgrade_hints
from ..metrics.fixed_budget import aggregate_convergence
from .utils import (
    LinePlotArgs,
    _check_measures,
    _create_plot_args,
    _hue,
    _save_fig,
)


@upgrade_hints
def plot_single_function_fixed_budget(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    by: Sequence[str] = ("algorithm_name",),
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
        by (Sequence[str], optional): The columns whose values tell the lines apart:
            runs
            columns or descriptors. Defaults to ["algorithm_name"].
        eval_min (float, optional): Minimum evaluation bound for the plot. If None, uses data minimum. Defaults to None.
        eval_max (float, optional): Maximum evaluation bound for the plot. If None, uses data maximum. Defaults to None.
        maximization (bool, optional): Whether the optimization problem is maximization. Defaults to False.
        measures (Iterable[str], optional): Aggregate measures to plot: columns of
            ``aggregate_convergence``, such as "geometric_mean", "mean", "median",
            "best", "worst" or "q50". Defaults to ["geometric_mean"].
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
        tuple[matplotlib.axes.Axes, pl.DataFrame]: The matplotlib axes object and the
            processed
            (melted/filtered) dataframe used to create the plot.
    """
    dt_agg = aggregate_convergence(
        data,
        eval_var=eval_var,
        fval_var=fval_var,
        by=by,
        eval_min=eval_min,
        eval_max=eval_max,
        maximization=maximization,
    )
    _check_measures(dt_agg, measures)
    dt_plot = (
        dt_agg.unpivot(index=[eval_var, *by])
        .filter(pl.col("variable").is_in(list(measures)))
        .sort(by, maintain_order=True)
    )
    pdf = dt_plot.to_pandas()

    plot_args = _create_plot_args(
        LinePlotArgs(
            xlabel=eval_var,
            ylabel=fval_var,
            title="Fixed-Budget Plot",
            xscale="log",
            yscale="log",
        ),
        plot_args,
    )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_args.figsize)
    else:
        fig = None

    sbs.lineplot(
        pdf,
        x=eval_var,
        y="value",
        style="variable",
        hue=_hue(pdf, by),
        palette=plot_args.line_colors,
        ax=ax,
    )

    ax = plot_args.apply(ax=ax)

    _save_fig(fig, file_name, plot_args=plot_args)

    return ax, dt_plot
