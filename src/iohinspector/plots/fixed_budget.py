from collections.abc import Iterable, Sequence

import matplotlib
import polars as pl

from .._upgrade import upgrade_hints
from ..metrics.fixed_budget import aggregate_convergence
from .utils import LinePlotArgs, _canvas, _create_plot_args, _measure_lines, _measures


@upgrade_hints
def plot_single_function_fixed_budget(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    by: Sequence[str] = ("algorithm_name",),
    eval_min: float | None = None,
    eval_max: float | None = None,
    maximization: bool = False,
    measures: Iterable[str] = ["geometric_mean"],
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str | None = None,
    plot_args: dict | LinePlotArgs = None,
):
    """Create a fixed-budget convergence plot showing algorithm performance over evaluation budgets.

    Newer: the figure kind "convergence" of ``iohinspector.figures`` draws the median and quartiles over the budgets (and ``"fixed_budget"`` one budget across functions), for
    publication and the website: ``render(FigureSpec(panels=[Panel("convergence")]), data)``.
    This function stays as it is.

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
    dt_plot = _measures(dt_agg, eval_var, by, measures)
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
    with _canvas(ax, plot_args, file_name) as ax:
        _measure_lines(pdf, eval_var, by, plot_args, ax)

    return ax, dt_plot
