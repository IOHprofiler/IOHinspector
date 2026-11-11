import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
from typing import Iterable, Sequence
from .._upgrade import upgrade_hints
from ..metrics.fixed_target import aggregate_running_time
from .utils import LinePlotArgs, _create_plot_args, _hue, _save_fig


@upgrade_hints
def plot_single_function_fixed_target(
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    by: Sequence[str] = ("algorithm_name",),
    f_min: float = None,
    f_max: float = None,
    scale_f_log: bool = True,
    eval_max: int = None,
    maximization: bool = False,
    measures: Iterable[str] = ["ERT"],
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str = None,
    plot_args: dict | LinePlotArgs = None,
):
    """Create a fixed-target plot showing Expected Running Time (ERT) analysis for algorithm performance.

    Visualizes how much computational budget (evaluations) algorithms need to reach specific target
    performance levels, allowing comparison of algorithm efficiency across different difficulty targets.

    Args:
        data (pl.DataFrame): Input dataframe containing optimization algorithm trajectory data.
        eval_var (str, optional): Which column contains the evaluation counts. Defaults to "evaluations".
        fval_var (str, optional): Which column contains the function/objective values. Defaults to "raw_y".
        by (Sequence[str], optional): The columns whose values tell the lines apart:
            runs
            columns or descriptors. Defaults to ["algorithm_name"].
        f_min (float, optional): Minimum function value bound for target range. If None, uses data minimum. Defaults to None.
        f_max (float, optional): Maximum function value bound for target range. If None, uses data maximum. Defaults to None.
        scale_f_log (bool, optional): Whether function values should be log-scaled for target sampling. Defaults to True.
        eval_max (int, optional): Maximum evaluation budget to consider. If None, uses data maximum. Defaults to None.
        maximization (bool, optional): Whether the optimization problem is maximization. Defaults to False.
        measures (Iterable[str], optional): Running time measures to plot. Valid options are "ERT" (Expected Running Time),
            "mean", "PAR-10", "min", "max". Defaults to ["ERT"].
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (str, optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | LinePlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "Fixed-Target Plot".
            - xlabel (str): X-axis label. Defaults to fval_var value.
            - ylabel (str): Y-axis label. Defaults to "value".
            - xscale (str): X-axis scale. Defaults to "log".
            - yscale (str): Y-axis scale ("log" or "linear"). Defaults to "log" if scale_f_log=True.
            - reverse_xaxis (bool): Whether to reverse x-axis. Defaults to True for minimization, False for maximization.
            - line_colors (Sequence[str]): Colors for different lines. Defaults to seaborn palette.
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other LinePlotArgs parameters (xlim, ylim, grid, legend, fontsize, etc.).

    Returns:
        tuple[matplotlib.axes.Axes, pl.DataFrame]: The matplotlib axes object and the
            processed
            (melted/filtered) dataframe used to create the plot.
    """

    dt_agg = aggregate_running_time(
        data,
        eval_var=eval_var,
        fval_var=fval_var,
        by=by,
        f_min=f_min,
        f_max=f_max,
        scale_f_log=scale_f_log,
        eval_max=eval_max,
        maximization=maximization,
    )

    dt_plot = (
        dt_agg.unpivot(index=[fval_var, *by])
        .filter(pl.col("variable").is_in(list(measures)))
        .sort(by, maintain_order=True)
    )
    pdf = dt_plot.to_pandas()

    plot_args = _create_plot_args(
        LinePlotArgs(
            xlabel=fval_var,
            title="Fixed-Target Plot",
            xscale="log",
            yscale="log" if scale_f_log else "linear",
            reverse_xaxis=not maximization,
        ),
        plot_args,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_args.figsize)
    else:
        fig = None

    sbs.lineplot(
        pdf,
        x=fval_var,
        y="value",
        style="variable",
        hue=_hue(pdf, by),
        palette=plot_args.line_colors,
        ax=ax,
    )

    plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args=plot_args)

    return ax, dt_plot
