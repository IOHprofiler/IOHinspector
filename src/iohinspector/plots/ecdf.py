import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
import polars as pl
from typing import Iterable, Optional
from iohinspector.metrics import get_data_ecdf
from iohinspector.plots.utils import LinePlotArgs, _create_plot_args, _save_fig

def plot_ecdf(
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    free_vars: Iterable[str] = ["algorithm_name"],
    maximization: bool = False,
    f_min: int = None,
    f_max: int = None,
    scale_f_log: bool = True,
    eval_values: Iterable[int] = None,
    eval_min: int = None,
    eval_max: int = None,
    scale_eval_log: bool = True,
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
    plot_args: dict | LinePlotArgs = None,
):
    """Plot Empirical Cumulative Distribution Function (ECDF) based on Empirical Attainment Functions.

    Creates line plots showing the cumulative probability of achieving different performance levels
    at various evaluation budgets, allowing comparison between algorithms or configurations.

    Args:
        data (pl.DataFrame): Input dataframe containing optimization algorithm trajectory data.
        fval_var (str, optional): Which column contains the function/performance values. Defaults to "raw_y".
        eval_var (str, optional): Which column contains the evaluation counts. Defaults to "evaluations".
        free_vars (Iterable[str], optional): Which columns contain the grouping variables for distinguishing 
            between different lines in the plot. Defaults to ["algorithm_name"].
        maximization (bool, optional): Whether the optimization problem is maximization. Defaults to False.
        f_min (int, optional): Minimum function value bound. If None, uses data minimum. Defaults to None.
        f_max (int, optional): Maximum function value bound. If None, uses data maximum. Defaults to None.
        scale_f_log (bool, optional): Whether function values should be log-scaled before normalization. Defaults to True.
        eval_values (Iterable[int], optional): Specific evaluation points to plot. If None, uses eval_min/eval_max 
            with scale_eval_log to sample points. Defaults to None.
        eval_min (int, optional): Minimum evaluation bound. If None, uses data minimum. Defaults to None.
        eval_max (int, optional): Maximum evaluation bound. If None, uses data maximum. Defaults to None.
        scale_eval_log (bool, optional): Whether the evaluation axis should be log-scaled. Defaults to True.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (Optional[str], optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | LinePlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "ECDF".
            - xlabel (str): X-axis label. Defaults to eval_var value.
            - ylabel (str): Y-axis label. Defaults to "eaf".
            - xscale (str): X-axis scale ("log" or "linear"). Defaults to "log" if scale_eval_log=True.
            - yscale (str): Y-axis scale ("log" or "linear"). Defaults to "log" if scale_f_log=True.
            - line_colors (Sequence[str]): Colors for different lines. Defaults to seaborn palette.
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other LinePlotArgs parameters (xlim, ylim, grid, legend, fontsize, etc.).

    Returns:
        tuple[matplotlib.axes.Axes, pd.DataFrame]: The matplotlib axes object and the processed 
            dataframe used to create the plot.
    """
    

    dt_plot = get_data_ecdf(
        data,
        fval_var=fval_var,
        eval_var=eval_var,
        free_vars=free_vars,
        maximization=maximization,
        f_min=f_min,
        f_max=f_max,
        scale_f_log=scale_f_log,
        eval_values=eval_values,
        eval_max=eval_max,
        eval_min=eval_min,
        scale_eval_log=scale_eval_log,
        turbo=True
    )

    plot_args = _create_plot_args(
        LinePlotArgs(
            xlabel= eval_var,
            ylabel= "eaf",
            title= "ECDF",
            xscale= "log" if scale_eval_log else "linear",
            yscale= "log" if scale_f_log else "linear",
        ),
        plot_args
    )


    dt_plot.sort_values(free_vars)
    if ax is None:
        fig, ax = plt.subplots(figsize=plot_args.figsize)

    if len(free_vars) == 1:
        hue_arg = free_vars[0]
        style_arg = free_vars[0]
    else:
        style_arg = free_vars[0]
        hue_arg = dt_plot[free_vars[1:]].apply(tuple, axis=1)

    
    sbs.lineplot(
        dt_plot,
        x= eval_var,
        y="eaf",
        style=style_arg,
        hue=hue_arg,
        palette=plot_args.line_colors,
        ax=ax,
    )

    plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args=plot_args)

    return ax, dt_plot