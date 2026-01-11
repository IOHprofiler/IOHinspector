from typing import Iterable, Optional

import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs

from iohinspector.metrics.ranking import get_robustrank_changes, get_robustrank_over_time
from iohinspector.plots.utils import BasePlotArgs, _create_plot_args, _save_fig
from iohinspector.metrics import get_tournament_ratings


def plot_tournament_ranking(
    data,
    alg_vars: Iterable[str] = ["algorithm_name"],
    fid_vars: Iterable[str] = ["function_name"],
    fval_var: str = "raw_y",
    nrounds: int = 25,
    maximization: bool = False,
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str = None,
    plot_args: dict | BasePlotArgs = None,
):
    """Plot ELO ratings from tournament-style algorithm competition across multiple problems.

    Creates a point plot with error bars showing ELO ratings calculated from pairwise algorithm 
    competitions. In each round, all algorithms compete against each other on every function,
    with performance samples determining winners and ELO rating updates.

    Args:
        data (pl.DataFrame): Input dataframe containing algorithm performance trajectory data.
        alg_vars (Iterable[str], optional): Which columns contain the algorithm identifiers that will compete. 
            Defaults to ["algorithm_name"].
        fid_vars (Iterable[str], optional): Which columns contain the problem/function identifiers for competition. 
            Defaults to ["function_name"].
        fval_var (str, optional): Which column contains the performance values. Defaults to "raw_y".
        nrounds (int, optional): Number of tournament rounds to simulate. Defaults to 25.
        maximization (bool, optional): Whether the performance should be maximized. Defaults to False.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (str, optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | BasePlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "Tournament Ranking".
            - xlabel (str): X-axis label. Defaults to "Algorithms".
            - ylabel (str): Y-axis label. Defaults to "ELO Rating".
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other BasePlotArgs parameters (xlim, ylim, xscale, yscale, grid, legend, fontsize, etc.).

    Returns:
        tuple[matplotlib.axes.Axes, pd.DataFrame]: The matplotlib axes object and the ELO ratings 
            dataframe used to create the plot.
    """
    # candlestick plot based on average and volatility
    dt_elo = get_tournament_ratings(
        data, alg_vars, fid_vars, fval_var, nrounds, maximization
    )

    plot_args = _create_plot_args(
        BasePlotArgs(
            title= "Tournament Ranking",
            xlabel="Algorithms",
            ylabel="ELO Rating",
            grid= True
        ),
        plot_args
    )


    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_args.figsize)
    else:
        fig = None

    sbs.pointplot(data=dt_elo, x=alg_vars[0], y="Rating", linestyle="none", ax=ax)

    ax.errorbar(
        dt_elo[alg_vars[0]],
        dt_elo["Rating"],
        yerr=dt_elo["Deviation"],
        fmt="o",
        color="blue",
        alpha=0.6,
        capsize=5,
        elinewidth=1.5,
    )
    
    plot_args.apply(ax)
    
    _save_fig(fig, file_name, plot_args)


    return ax, dt_elo


def robustranking():
    # to decide which plot(s) to use and what exact interface to define
    raise NotImplementedError()


def stats_comparison():
    # heatmap or graph of statistical comparisons
    raise NotImplementedError()


def winnning_fraction_heatmap():
    # nevergrad-like heatmap
    raise NotImplementedError()




def plot_robustrank_over_time(
    data: pl.DataFrame,
    obj_vars: Iterable[str],
    evals: Iterable[int],
    indicator: object,
    *,
    file_name: Optional[str] = None,
):
    """Plot robust ranking confidence intervals at distinct evaluation timesteps.

    Creates multiple subplots showing robust ranking analysis with confidence intervals 
    for algorithm performance at different evaluation budgets, using statistical comparison
    methods to handle uncertainty in performance measurements.

    Args:
        data (pl.DataFrame): Input dataframe containing algorithm performance trajectory data. 
            Must contain data for a single function only.
        obj_vars (Iterable[str]): Which columns contain the objective values for ranking calculation.
        evals (Iterable[int]): Evaluation timesteps at which to compute and plot rankings.
        indicator (object): Quality indicator object from iohinspector.indicators module.
        file_name (Optional[str], optional): Path to save the plot. If None, plot is not saved. Defaults to None.

    Returns:
        tuple[np.ndarray, tuple]: Array of matplotlib axes objects and a tuple containing 
            (comparison, benchmark) data used for the robust ranking analysis.

    Raises:
        ValueError: If data contains multiple functions (function_id has more than one unique value).
    """
    from robustranking.utils.plots import plot_ci_list

    if(data["function_id"].n_unique() > 1):
        raise ValueError("Robust ranking over time plot can only be generated for a single function at a time.")
    
    comparison, benchmark = get_robustrank_over_time(
        data=data,
        obj_vars=obj_vars,
        evals=evals,
        indicator=indicator,
    )

    plot_args =BasePlotArgs(
        figsize=(5*len(evals), 5),
    )
        
    
    fig, axs = plt.subplots(1, len(evals), figsize=plot_args.figsize, sharey=True)

    for ax, runtime in zip(axs.ravel(), benchmark.objectives):
        plot_ci_list(comparison, objective=runtime, ax=ax)
        if runtime != evals[0]:
            ax.set_ylabel("")
        if runtime != evals[-1]:
            ax.get_legend().remove()
        ax.set_title(runtime)

    _save_fig(fig, file_name, plot_args)

    return axs, comparison, benchmark

def plot_robustrank_changes(
    data: pl.DataFrame,
    obj_vars: Iterable[str],
    evals: Iterable[int],
    indicator: object,
    *,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: Optional[str] = None,
):
    """Plot robust ranking changes over evaluation timesteps as connected line plots.

    Creates a line plot showing how algorithm rankings evolve over time, with lines 
    connecting ranking positions across different evaluation budgets to visualize
    ranking stability and performance trajectory changes.

    Args:
        data (pl.DataFrame): Input dataframe containing algorithm performance trajectory data.
        obj_vars (Iterable[str]): Which columns contain the objective values for ranking calculation.
        evals (Iterable[int]): Evaluation timesteps at which to compute rankings and plot changes.
        indicator (object): Quality indicator object from iohinspector.indicators module.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. Defaults to None.
        file_name (Optional[str], optional): Path to save the plot. If None, plot is not saved. Defaults to None.

    Returns:
        tuple[matplotlib.axes.Axes, object]: The matplotlib axes object and the ranking 
            comparisons data used to create the plot.
    """
    from robustranking.utils.plots import plot_line_ranks

    comparisons = get_robustrank_changes(
        data=data,
        obj_vars=obj_vars,
        evals=evals,
        indicator=indicator,
    )

    plot_args = BasePlotArgs(
        figsize=(max(5 * len(evals), 16), 5),
    )
    

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=plot_args.figsize)
    else:
        fig = None
    plot_line_ranks(comparisons, ax=ax)

    plot_args.apply(ax)
    _save_fig(fig, file_name, plot_args)

    return ax, comparisons
