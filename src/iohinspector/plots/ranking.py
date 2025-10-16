from typing import Iterable, Optional
import polars as pl
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbs
from iohinspector.metrics import get_tournament_ratings
from iohinspector.indicators import add_indicator


def plot_tournament_ranking(
    data,
    alg_vars: Iterable[str] = ["algorithm_name"],
    fid_vars: Iterable[str] = ["function_name"],
    perf_var: str = "raw_y",
    nrounds: int = 25,
    maximization: bool = False,
    ax: matplotlib.axes._axes.Axes = None,
    file_name: str = None,
):
    """Method to plot ELO ratings of a set of algorithm on a set of problems.
    Calculated based on nrounds of competition, where in each round all algorithms face all others (pairwise) on every function.
    For each round, a sampled performance value is taken from the data and used to determine the winner.

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        alg_vars (Iterable[str], optional): Which variables specific the algortihms which will compete. Defaults to ["algorithm_name"].
        fid_vars (Iterable[str], optional): Which variables denote the problems on which will be competed. Defaults to ["function_name"].
        perf_var (str, optional): Which variable corresponds to the performance. Defaults to "raw_y".
        nrounds (int, optional): How many round should be played. Defaults to 25.
        maximization (bool, optional): Whether the performance should be maximized. Defaults to False.
        ax (matplotlib.axes._axes.Axes, optional): Existing matplotlib axis object to draw the plot on.
        file_name (str, optional): Where should the resulting plot be stored. Defaults to None. If existing axis is provided, this functionality is disabled.

    Returns:
        pd.DataFrame: pandas dataframe of the exact data used to create the plot
    """
    # candlestick plot based on average and volatility
    dt_elo = get_tournament_ratings(
        data, alg_vars, fid_vars, perf_var, nrounds, maximization
    )
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

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
    ax.grid()

    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)
    return dt_elo


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
    obj_columns: Iterable[str],
    evals: Iterable[int],
    indicator: object,
    filename_fig: Optional[str] = None,
):
    """Plot robust ranking at distinct timesteps

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        obj_columns (Iterable[str], optional): Which columns in 'data' correspond to the objectives.
        evals (Iterable[int]): Timesteps at which to get the rankings
        indicator (object): Indicator object from iohinspector.indicators
        filename_fig (str, optional): Where should the resulting plot be stored. Defaults to None. If existing axis is provided, this functionality is disabled.
    """
    from robustranking import Benchmark
    from robustranking.comparison import MOBootstrapComparison, BootstrapComparison
    from robustranking.utils.plots import plot_ci_list, plot_line_ranks

    df = add_indicator(
        data, indicator, objective_columns=obj_columns, evals=evals
    ).to_pandas()
    df_part = df[["evaluations", indicator.var_name, "algorithm_name", "run_id"]]
    dt_pivoted = pd.pivot(
        df_part,
        index=["algorithm_name", "run_id"],
        columns=["evaluations"],
        values=[indicator.var_name],
    ).reset_index()
    dt_pivoted.columns = ["algorithm_name", "run_id"] + evals
    benchmark = Benchmark()
    benchmark.from_pandas(dt_pivoted, "algorithm_name", "run_id", evals)
    comparison = MOBootstrapComparison(
        benchmark,
        alpha=0.05,
        minimise=indicator.minimize,
        bootstrap_runs=1000,
        aggregation_method=np.mean,
    )
    fig, axs = plt.subplots(1, 4, figsize=(16, 9), sharey=True)
    for ax, runtime in zip(axs.ravel(), benchmark.objectives):
        plot_ci_list(comparison, objective=runtime, ax=ax)
        if runtime != evals[0]:
            ax.set_ylabel("")
        if runtime != evals[-1]:
            ax.get_legend().remove()
        ax.set_title(runtime)

    plt.tight_layout()
    if filename_fig:
        plt.savefig(filename_fig)
        plt.close()


def plot_robustrank_changes(
    data: pl.DataFrame,
    obj_columns: Iterable[str],
    evals: Iterable[int],
    indicator: object,
    filename_fig: Optional[str] = None,
):
    """Plot robust ranking changes at distinct timesteps

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        obj_columns (Iterable[str], optional): Which columns in 'data' correspond to the objectives.
        evals (Iterable[int]): Timesteps at which to get the rankings
        indicator (object): Indicator object from iohinspector.indicators
        filename_fig (str, optional): Where should the resulting plot be stored. Defaults to None. If existing axis is provided, this functionality is disabled.
    """
    from robustranking import Benchmark
    from robustranking.comparison import MOBootstrapComparison, BootstrapComparison
    from robustranking.utils.plots import plot_ci_list, plot_line_ranks

    df = add_indicator(
        data, indicator, objective_columns=obj_columns, evals=evals
    ).to_pandas()
    df_part = df[["evaluations", indicator.var_name, "algorithm_name", "run_id"]]
    dt_pivoted = pd.pivot(
        df_part,
        index=["algorithm_name", "run_id"],
        columns=["evaluations"],
        values=[indicator.var_name],
    ).reset_index()
    dt_pivoted.columns = ["algorithm_name", "run_id"] + evals

    comparisons = {
        f"{eval}": BootstrapComparison(
            Benchmark().from_pandas(dt_pivoted, "algorithm_name", "run_id", eval),
            alpha=0.05,
            minimise=indicator.minimize,
            bootstrap_runs=1000,
        )
        for eval in evals
    }

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plot_line_ranks(comparisons, ax=ax)

    plt.tight_layout()
    if filename_fig:
        plt.savefig(filename_fig)
        plt.close()
