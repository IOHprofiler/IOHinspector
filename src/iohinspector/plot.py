import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from matplotlib.patches import Polygon, Rectangle
import seaborn as sbs
import matplotlib.pyplot as plt

font = {"size": 24}
plt.rc("font", **font)

from .manager import DataManager
from .metrics import aggegate_running_time, transform_fval, get_sequence
from .align import align_data
from .indicators import add_indicator, Final_NonDominated
from typing import Iterable, Optional
import polars as pl
import numpy as np

from moocore import eaf, eafdiff
import pandas as pd

from robustranking import Benchmark
from robustranking.comparison import MOBootstrapComparison, BootstrapComparison
from robustranking.utils.plots import plot_ci_list, plot_line_ranks

# tradeoff between simple (few parameters) and flexible. Maybe many parameter but everything with clear defaults?
# Can also make sure any useful function for data processing is available separately for more flexibility


def single_function_fixedbudget(
    data: pl.DataFrame,
    evaluation_variable: str = "evaluations",
    fval_variable: str = "raw_y",
    free_variables: Iterable[str] = ["algorithm_name"],
    f_min: float = None,
    f_max: float = None,
    max_budget: int = None,
    maximization: bool = False,
    measures: Iterable[str] = ["ERT"],
    scale_xlog: bool = True,
    scale_ylog: bool = True,
    file_name: str = None,
):
    """Create a fixed-budget plot for a given set of performance data.

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should        be generated from a DataManager.
        evaluation_variable (str, optional): Column in 'data' which corresponds to the number of evaluations. Defaults to "evaluations".
        fval_variable (str, optional): Column in 'data' which corresponds to the performance measure. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Columns in 'data' which correspond to the variables which will be used to distinguish between lines in the plot. Defaults to ["algorithm_name"].
        f_min (float, optional): Minimum value to use for the 'fval_variable', if not present the min of that column will be used. Defaults to None.
        f_max (float, optional): Maximum value to use for the 'fval_variable', if not present the max of that column will be used. Defaults to None.
        max_budget (int, optional): Maximum value to use for the 'evaluation_variable', if not present the max of that column will be used. Defaults to None.
        maximization (bool, optional): Boolean indicating whether the 'fval_variable' is being maximized. Defaults to False.
        measures (Iterable[str], optional): List of measures which should be used in the plot. Valid options are 'ERT', 'mean', 'PAR-10', 'min', 'max'. Defaults to ['ERT'].
        scale_xlog (bool, optional): Should the x-axis be log-scaled. Defaults to True.
        scale_ylog (bool, optional): Should the y-axis be log-scaled. Defaults to True.
        file_name (str, optional): Where should the resulting plot be stored. Defaults to None.

    Returns:
        pd.DataFrame: The final dataframe which was used to create the plot
    """
    dt_agg = aggegate_running_time(
        data,
        evaluation_variable=evaluation_variable,
        fval_variable=fval_variable,
        free_variables=free_variables,
        f_min=f_min,
        f_max=f_max,
        scale_flog=scale_xlog,
        max_budget=max_budget,
        maximization=maximization,
    )

    dt_molt = dt_agg.melt(id_vars=[fval_variable] + free_variables)
    dt_plot = dt_molt[dt_molt["variable"].isin(measures)]

    plt.figure(figsize=(16, 9))
    sbs.lineplot(
        dt_plot,
        x=fval_variable,
        y="value",
        style="variable",
        hue=dt_plot[free_variables].apply(tuple, axis=1),
    )
    if scale_xlog:
        plt.xscale("log")
    if scale_ylog:
        plt.yscale("log")

    if not maximization:
        plt.gca().invert_xaxis()

    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)

    return dt_plot


def single_function_fixedtarget():
    # same af fixed budget
    raise NotImplementedError


def plot_eaf_singleobj(
    data: pl.DataFrame,
    min_budget: int = None,
    max_budget: int = None,
    scale_xlog: bool = True,
    n_quantiles: int = 100,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    file_name: str = None,
):
    if min_budget is None:
        min_budget = data[eval_var].min()
    if max_budget is None:
        max_budget = data[eval_var].max()
    evals = get_sequence(min_budget, max_budget, 50, scale_xlog, True)
    long = align_data(data, np.array(evals, "uint64"), ["data_id"], output="long")

    quantiles = np.arange(0, 1 + 1 / ((n_quantiles - 1) * 2), 1 / (n_quantiles - 1))
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
    plt.ylim(long[fval_var].min(), long[fval_var].max())
    plt.xlim(min(evals), max(evals))
    ax.set_axisbelow(True)
    plt.grid(which="both", zorder=100)
    plt.yscale("log")
    if scale_xlog:
        plt.xscale("log")

    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)
    # plt.show()
    return long


def plot_eaf_pareto(
    data: pl.DataFrame,
    x_column: str,
    y_column: str,
    min_y: float = 0,
    max_y: float = 1,
    scale_xlog: bool = False,
    scale_ylog: bool = False,
    filename_fig: Optional[str] = None,
):
    """Plot the EAF for two arbitrary data columns. For the EAF-plot for single-objective
    optimization runs, the 'plot_eaf_singleobj' provides a simpler interface.

    Args:
        data (pl.DataFrame): _description_
        x_column (str): _description_
        y_column (str): _description_
    """

    eaf_data = eaf(np.array(data[[x_column, y_column, "data_id"]]))
    eaf_data_df = pd.DataFrame(eaf_data)

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
    plt.ylim(min_y, max_y)
    plt.xlim(min_x, max_x)
    ax.set_axisbelow(True)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    if scale_ylog:
        plt.yscale("log")
    if scale_xlog:
        plt.xscale("log")
    plt.grid(which="both", zorder=100)
    plt.tight_layout()
    if filename_fig:
        plt.savefig(filename_fig)
        plt.close()


def eaf_diffs(
    data1: pl.DataFrame,
    data2: pl.DataFrame,
    x_column: str,
    y_column: str,
    min_y: float = 0,
    max_y: float = 1,
    scale_xlog: bool = False,
    scale_ylog: bool = False,
    filename_fig: Optional[str] = None,
):
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
    plt.ylim(min_y, max_y)
    if scale_ylog:
        plt.yscale("log")
    if scale_xlog:
        plt.xscale("log")
    plt.tight_layout()
    if filename_fig:
        plt.savefig(filename_fig)
        plt.close()


def ecdf(
    data,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    free_vars: Iterable[str] = ["algorithm_name"],
    scale_xlog: bool = True,
    file_name: str = None,
):
    dt_plot = (
        transform_fval(data, fval_col=fval_var)
        .group_by([eval_var] + free_vars)
        .mean()
        .sort(eval_var)
    )
    plt.figure(figsize=(16, 9))
    sbs.lineplot(
        dt_plot.to_pandas(),
        x="evaluations",
        y="eaf",
        hue=dt_plot[free_vars].apply(tuple, axis=1),
    )
    if scale_xlog:
        plt.xscale("log")
    plt.grid()
    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)
    return dt_plot


def multi_function_fixedbudget():
    # either just loop over function column(s), or more advanced
    raise NotImplementedError


def multi_function_fixedtarget():
    raise NotImplementedError


def glicko2_ranking():
    # candlestick plot based on average and volatility
    raise NotImplementedError


def robustranking():
    # to decide which plot(s) to use and what exact interface to define
    raise NotImplementedError


def stats_comparison():
    # heatmap or graph of statistical comparisons
    raise NotImplementedError


def winnning_fraction_heatmap():
    # nevergrad-like heatmap
    raise NotImplementedError


def plot_paretofronts_2d(
    data: pl.DataFrame,
    obj_vars: Iterable[str] = ["raw_y", "F2"],
    free_vars: Iterable[str] = ["algorithm_name"],
):
    assert len(obj_vars) == 2

    df = add_indicator(data, Final_NonDominated(), obj_vars)

    plt.figure(figsize=(16, 9))
    sbs.scatterplot(df, x=data.obj_vars[0], y=data.obj_vars[1], hue=free_vars)
    plt.show()
    return df


def plot_indicator_over_time(
    data: pl.DataFrame,
    obj_columns: Iterable[str],
    indicator: object,
    evals_min: int = 0,
    evals_max: int = 50_000,
    nr_eval_steps: int = 50,
    free_variable: str = "algorithm_name",
    filename_fig: Optional[str] = None,
    filename_dataframe: Optional[str] = None,
):
    """_summary_

    Args:
        data (pl.DataFrame): _description_
        obj_columns (Iterable[str]): _description_
        indicator (object): _description_
        evals_min (int, optional): _description_. Defaults to 0.
        evals_max (int, optional): _description_. Defaults to 50_000.
        nr_eval_steps (int, optional): _description_. Defaults to 50.
        free_variable (str, optional): _description_. Defaults to 'algorithm_name'.
        filename_fig (Optional[str], optional): _description_. Defaults to None.
        filename_dataframe (Optional[str], optional): _description_. Defaults to None.
    """

    evals = get_sequence(
        evals_min, evals_max, nr_eval_steps, cast_to_int=True, scale_log=True
    )
    df = add_indicator(
        data, indicator, objective_columns=obj_columns, evals=evals
    ).to_pandas()

    plt.figure(figsize=(16, 9))
    sbs.lineplot(df, x="evaluations", y=indicator.var_name, hue=free_variable, palette=sbs.color_palette(n_colors=len(np.unique(data[free_variable]))))
    plt.xlabel("Evaluations")
    plt.xlim(evals_min, evals_max)
    plt.xscale("log")
    plt.grid()
    plt.tight_layout()
    if filename_fig:
        plt.savefig(filename_fig)
        plt.close()
    else:
        plt.show()
    if filename_dataframe:
        df.to_csv(filename_dataframe)


def plot_robustrank_over_time(
    data: pl.DataFrame,
    obj_columns: Iterable[str],
    evals: Iterable[int],
    indicator: object,
    filename_fig: Optional[str] = None,
):
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
    f"{eval}" : BootstrapComparison(
        Benchmark().from_pandas(dt_pivoted, "algorithm_name", "run_id", eval),
        alpha=0.05,
        minimise=indicator.minimize,
        bootstrap_runs=1000,
    ) for eval in evals
    }

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    plot_line_ranks(comparisons, ax=ax)

    plt.tight_layout()
    if filename_fig:
        plt.savefig(filename_fig)
        plt.close()
