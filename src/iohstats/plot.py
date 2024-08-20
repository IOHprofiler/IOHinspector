import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from matplotlib.patches import Polygon
import seaborn as sbs
import matplotlib.pyplot as plt
font = {'size'   : 24}
plt.rc('font', **font)

from .manager import DataManager
from .metrics import aggegate_running_time, transform_fval, get_sequence
from .align import align_data
from .indicators import add_indicator, Final_NonDominated
from typing import Iterable
import polars as pl
import numpy as np

#tradeoff between simple (few parameters) and flexible. Maybe many parameter but everything with clear defaults?
#Can also make sure any useful function for data processing is available separately for more flexibility

def single_function_fixedbudget(
        data : pl.DataFrame, 
        evaluation_variable: str = "evaluations",
        fval_variable: str = "raw_y",
        free_variables: Iterable[str] = ["algorithm_name"],
        f_min: float = None,
        f_max: float = None,
        max_budget: int = None,
        maximization: bool = False,
        measures : Iterable[str] = ['ERT'],
        scale_xlog : bool = True, 
        scale_ylog : bool = True,
        file_name : str = None

):
    dt_agg = aggegate_running_time(data, 
                                   evaluation_variable = evaluation_variable,
                                   fval_variable = fval_variable,
                                   free_variables = free_variables,
                                   f_min = f_min,
                                   f_max = f_max,
                                   scale_flog = scale_xlog,
                                   max_budget = max_budget,
                                   maximization = maximization)
    
    dt_molt = dt_agg.melt(id_vars=[fval_variable] + free_variables)
    dt_plot = dt_molt[dt_molt['variable'].isin(measures)]

    plt.figure(figsize=(16,9))
    sbs.lineplot(dt_plot, x=fval_variable, y='value', style='variable', hue=dt_plot[free_variables].apply(tuple, axis=1))
    if scale_xlog:
        plt.xscale('log')
    if scale_ylog:
        plt.yscale('log')

    if not maximization:
        plt.gca().invert_xaxis()

    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)

    return dt_plot

def single_function_fixedtarget():
    #same af fixed budget
    raise NotImplementedError

def eaf(data : pl.DataFrame,
        min_budget : int = None, 
        max_budget : int = None, 
        scale_xlog : bool = True,
        n_quantiles : int = 100,
        eval_var : str = 'evaluations',
        fval_var : str = 'raw_y',
        file_name : str = None,
        ):
    
    if min_budget is None:
        min_budget = data[eval_var].min()
    if max_budget is None:
        max_budget = data[eval_var].max()
    evals = get_sequence(min_budget, max_budget, 50, scale_xlog, True)
    long = align_data(data, np.array(evals, 'uint64'), ['data_id'], output='long')

    quantiles = np.arange(0,1+1/((n_quantiles-1)*2), 1/(n_quantiles-1))
    fig, ax = plt.subplots(figsize=(16,9))
    colors = sbs.color_palette('viridis', n_colors=len(quantiles))
    for quant, color in zip(quantiles, colors[::-1]):
        poly = np.array(long.group_by(eval_var).quantile(quant).sort(eval_var)[eval_var, fval_var])
        poly = np.append(poly, np.array([[max(poly[:,0]), long[fval_var].max()]]), axis=0)
        poly = np.append(poly, np.array([[min(poly[:,0]), long[fval_var].max()]]), axis=0)
        poly2 = np.repeat(poly,2, axis=0)
        poly2[2::2, 1] = poly[:,1][:-1]
        ax.add_patch(Polygon(poly2, facecolor = color))
    plt.ylim(long[fval_var].min(),long[fval_var].max())
    plt.xlim(min(evals),max(evals))
    ax.set_axisbelow(True)
    plt.grid(which='both', zorder=100)
    plt.yscale('log')
    if scale_xlog:
        plt.xscale('log')

    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)
    # plt.show()
    return long

def eaf_diffs():
    #either use moocore or the approximation version
    raise NotImplementedError

def ecdf(data,
         fval_var : str = 'raw_y', 
         eval_var : str = 'evaluations',
         free_vars : Iterable[str] = ['algorithm_name'],
         scale_xlog : bool = True,
         file_name : str = None):
    dt_plot = transform_fval(data, fval_col=fval_var).group_by([eval_var] + free_vars).mean().sort(eval_var)
    plt.figure(figsize=(16,9))
    sbs.lineplot(dt_plot.to_pandas(), x='evaluations', y='eaf', hue=dt_plot[free_vars].apply(tuple, axis=1))
    if scale_xlog:
        plt.xscale('log')
    plt.grid()
    if file_name:
        plt.tight_layout()
        plt.savefig(file_name)
    return dt_plot

def multi_function_fixedbudget():
    #either just loop over function column(s), or more advanced
    raise NotImplementedError

def multi_function_fixedtarget():
    raise NotImplementedError

def glicko2_ranking():
    #candlestick plot based on average and volatility
    raise NotImplementedError

def robustranking():
    #to decide which plot(s) to use and what exact interface to define
    raise NotImplementedError

def stats_comparison():
    #heatmap or graph of statistical comparisons
    raise NotImplementedError

def winnning_fraction_heatmap():
    #nevergrad-like heatmap
    raise NotImplementedError



def plot_paretofronts_2d(data : pl.DataFrame,
                        obj_vars : Iterable[str] = ['raw_y', 'F2'],
                        free_vars : Iterable[str] = ['algorithm_name']):
    assert len(obj_vars) == 2


    df = add_indicator(data, Final_NonDominated(), obj_vars)

    plt.figure(figsize=(16,9))
    sbs.scatterplot(df, x=data.obj_vars[0], y=data.obj_vars[1],
                    hue=free_vars)
    plt.show()
    return df
