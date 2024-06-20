import polars as pl
import numpy as np

from typing import Iterable, Callable


def get_sequence(min : float, max : float, len : float, scale_log : bool = False, cast_to_int : bool = False) -> np.ndarray:
    """Create sequence of points, used for subselecting targets / budgets for allignment and data processing

    Args:
        min (float): Starting point of the range
        max (float): Final point of the range
        len (float): Number of steps
        scale_log (bool): Whether values should be scaled logarithmically. Defaults to False
        version (str, optional): Whether the value should be casted to integers (e.g. in case of budget) or not. Defaults to False.

    Returns:
        np.ndarray: Array of evenly spaced values
    """
    transform = lambda x: x
    if scale_log:
        assert(min > 0)
        min = np.log10(min)
        max = np.log10(max)
        transform = lambda x: 10**x
    values = transform(np.arange(min, max+(max-min)/(2*(len-1)), (max-min)/(len-1), dtype=float))
    if cast_to_int:
        return np.unique(np.array(values, dtype=int))
    return np.unique(values)

def _geometric_mean(series: pl.Series) -> float:
    """Helper function for polars: geometric mean"""
    return np.exp(np.log(series).mean())


def _convergence(data : DataSet,
                 custom_op : Callable[[pl.Series], float] = None):
    """Internal function for getting fixed-budget information. 

    Args:
        data (DataSet): The data object to use for getting the performance. Note that the fval, evaluation and free variables as defined in 
        this object determine the axes of the final performance (most data will have 'raw_y', 'evaluations' and ['algId'] as defaults)
        custom_op (callable, optional): Any custom aggregation . Defaults to None.

    Returns:
        _type_: _description_
    """
    
    #Bookkeeping to ensure flexibility of analyses
    free_variables = data.free_variables
    fval_variable = data.fval_variable
    evaluation_variable = data.evaluation_variable #To think about whether this should always be int or not (cpu time?)

    #Getting alligned data (to check if e.g. limits should be args for this function)
    x_values = get_sequence(data.min_budget, data.max_budget, 50, 'budget')
    data_aligned = data.get_aligned(x_values, output='long')

    group_variables = free_variables + [evaluation_variable]
    aggregations = [
        pl.mean(fval_variable).alias('mean'),
        pl.min(fval_variable).alias('min'),
        pl.max(fval_variable).alias('max'),
        pl.median(fval_variable).alias('median'),
        pl.std(fval_variable).alias('std'),
        pl.col(fval_variable).apply(lambda s: _geometric_mean(s)).alias('geometric_mean'),
    ]

    if custom_op is not None:
        aggregations.append(pl.col(fval_variable).apply(lambda s: custom_op(s)).alias(custom_op.__name__))

    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)

    return dt_plot.sort(evaluation_variable).to_parndas() 

def _eaf_transform(long, lb = 1e-8, ub = 1e8, scale_log = True, maximization = False):
    
    if scale_log:
        lb = np.log10(lb)
        ub = np.log10(ub)
        res = long.with_columns(
            ((pl.col("raw_y").log10() - lb)/(ub-lb)).clip(0,1).alias('eaf')
        )
    else:
        res = long.with_columns(
            ((pl.col("raw_y") - lb)/(ub-lb)).clip(0,1).alias('eaf')
        )
    if maximization: 
        return res
    return res.with_columns(
        (1-pl.col("eaf")).alias('eaf')
    )

def _eaf(data : DataSet, 
                     ):
    #same as _convergence, but apply use _eaf_transform and make 'eaf' the fval_variable
    pass

def _aocc(group):
    new_row = pl.DataFrame({'evaluations': [0,max_budget], 'eaf': [group['eaf'].min(),group['eaf'].max()]})
    group = pl.concat([group, new_row], how='diagonal').sort('evaluations').filter(pl.col('evaluations') <= max_budget)
    return group.with_columns(
        ((pl.col('evaluations').diff(n=1, null_behavior="ignore") * (pl.col('eaf').shift(1)))/max_budget).alias('aocc_contribution')
    )

def get_aocc(data : DataSet, 
                     ):
    df = get_unaligned(data)
    #TODO: add column to each group with max budget

    eaf = _eaf_transform(df)
    aocc_contribs = eaf.group_by(*['run_id']).map_groups(_aocc)
    aoccs = aocc_contribs.group_by(*['run_id']).agg(pl.col('aocc_contribution').sum()).mean().drop('run_id')

    return aoccs

def plot_eaf(data):
    dt_eaf = _eaf(data).drop(['raw_y', 'run_id'])
    fig, ax = plt.subplots(figsize=(16,9))
    quantiles = np.arange(0,1+(0.05/2), 0.05)
    colors = sbs.color_palette('viridis', n_colors=len(quantiles))
    for quant, color in zip(quantiles[::-1], colors[::-1]):
        poly = np.array(dt_eaf.group_by('evaluations').quantile(0.1).sort('evaluations'))
        poly = np.append(poly, np.array([[max(poly[:,0]), 0]]), axis=0)
        poly2 = np.repeat(poly,2, axis=0)
        poly2[2::2, 1] = poly[:,1][:-1]
        ax.add_patch(Polygon(poly2, facecolor = color))
    plt.ylim(0,1)
    plt.xlim(1,10000000)
    plt.xscale('log')
    plt.show()


def _running_time(data : DataSet,
                 custom_op : Callable[[pl.Series], float] = None):
    """Internal function for getting fixed-budget information. 

    Args:
        data (DataSet): The data object to use for getting the performance. Note that the fval, evaluation and free variables as defined in 
        this object determine the axes of the final performance (most data will have 'raw_y', 'evaluations' and ['algId'] as defaults)
        custom_op (callable, optional): Any custom aggregation . Defaults to None.

    Returns:
        _type_: _description_
    """
    
    #Bookkeeping to ensure flexibility of analyses
    free_variables = data.free_variables
    eval_variable = data.eval_variable
    evaluation_variable = data.evaluation_variable #To think about whether this should always be int or not (cpu time?)

    #Getting alligned data (to check if e.g. limits should be args for this function)
    f_values = get_sequence(data.min_fval, data.max_fval, 50, 'target')
    data_aligned = data.get_aligned_target(f_values, output='long')

    budget = data.max_budget

    group_variables = free_variables + [evaluation_variable]
    aggregations = [
        pl.mean(eval_variable).alias('mean'),
        pl.min(eval_variable).alias('min'),
        pl.max(eval_variable).alias('max'),
        pl.median(eval_variable).alias('median'),
        pl.std(eval_variable).alias('std'),
        pl.col(eval_variable).is_finite().mean().alias('success_ratio'),
        pl.col(eval_variable).is_finite().sum().alias('success_count'),
        (pl.col(eval_variable).replace(np.inf, budget).sum() / pl.col(eval_variable).is_finite().sum()).alias('ERT'),
        (pl.col(eval_variable).replace(np.inf, budget * 10).sum() / pl.col(eval_variable).count()).alias('PAR-10')
    ]

    if custom_op is not None:
        aggregations.append(pl.col(eval_variable).apply(lambda s: custom_op(s)).alias(custom_op.__name__))

    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)

    return dt_plot.sort(evaluation_variable).to_parndas() 

#### Multi-objective

from pygmo import hypervolume
from scipy.spatial.distance import cdist

class Anytime_IGD:
    def __init__(self, reference_set : np.ndarray):
        self.reference_set = reference_set

    def __call__(self, group : pl.DataFrame, objective_columns : Iterable) -> pl.DataFrame:
        points = np.array(group[objective_columns])
        group = group.with_columns(
            pl.Series(name="igd", values=np.mean(np.minimum.accumulate(cdist(self.reference_set, points), axis=1), axis=0))
        )
        return group
    
class Anytime_IGDPlus:
    def __init__(self, reference_set : np.ndarray):
        self.reference_set = reference_set

    def __call__(self, group : pl.DataFrame, objective_columns : Iterable) -> pl.DataFrame:
        points = np.array(group[objective_columns])
        group = group.with_columns(
            pl.Series(name="igd+", values=np.mean(np.minimum.accumulate(cdist(self.reference_set, points, metric=lambda x,y : np.sqrt(np.clip(y-x, 0, None)**2).sum()), axis=1), axis=0))
        )
        return group
    
class Anytime_HyperVolume:
    def __init__(self, reference_point : np.ndarray):
        self.reference_point = reference_point

    def __call__(self, group : pl.DataFrame, objective_columns : Iterable, reference_set : np.ndarray) -> pl.DataFrame:
        group = group.with_columns(
            pl.Series(name="hv", values=[hypervolume(np.array(group[objective_columns])[:i]).compute((self.reference_point)) for i in range(1, len(group)+1)])
        )
        return group
    
class Anytime_NonDominated:
    def __call__(self, group : pl.DataFrame, objective_columns : Iterable):
        objectives = np.array(group[objective_columns])
        is_efficient = np.ones(objectives.shape[0], dtype = bool)
        for i, c in enumerate(objectives[1:]):
            if is_efficient[i+1]:
                is_efficient[i+1:][is_efficient[i+1:]] = np.any(objectives[i+1:][is_efficient[i+1:]]<c, axis=1)  # Keep any later point with a lower cost
                is_efficient[i+1] = True  # And keep self
        group = group.with_columns(
            pl.Series(name="nondominated", values=is_efficient)
        )
        return group
    

