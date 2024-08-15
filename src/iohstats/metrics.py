import polars as pl
import numpy as np

from typing import Iterable, Callable

from functools import partial
from .align import align_data

import pandas as pd

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


def aggegate_convergence(data : pl.DataFrame,
                          evaluation_variable : str = 'evaluations',
                          fval_variable : str = 'raw_y',
                          free_variables : Iterable[str] = ['algorithm_name'],
                          f_min : float = None,
                          f_max : float = None,
                          max_budget : int = None,
                 custom_op : Callable[[pl.Series], float] = None,
                 return_as_pandas : bool = True):
    """Internal function for getting fixed-budget information. 

    Args:
        data (DataSet): The data object to use for getting the performance. Note that the fval, evaluation and free variables as defined in 
        this object determine the axes of the final performance (most data will have 'raw_y', 'evaluations' and ['algId'] as defaults)
        custom_op (callable, optional): Any custom aggregation . Defaults to None.

    Returns:
        _type_: _description_
    """
    
    #Getting alligned data (to check if e.g. limits should be args for this function)
    if f_min is None:
        f_min = data[evaluation_variable].min()
    if f_max is None:
        f_max = data[evaluation_variable].max()
    f_values = get_sequence(f_min, f_max, 50, scale_log=True)
    group_variables = free_variables + [evaluation_variable]
    data_aligned = align_data(data.cast({evaluation_variable : pl.Int64}), f_values, group_cols=['data_id'] + free_variables, x_col=evaluation_variable, y_col=fval_variable, maximization=True)

    aggregations = [
        pl.mean(fval_variable).alias('mean'),
        pl.min(fval_variable).alias('min'),
        pl.max(fval_variable).alias('max'),
        pl.median(fval_variable).alias('median'),
        pl.std(fval_variable).alias('std'),
        pl.col(fval_variable).map_elements(lambda s: _geometric_mean(s), return_dtype=pl.Float64).alias('geometric_mean'),
    ]

    if custom_op is not None:
        aggregations.append(pl.col(evaluation_variable).map_elements(lambda s: custom_op(s), return_dtype=pl.Float64).alias(custom_op.__name__))
    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    if return_as_pandas:
        return dt_plot.sort(evaluation_variable).to_pandas() 
    return dt_plot.sort(evaluation_variable)

def transform_fval(data, lb = 1e-8, ub = 1e8, scale_log = True, maximization = False, fval_col = 'raw_y'):
    
    if scale_log:
        lb = np.log10(lb)
        ub = np.log10(ub)
        res = data.with_columns(
            ((pl.col(fval_col).log10() - lb)/(ub-lb)).clip(0,1).alias('eaf')
        )
    else:
        res = data.with_columns(
            ((pl.col(fval_col) - lb)/(ub-lb)).clip(0,1).alias('eaf')
        )
    if maximization: 
        return res
    return res.with_columns(
        (1-pl.col("eaf")).alias('eaf')
    )


def _aocc(group):
    new_row = pl.DataFrame({'evaluations': [0,max_budget], 'eaf': [group['eaf'].min(),group['eaf'].max()]})
    group = pl.concat([group, new_row], how='diagonal').sort('evaluations').filter(pl.col('evaluations') <= max_budget)
    return group.with_columns(
        ((pl.col('evaluations').diff(n=1, null_behavior="ignore") * (pl.col('eaf').shift(1)))/max_budget).alias('aocc_contribution')
    )

def get_aocc(data : pl.DataFrame, 
                     ):
    # df = get_unaligned(data)
    #TODO: add column to each group with max budget

    eaf = _eaf_transform(data)
    aocc_contribs = eaf.group_by(*['data_id']).map_groups(_aocc)
    aoccs = aocc_contribs.group_by(*['data_id']).agg(pl.col('aocc_contribution').sum()).mean()

    return aoccs

def get_glicko2_ratings(data : pl.DataFrame, 
                        alg_vars : Iterable[str] = ['algorithm_name'],
                        fid_vars : Iterable[str] = ['function_name'],
                        perf_var : str = 'raw_y',
                        nrounds : int = 5,
                        ):
    from skelo import Glicko2Estimator
    alg_vars = ['algorithm_name', 'algorithm_info']
    fid_vars = ['function_name']
    perf_var = 'hv'
    players = data[alg_vars].unique()
    n_players = players.shape[0]
    fids = data[fid_vars].unique()
    aligned_comps = data.pivot(index=alg_vars, columns=fid_vars, values=perf_var, aggregate_function=pl.element())
    comp_arr = np.array(aligned_comps[aligned_comps.columns[len(alg_vars):]])

    nrounds = 5
    rng = np.random.default_rng()
    fids_shuffled = [i for i in range(len(fids))]
    p1_order = [i for i in range(n_players)]
    p2_order = [i for i in range(n_players)]
    records = []
    for round in range(nrounds):
        rng.shuffle(fids_shuffled)
        for fid in fids_shuffled:
            rng.shuffle(p1_order)
            for p1 in p1_order:
                rng.shuffle(p2_order)
                for p2 in p2_order:
                    if p1 == p2:
                        continue
                    s1 = rng.choice(comp_arr[p1][fid], 1)[0]
                    s2 = rng.choice(comp_arr[p2][fid], 1)[0]
                    if not np.isfinite(s1):
                        if not np.isfinite(s2):
                            won = 0.5
                        else:
                            won = 0.0
                    else:
                        if not np.isfinite(s2):
                            won = 1.0
                        elif s1 == s2:
                            won = 0.5
                        else:
                            won = float(s1 < s2)

                    records.append([round, p1, p2, won])
    dt_comp = pd.DataFrame.from_records(records, columns = ['round', 'p1', 'p2', 'outcome'])
    model = Glicko2Estimator(key1_field='p1', key2_field='p2', timestamp_field='round').fit(dt_comp, dt_comp['outcome'])
    ratings = np.array(model.rating_model.to_frame()[np.isnan(model.rating_model.to_frame()['valid_to'])]['rating'])
    rating_dt = pd.DataFrame([[rating[0] for rating in ratings],
              [rating[1] for rating in ratings],
              players[players.columns[0]],
              players[players.columns[1]]]).transpose()
    rating_dt.columns = ['Rating', 'Deviation', players.columns[0], players.columns[1]]
    return rating_dt

def aggegate_running_time(data : pl.DataFrame,
                          evaluation_variable : str = 'evaluations',
                          fval_variable : str = 'raw_y',
                          free_variables : Iterable[str] = ['algorithm_name'],
                          f_min : float = None,
                          f_max : float = None,
                          max_budget : int = None,
                 custom_op : Callable[[pl.Series], float] = None,
                 return_as_pandas : bool = True):
    """Internal function for getting fixed-budget information. 

    Args:
        data (DataSet): The data object to use for getting the performance. Note that the fval, evaluation and free variables as defined in 
        this object determine the axes of the final performance (most data will have 'raw_y', 'evaluations' and ['algId'] as defaults)
        custom_op (callable, optional): Any custom aggregation . Defaults to None.

    Returns:
        _type_: _description_
    """
    
    #Getting alligned data (to check if e.g. limits should be args for this function)
    if f_min is None:
        f_min = data[fval_variable].min()
    if f_max is None:
        f_max = data[fval_variable].max()
    f_values = get_sequence(f_min, f_max, 50, scale_log=True)
    group_variables = free_variables + [fval_variable]
    data_aligned = align_data(data, f_values, group_cols=['data_id'] + free_variables, x_col=fval_variable, y_col=evaluation_variable, maximization=True)
    if max_budget is None:
        max_budget = data[evaluation_variable].max()

    aggregations = [
        pl.mean(evaluation_variable).alias('mean'),
        pl.min(evaluation_variable).alias('min'),
        pl.max(evaluation_variable).alias('max'),
        pl.median(evaluation_variable).alias('median'),
        pl.std(evaluation_variable).alias('std'),
        pl.col(evaluation_variable).is_finite().mean().alias('success_ratio'),
        pl.col(evaluation_variable).is_finite().sum().alias('success_count'),
        (pl.col(evaluation_variable).replace(np.inf, max_budget).sum() / pl.col(evaluation_variable).is_finite().sum()).alias('ERT'),
        (pl.col(evaluation_variable).replace(np.inf, max_budget * 10).sum() / pl.col(evaluation_variable).count()).alias('PAR-10')
    ]

    if custom_op is not None:
        aggregations.append(pl.col(evaluation_variable).apply(lambda s: custom_op(s)).alias(custom_op.__name__))
    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    if return_as_pandas:
        return dt_plot.sort(fval_variable).to_pandas() 
    return dt_plot.sort(fval_variable)

#### Multi-objective

from moocore import hypervolume
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

    def __call__(self, group : pl.DataFrame, objective_columns : Iterable) -> pl.DataFrame:
        #clip is here to avoid negative values; note that this assumes minimization for all objectives
        obj_vals = np.clip(np.array(group[objective_columns]), None, self.reference_point)
        if len(objective_columns) == 2:
            hvs = self._incremental_hv(obj_vals)
        else:
            hvs = [hypervolume(obj_vals[:i], ref = self.reference_point) for i in range(1, len(group)+1)]
        group = group.with_columns(
            pl.Series(name="hv", values=hvs)
        )
        return group
    
    def _incremental_hv(self, points):
        sorted_array = [[-np.inf, self.reference_point[1]], [self.reference_point[0], -np.inf]]
        current_hypervolume = 0.0
        all_hv = []
        for point in np.array(points):
            dominated_idxs = [i for i,p in enumerate(sorted_array) if point[0] <= p[0] and point[1] <= p[1] and (point[0] < p[0] or point[1] < p[1])]
            point = tuple(point)
            if len(dominated_idxs) > 0:
                index = min(dominated_idxs)
                for _ in dominated_idxs:
                    dom_point = sorted_array[index]
                    left_neighbor = sorted_array[index - 1]
                    right_neighbor = sorted_array[index + 1]
                    current_hypervolume -= (left_neighbor[1] - dom_point[1]) * (right_neighbor[0] - dom_point[0])
                    sorted_array = [p for p in sorted_array if p is not dom_point]

            index = next((i for i, p in enumerate(sorted_array) if p[0] > point[0] or (p[0] == point[0] and p[1] < point[1])), len(sorted_array))
            sorted_array.insert(index, point)
            left_neighbor = sorted_array[index - 1]
            right_neighbor = sorted_array[index + 1]
            current_hypervolume += (left_neighbor[1] - point[1]) * (right_neighbor[0] - point[0])
            all_hv.append(current_hypervolume)
        return all_hv

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
    
class Final_NonDominated:
    def __call__(self, group : pl.DataFrame, objective_columns : Iterable):
        objectives = np.array(group[objective_columns])
        is_efficient = np.ones(objectives.shape[0], dtype = bool)
        for i, c in enumerate(objectives):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(objectives[is_efficient]<c, axis=1)  # Keep any later point with a lower cost
                is_efficient[i] = True  # And keep self
        group = group.with_columns(
            pl.Series(name="final_nondominated", values=is_efficient)
        )
        return group
    
def add_indicator(df : pl.DataFrame, indicator : object, objective_columns : Iterable):
    indicator_callable = partial(indicator, objective_columns = objective_columns)
    return df.group_by('data_id').map_groups(indicator_callable)
