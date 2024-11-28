import polars as pl
import numpy as np

from typing import Iterable, Callable, Optional

from functools import partial
from .align import align_data

import pandas as pd


def get_sequence(
    min: float,
    max: float,
    len: float,
    scale_log: bool = False,
    cast_to_int: bool = False,
) -> np.ndarray:
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
        assert min > 0
        min = np.log10(min)
        max = np.log10(max)
        transform = lambda x: 10**x
    values = transform(
        np.arange(
            min,
            max + (max - min) / (2 * (len - 1)),
            (max - min) / (len - 1),
            dtype=float,
        )
    )
    if cast_to_int:
        return np.unique(np.array(values, dtype=int))
    return np.unique(values)


def _geometric_mean(series: pl.Series) -> float:
    """Helper function for polars: geometric mean"""
    return np.exp(np.log(series).mean())


def aggegate_convergence(
    data: pl.DataFrame,
    evaluation_variable: str = "evaluations",
    fval_variable: str = "raw_y",
    free_variables: Iterable[str] = ["algorithm_name"],
    x_min: int = None,
    x_max: int = None,
    custom_op: Callable[[pl.Series], float] = None,
    maximization: bool = False,
    return_as_pandas: bool = True,
):
    """Function to aggregate performance on a fixed-budget perspective

    Args:
        data (pl.DataFrame): The data object to use for getting the performance. Note that the fval, evaluation and free variables as defined in
        this object determine the axes of the final performance (most data will have 'raw_y', 'evaluations' and ['algId'] as defaults)
        evaluation_variable (str, optional): Column name for evaluation number. Defaults to "evaluations".
        fval_variable (str, optional): Column name for function value. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Column name for free variables (variables over which performance should not be aggregated). Defaults to ["algorithm_name"].
        x_min (int, optional): Minimum evaulation value to use. Defaults to None (minimum present in data).
        x_max (int, optional): Maximum evaulation value to use. Defaults to None (maximum present in data).
        custom_op (Callable[[pl.Series], float], optional): Custom aggregation method for performance values. Defaults to None.
        maximization (bool, optional): Whether performance metric is being maximized or not. Defaults to False.
        return_as_pandas (bool, optional): Whether the data should be returned as Pandas (True) or Polars (False) object. Defaults to True.

    Returns:
        DataFrame: Depending on 'return_as_pandas', a pandas or polars DataFrame with the aggregated performance values
    """

    # Getting alligned data (to check if e.g. limits should be args for this function)
    if x_min is None:
        x_min = data[evaluation_variable].min()
    if x_max is None:
        x_max = data[evaluation_variable].max()
    x_values = get_sequence(x_min, x_max, 50, scale_log=True, cast_to_int=True)
    group_variables = free_variables + [evaluation_variable]
    data_aligned = align_data(
        data.cast({evaluation_variable: pl.Int64}),
        x_values,
        group_cols=["data_id"] + free_variables,
        x_col=evaluation_variable,
        y_col=fval_variable,
        maximization=maximization,
    )

    aggregations = [
        pl.mean(fval_variable).alias("mean"),
        pl.min(fval_variable).alias("min"),
        pl.max(fval_variable).alias("max"),
        pl.median(fval_variable).alias("median"),
        pl.std(fval_variable).alias("std"),
        pl.col(fval_variable)
        .map_elements(lambda s: _geometric_mean(s), return_dtype=pl.Float64)
        .alias("geometric_mean"),
    ]

    if custom_op is not None:
        aggregations.append(
            pl.col(evaluation_variable)
            .map_elements(lambda s: custom_op(s), return_dtype=pl.Float64)
            .alias(custom_op.__name__)
        )
    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    if return_as_pandas:
        return dt_plot.sort(evaluation_variable).to_pandas()
    return dt_plot.sort(evaluation_variable)


def transform_fval(
    data: pl.DataFrame,
      lb:float=1e-8, 
      ub:float=1e8, 
      scale_log:bool=True, 
      maximization:bool=False, 
      fval_col:str="raw_y"
):
    """Helper function to transform function values (min-max normalization based on provided bounds and scaling)

    Args:
        data (pl.DataFrame): The data object to use for getting the performance.
        lb (float, optional): Lower bound for scaling of function values. Defaults to 1e-8.
        ub (float, optional): Upper bound for scaling of function values. Defaults to 1e8.
        scale_log (bool, optional): Whether function values should be log-scaled before scaling. Defaults to True.
        maximization (bool, optional): Whether function values is being maximized. Defaults to False.
        fval_col (str, optional): Which column in data to use. Defaults to "raw_y".

    Returns:
        _type_: a copy of the original data with a new column 'eaf' with the scaled function values (which is always to be maximized)
    """
    if scale_log:
        lb = np.log10(lb)
        ub = np.log10(ub)
        res = data.with_columns(
            ((pl.col(fval_col).log10() - lb) / (ub - lb)).clip(0, 1).alias("eaf")
        )
    else:
        res = data.with_columns(
            ((pl.col(fval_col) - lb) / (ub - lb)).clip(0, 1).alias("eaf")
        )
    if maximization:
        return res
    return res.with_columns((1 - pl.col("eaf")).alias("eaf"))


def _aocc(group: pl.DataFrame, max_budget: int, fval_col: str = "eaf"):
    group = group.cast({"evaluations": pl.Int64}).filter(
        pl.col("evaluations") <= max_budget
    )
    new_row = pl.DataFrame(
        {
            "evaluations": [0, max_budget],
            fval_col: [group[fval_col].min(), group[fval_col].max()],
        }
    )
    group = (
        pl.concat([group, new_row], how="diagonal")
        .sort("evaluations")
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
    )
    return group.with_columns(
        (
            (
                pl.col("evaluations").diff(n=1, null_behavior="ignore")
                * (pl.col(fval_col).shift(1))
            )
            / max_budget
        ).alias("aocc_contribution")
    )


def get_aocc(
    data: pl.DataFrame,
    max_budget: int,
    fval_col: str = "eaf",
    group_cols: Iterable[str] = ["function_name", "algorithm_name"],
):
    """Helper function for AOCC calculations

    Args:
        data (pl.DataFrame): The data object to use for getting the performance.
        max_budget (int): Maxium value of evaluations to use
        fval_col (str, optional): Which data column specifies the performance value. Defaults to "eaf".
        group_cols (Iterable[str], optional): Which columns to NOT aggregate over. Defaults to ["function_name", "algorithm_name"].

    Returns:
        pl.DataFrame: a polars dataframe with the area under the EAF (=area over convergence curve)
    """
    aocc_contribs = data.group_by(*["data_id"]).map_groups(
        partial(_aocc, max_budget=max_budget, fval_col=fval_col)
    )
    aoccs = aocc_contribs.group_by(["data_id"] + group_cols).agg(
        pl.col("aocc_contribution").sum()
    )
    aoccs.group_by(group_cols).agg(pl.col("aocc_contribution").mean().alias("AOCC"))
    return aoccs


def get_glicko2_ratings(
    data: pl.DataFrame,
    alg_vars: Iterable[str] = ["algorithm_name"],
    fid_vars: Iterable[str] = ["function_name"],
    perf_var: str = "raw_y",
    nrounds: int = 25,
):
    """Method to calculate Glicko2 ratings of a set of algorithm on a set of problems. 
    Calculated based on nrounds of competition, where in each round all algorithms face all others (pairwise) on every function.
    For each round, a sampled performance value is taken from the data and used to determine the winner. 

    Args:
        data (pl.DataFrame): The data object to use for getting the performance.
        alg_vars (Iterable[str], optional): Which variables specific the algortihms which will compete. Defaults to ["algorithm_name"].
        fid_vars (Iterable[str], optional): Which variables denote the problems on which will be competed. Defaults to ["function_name"].
        perf_var (str, optional): Which variable corresponds to the performance. Defaults to "raw_y".
        nrounds (int, optional): How many round should be played. Defaults to 25.

    Returns:
        pd.DataFrame: Pandas dataframe with rating, deviation and volatility for each 'alg_vars' combination
    """
    try:
        from skelo.model.glicko2 import Glicko2Estimator
    except:
        print("This functionality requires the 'skelo' package, which is not found. Please install it to use this function")
        return
    
    players = data[alg_vars].unique()
    n_players = players.shape[0]
    fids = data[fid_vars].unique()
    aligned_comps = data.pivot(
        index=alg_vars,
        columns=fid_vars,
        values=perf_var,
        aggregate_function=pl.element(),
    )
    comp_arr = np.array(aligned_comps[aligned_comps.columns[len(alg_vars) :]])

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
                            won = float(s1 < s2) #TODO: maximization argument!

                    records.append([round, p1, p2, won])
    dt_comp = pd.DataFrame.from_records(
        records, columns=["round", "p1", "p2", "outcome"]
    )
    model = Glicko2Estimator(
        key1_field="p1", key2_field="p2", timestamp_field="round"
    ).fit(dt_comp, dt_comp["outcome"])
    ratings = np.array(
        model.rating_model.to_frame()[
            np.isnan(model.rating_model.to_frame()["valid_to"])
        ]["rating"]
    )
    rating_dt = pd.DataFrame(
        [
            [rating[0] for rating in ratings],
            [rating[1] for rating in ratings],
            *players[players.columns],
        ]
    ).transpose()
    rating_dt.columns = ["Rating", "Deviation", *players.columns]
    return rating_dt


def aggegate_running_time(
    data: pl.DataFrame,
    evaluation_variable: str = "evaluations",
    fval_variable: str = "raw_y",
    free_variables: Iterable[str] = ["algorithm_name"],
    f_min: float = None,
    f_max: float = None,
    scale_flog: bool = True,
    max_budget: int = None,
    maximization: bool = False,
    custom_op: Callable[[pl.Series], float] = None,
    return_as_pandas: bool = True,
):
    """Function to aggregate performance on a fixed-target perspective

    Args:
        data (pl.DataFrame): The data object to use for getting the performance. Note that the fval, evaluation and free variables as defined in
        this object determine the axes of the final performance (most data will have 'raw_y', 'evaluations' and ['algId'] as defaults)
        evaluation_variable (str, optional): Column name for evaluation number. Defaults to "evaluations".
        fval_variable (str, optional): Column name for function value. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Column name for free variables (variables over which performance should not be aggregated). Defaults to ["algorithm_name"].
        f_min (int, optional): Minimum function value to use. Defaults to None (minimum present in data).
        f_max (int, optional): Maximum function value to use. Defaults to None (maximum present in data).
        scale_flog (bool): Whether or not function values should be scaled logarithmically for the x-axis. Defaults to True.
        max_budget: If present, what budget value should be the maximum considered. Defaults to None.
        custom_op (Callable[[pl.Series], float], optional): Custom aggregation method for performance values. Defaults to None.
        maximization (bool, optional): Whether performance metric is being maximized or not. Defaults to False.
        return_as_pandas (bool, optional): Whether the data should be returned as Pandas (True) or Polars (False) object. Defaults to True.

    Returns:
        DataFrame: Depending on 'return_as_pandas', a pandas or polars DataFrame with the aggregated performance values
    """

    # Getting alligned data (to check if e.g. limits should be args for this function)
    if f_min is None:
        f_min = data[fval_variable].min()
    if f_max is None:
        f_max = data[fval_variable].max()
    f_values = get_sequence(f_min, f_max, 50, scale_log=scale_flog)
    group_variables = free_variables + [fval_variable]
    data_aligned = align_data(
        data,
        f_values,
        group_cols=["data_id"] + free_variables,
        x_col=fval_variable,
        y_col=evaluation_variable,
        maximization=maximization,
    )
    if max_budget is None:
        max_budget = data[evaluation_variable].max()

    aggregations = [
        pl.col(evaluation_variable).replace(np.inf, max_budget).mean().alias("mean"),
        # pl.mean(evaluation_variable).alias("mean"),
        pl.col(evaluation_variable).replace(np.inf, max_budget).min().alias("min"),
        pl.col(evaluation_variable).replace(np.inf, max_budget).max().alias("max"),
        pl.col(evaluation_variable).replace(np.inf, max_budget).median().alias("median"),
        pl.col(evaluation_variable).replace(np.inf, max_budget).std().alias("std"),
        pl.col(evaluation_variable).is_finite().mean().alias("success_ratio"),
        pl.col(evaluation_variable).is_finite().sum().alias("success_count"),
        (
            pl.col(evaluation_variable).replace(np.inf, max_budget).sum()
            / pl.col(evaluation_variable).is_finite().sum()
        ).alias("ERT"),
        (
            pl.col(evaluation_variable).replace(np.inf, max_budget * 10).sum()
            / pl.col(evaluation_variable).count()
        ).alias("PAR-10"),
    ]

    if custom_op is not None:
        aggregations.append(
            pl.col(evaluation_variable)
            .apply(lambda s: custom_op(s))
            .alias(custom_op.__name__)
        )
    dt_plot = data_aligned.group_by(*group_variables).agg(aggregations)
    if return_as_pandas:
        return dt_plot.sort(fval_variable).to_pandas()
    return dt_plot.sort(fval_variable)

def add_normalized_objectives(data: pl.DataFrame, obj_cols: Iterable[str], max_vals: Optional[pl.DataFrame] = None):
    """Add new normalized columns to provided dataframe based on the provided objective columns

    Args:
        data (pl.DataFrame): The original dataframe
        obj_cols (Iterable[str]): The names of each objective column
        max_vals (Optional[pl.DataFrame]): If provided, these values will be used as the maxima instead of the values found in `data`

    Returns:
        _type_: The original `data` DataFrame with a new column 'objI' added for each objective, for I=1...len(obj_cols)
    """
    if type(max_vals) == pl.DataFrame:
        return data.with_columns([(data[colname]/max_vals[colname].max()).alias(f'obj{idx + 1}') for idx, colname in enumerate(obj_cols)])
    else:
        return data.with_columns([(data[colname]/data[colname].max()).alias(f'obj{idx + 1}') for idx, colname in enumerate(obj_cols)])

