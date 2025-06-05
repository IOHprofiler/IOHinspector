from functools import partial
from warnings import warn
from typing import Iterable, Callable, Optional

import polars as pl
import numpy as np
import pandas as pd
from skelo.model.elo import EloEstimator

from .align import align_data




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
    lb: float = 1e-8,
    ub: float = 1e8,
    scale_log: bool = True,
    maximization: bool = False,
    fval_col: str = "raw_y",
):
    """Helper function to transform function values (min-max normalization based on provided bounds and scaling)

    Args:
        data (pl.DataFrame): The data object to use for getting the performance.
        lb (float, optional): Lower bound for scaling of function values. If None, it is the max value found in data. Defaults to 1e-8.
        ub (float, optional): Upper bound for scaling of function values. If None, it is the max value found in data. Defaults to 1e8.
        scale_log (bool, optional): Whether function values should be log-scaled before scaling. Defaults to True.
        maximization (bool, optional): Whether function values is being maximized. Defaults to False.
        fval_col (str, optional): Which column in data to use. Defaults to "raw_y".

    Returns:
        _type_: a copy of the original data with a new column 'eaf' with the scaled function values (which is always to be maximized)
    """
    if ub == None:
        ub = data[fval_col].max()
    if lb == None:
        lb = data[fval_col].min()
        if lb <= 0 and scale_log:
            lb = 1e-8
            warnings.warn(
                "If using logarithmic scaling, lb should be set to prevent errors in log-calculation. Lb is being overwritten to 1e-8 to avoid this."
            )
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
    return aoccs.group_by(group_cols).agg(
        pl.col("aocc_contribution").mean().alias("AOCC")
    )


def get_tournament_ratings(
    data: pl.DataFrame,
    alg_vars: Iterable[str] = ["algorithm_name"],
    fid_vars: Iterable[str] = ["function_name"],
    perf_var: str = "raw_y",
    nrounds: int = 25,
    maximization: bool = False,
):
    """Method to calculate ratings of a set of algorithm on a set of problems.
    Calculated based on nrounds of competition, where in each round all algorithms face all others (pairwise) on every function.
    For each round, a sampled performance value is taken from the data and used to determine the winner.
    This function uses the ELO rating scheme, as opposed to the Glicko2 scheme used in the IOHanalyzer. Deviations are estimated based on the last 5% of rounds.

    Args:
        data (pl.DataFrame): The data object to use for getting the performance.
        alg_vars (Iterable[str], optional): Which variables specific the algortihms which will compete. Defaults to ["algorithm_name"].
        fid_vars (Iterable[str], optional): Which variables denote the problems on which will be competed. Defaults to ["function_name"].
        perf_var (str, optional): Which variable corresponds to the performance. Defaults to "raw_y".
        nrounds (int, optional): How many round should be played. Defaults to 25.
        maximization (bool, optional): Whether the performance metric is being maximized. Defaults to False.

    Returns:
        pd.DataFrame: Pandas dataframe with rating, deviation and volatility for each 'alg_vars' combination
    """
    fids = data[fid_vars].unique()
    aligned_comps = data.pivot(
        index=alg_vars,
        columns=fid_vars,
        values=perf_var,
        aggregate_function=pl.element(),
    )
    players = aligned_comps[alg_vars]
    n_players = players.shape[0]
    comp_arr = np.array(aligned_comps[aligned_comps.columns[len(alg_vars) :]])

    rng = np.random.default_rng()
    fids = [i for i in range(len(fids))]
    lplayers = [i for i in range(n_players)]
    records = []
    for r in range(nrounds):
        for fid in fids:
            for p1 in lplayers:
                for p2 in lplayers:
                    if p1 == p2:
                        continue
                    s1 = rng.choice(comp_arr[p1][fid], 1)[0]
                    s2 = rng.choice(comp_arr[p2][fid], 1)[0]
                    if s1 == s2:
                        won = 0.5
                    else:
                        won = abs(float(maximization) - float(s1 < s2))

                    records.append([r, p1, p2, won])
                    
    dt_comp = pd.DataFrame.from_records(
        records, columns=["round", "p1", "p2", "outcome"]
    )
    dt_comp = dt_comp.sample(frac=1).sort_values("round")
    model = EloEstimator(key1_field="p1", key2_field="p2", timestamp_field="round").fit(
        dt_comp, dt_comp["outcome"]
    )
    model_dt = model.rating_model.to_frame()
    ratings = np.array(model_dt[np.isnan(model_dt["valid_to"])]["rating"])
    deviations = (
        model_dt.query(f"valid_from >= {nrounds * 0.95}").groupby("key")["rating"].std()
    )
    rating_dt_elo = pd.DataFrame(
        [
            ratings,
            deviations,
            *players[players.columns],
        ]
    ).transpose()
    rating_dt_elo.columns = ["Rating", "Deviation", *players.columns]
    return rating_dt_elo


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
        pl.col(evaluation_variable)
        .replace(np.inf, max_budget)
        .median()
        .alias("median"),
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


def add_normalized_objectives(
    data: pl.DataFrame, obj_cols: Iterable[str], max_vals: Optional[pl.DataFrame] = None, min_vals: Optional[pl.DataFrame] = None
):
    """Add new normalized columns to provided dataframe based on the provided objective columns

    Args:
        data (pl.DataFrame): The original dataframe
        obj_cols (Iterable[str]): The names of each objective column
        max_vals (Optional[pl.DataFrame]): If provided, these values will be used as the maxima instead of the values found in `data`
        min_vals (Optional[pl.DataFrame]): If provided, these values will be used as the minima instead of the values found in `data`

    Returns:
        _type_: The original `data` DataFrame with a new column 'objI' added for each objective, for I=1...len(obj_cols)
    """
    if type(max_vals) == pl.DataFrame:
        data_max = [max_vals[colname].max() for colname in obj_cols]
    else:
        data_max = [data[colname].max() for colname in obj_cols]
    if type(min_vals) == pl.DataFrame:
        data_min = [min_vals[colname].min() for colname in obj_cols]
    else:
        data_min = [data[colname].min() for colname in obj_cols]
    return data.with_columns(
        [
            ((data[colname] - data_min[idx]) / (data_max[idx] - data_min[idx])).alias(f"obj{idx + 1}")
            for idx, colname in enumerate(obj_cols)
        ]
    )


def _get_nodeidx(xloc, yval, nodes, epsilon):
    if len(nodes) == 0:
        return -1
    candidates = nodes[np.isclose(nodes["y"], yval, atol=epsilon)]
    if len(candidates) == 0:
        return -1
    idxs = np.all(
        np.isclose(np.array(candidates)[:, : len(xloc)], xloc, atol=epsilon), axis=1
    )
    if any(idxs):
        return candidates[idxs].index[0]
    return -1


def get_attractor_network(
    data,
    coord_vars=["x1", "x2"],
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    maximization: bool = False,
    beta=40,
    epsilon=0.0001,
    eval_max=None,
):
    """Create an attractor network from the provided data

    Args:
        data (pl.DataFrame): The original dataframe, should contain the performance and position information
        coord_vars (Iterable[str], optional): Which columns correspond to position information. Defaults to ['x1', 'x2'].
        fval_var (str, optional): Which column corresponds to performance. Defaults to 'raw_y'.
        eval_var (str, optional): Which column corresponds to evaluations. Defaults to 'evaluations'.
        maximization (bool, optional): Whether fval_var is to be maximized. Defaults to False.
        beta (int, optional): Minimum stagnation lenght. Defaults to 40.
        epsilon (float, optional): Radius below which positions should be considered identical in the network. Defaults to 0.0001.
        eval_max (int, optional): Maximum evaluation number. Defaults to the maximum of eval_var if None.
    Returns:
        pd.DataFrame, pd.DataFrame: two dataframes containing the nodes and edges of the network respectively.
    """

    running_idx = 0
    running_edgeidx = 0
    nodes = pd.DataFrame(columns=[*coord_vars, "y", "count", "evals"])
    edges = pd.DataFrame(columns=["start", "end", "count", "stag_length_avg"])
    if eval_max is None:
        eval_max = max(data[eval_var])

    for run_id in data["data_id"].unique():
        dt_group = data.filter(
            pl.col("data_id") == run_id, pl.col(eval_var) <= eval_max
        )
        if maximization:
            ys = np.maximum.accumulate(np.array(dt_group[fval_var]))
        else:
            ys = np.minimum.accumulate(np.array(dt_group[fval_var]))
        xs = np.array(dt_group[coord_vars])

        stopping_points = np.where(np.abs(np.diff(ys, prepend=np.inf)) > 0)[0]
        evals = np.array(dt_group[eval_var])

        stagnation_lengths = np.diff(evals[stopping_points], append=eval_max)
        edge_lengths = stagnation_lengths[stagnation_lengths > beta]
        real_idxs = [stopping_points[i] for i in np.where(stagnation_lengths > beta)[0]]

        xloc = xs[real_idxs[0]]
        yval = ys[real_idxs[0]]
        nodeidx = _get_nodeidx(xloc, yval, nodes, epsilon)
        if nodeidx == -1:
            nodes.loc[running_idx] = [*xloc, yval, 1, evals[real_idxs[0]]]
            node1 = running_idx
            running_idx += 1
        else:
            nodes.loc[nodeidx, "evals"] += evals[real_idxs[0]]
            nodes.loc[nodeidx, "count"] += 1
            node1 = nodeidx

        if len(real_idxs) == 1:
            continue

        for i in range(len(real_idxs) - 1):
            xloc = xs[real_idxs[i + 1]]
            yval = ys[real_idxs[i + 1]]
            nodeidx = _get_nodeidx(xloc, yval, nodes, epsilon)
            if nodeidx == -1:
                nodes.loc[running_idx] = [*xloc, yval, 1, evals[real_idxs[i + 1]]]
                node2 = running_idx
                running_idx += 1
            else:
                nodes.loc[nodeidx, "evals"] += evals[real_idxs[i + 1]]
                nodes.loc[nodeidx, "count"] += 1
                node2 = nodeidx

            edgelen = edge_lengths[i]
            edge_idxs = edges.query(f"start == {node1} & end == {node2}").index
            if len(edge_idxs) == 0:
                edges.loc[running_edgeidx] = [node1, node2, 1, edgelen]
                running_edgeidx += 1
            else:
                curr_count = edges.loc[edge_idxs[0]]["count"]
                curr_len = edges.loc[edge_idxs[0]]["stag_length_avg"]
                edges.loc[edge_idxs[0], "stag_length_avg"] = (
                    curr_len * curr_count + edgelen
                ) / (curr_count + 1)
                edges.loc[edge_idxs[0], "count"] += 1
            node1 = node2
    return nodes, edges


def get_data_ecdf(
    data,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    free_vars: Iterable[str] = ["algorithm_name"],
    maximization: bool = False,
    x_values: Iterable[int] = None,
    x_min: int = None,
    x_max: int = None,
    scale_xlog: bool = True,
    y_min: int = None,
    y_max: int = None,
    scale_ylog: bool = True,
):
    """Function to plot empirical cumulative distribution function (Based on EAF)

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        eval_var (str, optional): Column in 'data' which corresponds to the number of evaluations. Defaults to "evaluations".
        fval_var (str, optional): Column in 'data' which corresponds to the performance measure. Defaults to "raw_y".
        free_vars (Iterable[str], optional): Columns in 'data' which correspond to groups over which data should not be aggregated. Defaults to ["algorithm_name"].
        maximization (bool, optional): Boolean indicating whether the 'fval_var' is being maximized. Defaults to False.
        measures (Iterable[str], optional): List of measures which should be used in the plot. Valid options are 'geometric_mean', 'mean', 'median', 'min', 'max'. Defaults to ['geometric_mean'].
        x_values (Iterable[int], optional): List of x-values at which to get the ECDF data. If not provided, the x_min, x_max and scale_xlog arguments will be used to sample these points.
        scale_xlog (bool, optional): Should the x-samples be log-scaled. Defaults to True.
        x_min (float, optional): Minimum value to use for the 'eval_var', if not present the min of that column will be used. Defaults to None.
        x_max (float, optional): Maximum value to use for the 'eval_var', if not present the max of that column will be used. Defaults to None.
        scale_ylog (bool, optional): Should the y-values be log-scaled before normalization. Defaults to True.
        y_min (float, optional): Minimum value to use for the 'fval_var', if not present the min of that column will be used. Defaults to None.
        y_max (float, optional): Maximum value to use for the 'fval_var', if not present the max of that column will be used. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe of the ECDF data.
    """
    if x_values is None:
        if x_min is None:
            x_min = data[eval_var].min()
        if x_max is None:
            x_max = data[eval_var].max()
        x_values = get_sequence(
            x_min, x_max, 50, scale_log=scale_xlog, cast_to_int=True
        )
    data_aligned = align_data(
        data.cast({eval_var: pl.Int64}),
        x_values,
        group_cols=["data_id"],
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
    )
    dt_ecdf = (
        transform_fval(
            data_aligned,
            fval_col=fval_var,
            maximization=maximization,
            lb=y_min,
            ub=y_max,
            scale_log=scale_ylog,
        )
        .group_by([eval_var] + free_vars)
        .mean()
        .sort(eval_var)
    ).to_pandas()
    return dt_ecdf

def get_trajectory(data: pl.DataFrame, 
                   traj_length: int = None,
                   min_fevals: int = 1,
                   evaluation_variable: str = "evaluations",
                   fval_variable: str = "raw_y",
                   free_variables: Iterable[str] = ["algorithm_name"],
                    maximization: bool = False
) -> pl.DataFrame:
    """get the trajectory of the performance of the algorithms in the data
    This function aligns the data to a fixed number of evaluations and returns the performance trajectory.

    Args:
        data (pl.DataFrame): The DataFrame resulting from loading the data from a DataManager.
        traj_length (int, optional): Length of the trajecotry. Defaults to None.
        min_fevals (int, optional): Evaluation number from which to start the trajectory. Defaults to 1.
        evaluation_variable (str, optional): Variable corresponding to evaluation count in `data`. Defaults to "evaluations".
        fval_variable (str, optional): Variable corresponding to function value in `data`. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Free variables in `data`. Defaults to ["algorithm_name"].
        maximization (bool, optional): Whether the data is maximizing or not. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame: A polars DataFrame with the aligned data, where each row corresponds to a specific evaluation count and the performance value.
    """
    if traj_length is None:
        max_fevals = data[eval_var].max()
    else:
        max_fevals = traj_length + min_fevals
    x_values = np.arange(min_fevals, max_fevals + 1) 
    data_aligned = align_data(
        data.cast({evaluation_variable: pl.Int64}),
        x_values,
        group_cols=["data_id"] + free_variables,
        x_col=evaluation_variable,
        y_col=fval_variable,
        maximization=maximization,
    )
    return data_aligned