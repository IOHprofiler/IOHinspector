from functools import partial
from warnings import warn
from typing import Iterable, Callable, Optional

import polars as pl
import numpy as np
import pandas as pd
from skelo.model.elo import EloEstimator

from .align import turbo_align, align_data






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