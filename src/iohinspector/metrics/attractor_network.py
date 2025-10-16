import numpy as np
import pandas as pd
import polars as pl
from typing import Iterable, Tuple


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
        if not real_idxs:
            continue 

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