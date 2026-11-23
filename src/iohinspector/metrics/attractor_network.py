from collections.abc import Iterable
from itertools import pairwise

import numpy as np
import pandas as pd
import polars as pl

from .._upgrade import upgrade_hints


def _get_nodeidx(xloc: np.ndarray, yval: float, nodes: pd.DataFrame, epsilon: float):
    """Internal helper function to find existing node index based on position and function value.

    Args:
        xloc (array-like): Position coordinates to search for in the network.
        yval (float): Function value to match with existing nodes.
        nodes (pd.DataFrame): DataFrame containing existing network nodes.
        epsilon (float): Tolerance threshold for considering positions as identical.

    Returns:
        int: Index of matching node if found, -1 otherwise.
    """
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


@upgrade_hints
def get_attractor_network(
    data: pl.DataFrame,
    coord_vars: Iterable[str] = ["x1", "x2"],
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    maximization: bool = False,
    beta: int = 40,
    epsilon: float = 0.0001,
    eval_max=None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create an attractor network from optimization trajectory data.

    Args:
        data (pl.DataFrame): The original dataframe containing performance and position information.
        coord_vars (Iterable[str], optional): Which columns correspond to position information. Defaults to ["x1", "x2"].
        fval_var (str, optional): Which column corresponds to performance values. Defaults to "raw_y".
        eval_var (str, optional): Which column corresponds to evaluation numbers. Defaults to "evaluations".
        maximization (bool, optional): Whether fval_var is to be maximized. Defaults to False.
        beta (int, optional): Minimum stagnation length threshold. Defaults to 40.
        epsilon (float, optional): Radius below which positions should be considered identical in the network. Defaults to 0.0001.
        eval_max (int, optional): Maximum evaluation number to consider. Defaults to the maximum of eval_var if None.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: The nodes and the edges of the network. A
            node's
            number (``start`` and ``end`` of an edge) is its row number.
    """

    nodes = pd.DataFrame(columns=[*coord_vars, "y", "count", "evals"])
    edges = pd.DataFrame(columns=["start", "end", "count", "stag_length_avg"])
    edge_rows: dict[tuple, int] = {}  # (start, end) -> row of edges
    if eval_max is None:
        eval_max = max(data[eval_var])

    def node(xloc, yval, evals) -> int:
        """The row of the node at (xloc, yval), added or counted once more."""
        index = _get_nodeidx(xloc, yval, nodes, epsilon)
        if index == -1:
            index = len(nodes)
            nodes.loc[index] = [*xloc, yval, 1, evals]
        else:
            nodes.loc[index, "evals"] += evals
            nodes.loc[index, "count"] += 1
        return index

    runs = data.filter(pl.col(eval_var) <= eval_max).partition_by(
        "run_id", as_dict=True
    )
    for run_id in data["run_id"].unique(maintain_order=True):
        if (run_id,) not in runs:
            continue
        dt_group = runs[run_id,]
        best = np.maximum if maximization else np.minimum
        ys = best.accumulate(np.array(dt_group[fval_var]))
        xs = np.array(dt_group[coord_vars])
        evals = np.array(dt_group[eval_var])

        stopping_points = np.where(np.abs(np.diff(ys, prepend=np.inf)) > 0)[0]
        stagnation_lengths = np.diff(evals[stopping_points], append=eval_max)
        long = stagnation_lengths > beta
        ids = [node(xs[i], ys[i], evals[i]) for i in stopping_points[long]]
        for (node1, node2), edgelen in zip(pairwise(ids), stagnation_lengths[long]):
            row = edge_rows.get((node1, node2))
            if row is None:
                row = edge_rows[node1, node2] = len(edges)
                edges.loc[row] = [node1, node2, 1, edgelen]
            else:
                count, length = (
                    edges.loc[row, "count"],
                    edges.loc[row, "stag_length_avg"],
                )
                edges.loc[row, "stag_length_avg"] = (length * count + edgelen) / (
                    count + 1
                )
                edges.loc[row, "count"] += 1
    return pl.from_pandas(nodes.infer_objects()), pl.from_pandas(edges.infer_objects())
