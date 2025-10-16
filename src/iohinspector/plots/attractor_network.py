import numpy as np
import pandas as pd
import polars as pl
from typing import Iterable, Tuple
import matplotlib
import matplotlib.pyplot as plt
from iohinspector.metrics import get_attractor_network


def plot_attractor_network(
    data,
    coord_vars: Iterable[str] = ["x1", "x2"],
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    maximization: bool = False,
    beta=40,
    epsilon=0.0001,
    file_name: str = None,
    ax: matplotlib.axes.Axes = None,
):
    """Plot an attractor network from the provided data

    Args:
        data (pl.DataFrame): The original dataframe, should contain the performance and position information
        coord_vars (Iterable[str], optional): Which columns correspond to position information. Defaults to ['x1', 'x2'].
        fval_var (str, optional): Which column corresponds to performance. Defaults to 'raw_y'.
        eval_var (str, optional): Which column corresponds to evaluations. Defaults to 'evaluations'.
        maximization (bool, optional): Whether fval_var is to be maximized. Defaults to False.
        beta (int, optional): Minimum stagnation lenght. Defaults to 40.
        epsilon (float, optional): Radius below which positions should be considered identical in the network. Defaults to 0.0001.

    Returns:
        pd.DataFrame, pd.DataFrame: two dataframes containing the nodes and edges of the network respectively.
    """
    try:
        import networkx as nx
    except:
        print("NetworkX is required to use this plot type")
        return
    from sklearn.manifold import MDS

    nodes, edges = get_attractor_network(
        data = data,
        coord_vars = coord_vars,
        fval_var = fval_var,
        eval_var= eval_var,
        maximization = maximization,
        beta = beta,
        epsilon = epsilon
    )
    network = nx.DiGraph()
    for idx, row in nodes.iterrows():
        network.add_node(
            idx,
            decision=np.array(row)[: len(coord_vars)],
            fitness=row["y"],
            hitcount=row["count"],
            evals=row["evals"] / row["count"],
        )

    for _, row in edges.iterrows():
        network.add_edge(
            row["start"],
            row["end"],
            weight=row["count"],
            evaldiff=row["stag_length_avg"],
        )
    network.remove_edges_from(nx.selfloop_edges(network))

    decision_matrix = [network.nodes[node]["decision"] for node in network.nodes()]
    mds = MDS(n_components=1, random_state=0)
    x_positions = mds.fit_transform(
        decision_matrix
    ).flatten()  # Flatten to get 1D array for x-axis
    y_positions = [network.nodes[node]["fitness"] for node in network.nodes()]
    pos = {
        node: (x, y) for node, x, y in zip(network.nodes(), x_positions, y_positions)
    }

    hitcounts = [network.nodes[node]["hitcount"] for node in network.nodes()]
    if len(hitcounts) > 1:
        min_hitcount = min(hitcounts)
        max_hitcount = max(hitcounts)
    # Node sizes and colors based on fitness values (as in your original code)
    if len(hitcounts) > 1 and np.std(hitcounts) > 0:
        node_sizes = [
            100
            + (
                400
                * (network.nodes[node]["hitcount"] - min_hitcount)
                / (max_hitcount - min_hitcount)
            )
            for node in network.nodes()
        ]
    else:
        node_sizes = [500] * len(hitcounts)
    fitness_values = y_positions  # Reuse y_positions as they represent 'fitness'
    norm = plt.Normalize(min(fitness_values), max(fitness_values))
    node_colors = plt.cm.viridis(norm(fitness_values))

    # Draw the graph
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(
        network,
        pos=pos,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors[:, :3],
        edge_color="gray",
        width=2,
        ax=ax,
    )

    # Add colorbar for fitness values
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array(fitness_values)
    ax.set_xlabel("MDS-reduced decision vector")
    ax.set_ylabel("fitness")
    if file_name:
        fig.tight_layout()
        fig.savefig(file_name)