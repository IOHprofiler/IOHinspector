import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt

from iohinspector.metrics import get_attractor_network
from iohinspector.plots.utils import BasePlotArgs, _create_plot_args, _save_fig

@dataclass
class AttractorNetworkPlotArgs(BasePlotArgs):
    color_map: str = "viridis"
    
    def as_dict(self):
        """Convert the attractor network plot arguments to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all attractor network plot configuration parameters including color map.
        """
        results = super().as_dict()
        results["color_map"] = self.color_map
        return results

    def apply(self, ax):
        """Apply attractor network plot properties to a matplotlib Axes object.

        Args:
            ax: matplotlib Axes instance to apply the attractor network plot properties to.

        Returns:
            ax: The modified matplotlib Axes object with attractor network plot properties applied.
        """
        return super().apply(ax)

    def override(self, other):
        """Update attractor network plot arguments in place with values from another source.

        Args:
            other: Attractor network plot arguments to override current values with.
        """
        return super().override(other)


def plot_attractor_network(
    data: pl.DataFrame,
    coord_vars: Iterable[str] = ["x0", "x1"],
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    maximization: bool = False,
    beta: int = 40,
    epsilon: float = 0.0001,
    *,
    ax: matplotlib.axes.Axes = None,
    file_name: str = None,
    plot_args: dict | AttractorNetworkPlotArgs = None,
    
):
    """Plot an attractor network visualization from optimization algorithm data.

    Creates a network graph where nodes represent attractors (stable points) in the search space
    and edges represent transitions between them. Node sizes reflect visit frequency and colors
    represent fitness values.

    Args:
        data (pl.DataFrame): Input dataframe containing optimization algorithm trajectory data.
        coord_vars (Iterable[str], optional): Which columns contain the decision variable coordinates. 
            Defaults to ["x0", "x1"].
        fval_var (str, optional): Which column contains the fitness/objective values. Defaults to "raw_y".
        eval_var (str, optional): Which column contains the evaluation counts. Defaults to "evaluations".
        maximization (bool, optional): Whether the optimization problem is maximization. Defaults to False.
        beta (int, optional): Minimum stagnation length for attractor detection. Defaults to 40.
        epsilon (float, optional): Distance threshold below which positions are considered identical. 
            Defaults to 0.0001.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None, creates new figure. 
            Defaults to None.
        file_name (str, optional): Path to save the plot. If None, plot is not saved. Defaults to None.
        plot_args (dict | AttractorNetworkPlotArgs, optional): Plot styling arguments. Can include:
            - title (str): Plot title. Defaults to "Attractor Network".
            - xlabel (str): X-axis label. Defaults to "MDS-reduced decision vector".
            - ylabel (str): Y-axis label. Defaults to "fitness".
            - color_map (str): Colormap for node colors based on fitness. Defaults to "viridis".
            - figsize (Tuple[float, float]): Figure size. Defaults to (16, 9).
            - All other BasePlotArgs parameters (xlim, ylim, xscale, yscale, grid, legend, etc.).

    Returns:
        tuple[matplotlib.axes.Axes, pd.DataFrame, pd.DataFrame]: The matplotlib axes object 
            and two dataframes with the nodes and edges of the attractor network.
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

    plot_args = _create_plot_args(
        AttractorNetworkPlotArgs(
            title="Attractor Network",
            xlabel="MDS-reduced decision vector",
            ylabel="fitness",
            color_map="viridis"
        ),
        plot_args
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

    D = [network.nodes[node]["decision"] for node in network.nodes()]
    mds = MDS(n_components=1, random_state=0, n_init=4)
    if len(D[0]) == len(D):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            x_positions = mds.fit_transform(D)
    else:
        x_positions = mds.fit_transform(D)
    
    x_positions = x_positions.flatten()  # Flatten to get 1D array for x-axis
    y_positions = [network.nodes[node]["fitness"] for node in network.nodes()]
    pos = {
        node: (x, y) for node, x, y in zip(network.nodes(), x_positions, y_positions)
    }

    hitcounts = [network.nodes[node]["hitcount"] for node in network.nodes()]
    if len(hitcounts) > 1:
        min_hitcount = min(hitcounts)
        max_hitcount = max(hitcounts)
   
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
    
    if(plot_args.yscale == "log"):
        norm = matplotlib.colors.LogNorm(min(fitness_values), max(fitness_values))
    else:
        norm = plt.Normalize(min(fitness_values), max(fitness_values))

    # Safely get colormap name or default to 'viridis' if not present on plot_args
    cmap_name = getattr(plot_args, "color_map", "viridis")
    cmap = plt.get_cmap(cmap_name)
    node_colors = cmap(norm(fitness_values))

    if ax is None:
        fig, ax = plt.subplots(figsize=plot_args.figsize)
    else:
        fig = None
        
    nx.draw(
        network,
        pos=pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors[:, :3],
        edge_color="gray",
        width=2,
        ax=ax,
    )
    # ensure the axis frame, ticks and grid are visible
    ax.set_axis_on()
    ax.set_aspect("auto")

    # Add colorbar for fitness values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(fitness_values)
    plt.colorbar(sm, ax=ax)

    plot_args.apply(ax)

    _save_fig(fig, file_name, plot_args)
    return ax, nodes, edges