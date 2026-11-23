import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from iohinspector.metrics import get_attractor_network
from iohinspector.plots.utils import BasePlotArgs, _canvas, _create_plot_args


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
    file_name: str | None = None,
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
        tuple[matplotlib.axes.Axes, pl.DataFrame, pl.DataFrame]: The matplotlib axes
            object
            and two dataframes with the nodes and edges of the attractor network.
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required to use this plot type")
        return
    from sklearn.manifold import MDS

    nodes, edges = get_attractor_network(
        data=data,
        coord_vars=coord_vars,
        fval_var=fval_var,
        eval_var=eval_var,
        maximization=maximization,
        beta=beta,
        epsilon=epsilon,
    )

    plot_args = _create_plot_args(
        AttractorNetworkPlotArgs(
            title="Attractor Network",
            xlabel="MDS-reduced decision vector",
            ylabel="fitness",
        ),
        plot_args,
    )

    network = nx.DiGraph()
    network.add_nodes_from(range(nodes.height))
    network.add_edges_from(
        (start, end)
        for start, end in edges.select("start", "end").iter_rows()
        if start != end
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mds = MDS(n_components=1, random_state=0, n_init=4)
        x_positions = mds.fit_transform(list(nodes.select(coord_vars).to_numpy()))

    fitness_values = nodes["y"].to_list()
    pos = dict(enumerate(zip(x_positions.flatten(), fitness_values)))
    hitcounts = nodes["count"].to_numpy()
    if len(hitcounts) > 1 and np.std(hitcounts) > 0:
        low, high = hitcounts.min(), hitcounts.max()
        node_sizes = list(100 + 400 * (hitcounts - low) / (high - low))
    else:
        node_sizes = [500] * len(hitcounts)

    if plot_args.yscale == "log":
        norm = matplotlib.colors.LogNorm(min(fitness_values), max(fitness_values))
    else:
        norm = plt.Normalize(min(fitness_values), max(fitness_values))
    cmap = plt.get_cmap(plot_args.color_map)
    node_colors = cmap(norm(fitness_values))

    with _canvas(ax, plot_args, file_name) as ax:
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
    return ax, nodes, edges
