import numpy as np
import polars as pl
from typing import Sequence
from .._upgrade import upgrade_hints
from ..align import align_data


@upgrade_hints
def get_trajectory(
    data: pl.DataFrame,
    traj_length: int = None,
    min_fevals: int = 1,
    evaluation_variable: str = "evaluations",
    fval_variable: str = "raw_y",
    by: Sequence[str] = ("algorithm_name",),
    maximization: bool = False,
) -> pl.DataFrame:
    """Generate aligned performance trajectories for algorithm comparison over fixed evaluation sequences.

    Args:
        data (pl.DataFrame): The data object containing algorithm performance trajectory data.
        traj_length (int, optional): Length of the trajectory to generate. If None, uses maximum evaluations from data. Defaults to None.
        min_fevals (int, optional): Starting evaluation number for the trajectory. Defaults to 1.
        evaluation_variable (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        fval_variable (str, optional): Which column contains the function values. Defaults to "raw_y".
        by (Sequence[str], optional): The columns to group by: runs columns or
            descriptors. Defaults to ["algorithm_name"].
        maximization (bool, optional): Whether the performance metric is being maximized. Defaults to False.

    Returns:
        pl.DataFrame: A DataFrame with aligned trajectory data where each row
            corresponds to a specific evaluation and performance value.
    """
    if traj_length is None:
        max_fevals = data[evaluation_variable].max()
    else:
        max_fevals = traj_length + min_fevals
    x_values = np.arange(min_fevals, max_fevals + 1)

    return align_data(
        data.cast({evaluation_variable: pl.Int64}),
        x_values,
        group_cols=["run_id", *by],
        x_col=evaluation_variable,
        y_col=fval_variable,
        maximization=maximization,
        silence_warning=True,
    )
