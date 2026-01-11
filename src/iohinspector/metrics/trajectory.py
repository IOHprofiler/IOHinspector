import numpy as np
import polars as pl
import pandas as pd
from typing import Iterable
from iohinspector.align import align_data




def get_trajectory(data: pl.DataFrame, 
                   traj_length: int = None,
                   min_fevals: int = 1,
                   evaluation_variable: str = "evaluations",
                   fval_variable: str = "raw_y",
                   free_variables: Iterable[str] = ["algorithm_name"],
                   maximization: bool = False,
                   return_as_pandas: bool = True,
) -> pl.DataFrame | pd.DataFrame:
    """Generate aligned performance trajectories for algorithm comparison over fixed evaluation sequences.

    Args:
        data (pl.DataFrame): The data object containing algorithm performance trajectory data.
        traj_length (int, optional): Length of the trajectory to generate. If None, uses maximum evaluations from data. Defaults to None.
        min_fevals (int, optional): Starting evaluation number for the trajectory. Defaults to 1.
        evaluation_variable (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        fval_variable (str, optional): Which column contains the function values. Defaults to "raw_y".
        free_variables (Iterable[str], optional): Which columns to NOT aggregate over. Defaults to ["algorithm_name"].
        maximization (bool, optional): Whether the performance metric is being maximized. Defaults to False.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pl.DataFrame or pd.DataFrame: A DataFrame with aligned trajectory data where each row corresponds to a specific evaluation and performance value.
    """
    if traj_length is None:
        max_fevals = data[evaluation_variable].max()
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
        silence_warning=True
    )
    if return_as_pandas:
        data_aligned = data_aligned.to_pandas()
    return data_aligned