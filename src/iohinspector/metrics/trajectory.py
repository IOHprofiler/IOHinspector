import numpy as np
import polars as pl
from typing import Iterable
from iohinspector.align import align_data




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
    )
    return data_aligned