import numpy as np
import polars as pl
import pandas as pd
from typing import Iterable, Optional



def get_heatmap_single_run_data(
    data: pl.DataFrame,
    vars: Iterable[str],
    eval_var: str = "evaluations",
    var_mins: Iterable[float] = [-5],
    var_maxs: Iterable[float] = [5],
    return_as_pandas: bool = True,
) -> pl.DataFrame | pd.DataFrame:
    """Generate normalized heatmap data showing search space points evaluated in a single optimization run.

    Args:
        data (pl.DataFrame): The data object containing single-run optimization trajectory data.
        vars (Iterable[str]): Which columns correspond to the search space variable values.
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        var_mins (Iterable[float], optional): Minimum bounds for normalization of variables. Defaults to [-5].
        var_maxs (Iterable[float], optional): Maximum bounds for normalization of variables. Defaults to [5].
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pd.DataFrame or pl.DataFrame: A DataFrame with normalized variable values arranged for heatmap visualization.
    """
    assert data["data_id"].n_unique() == 1
    dt = data[vars].transpose().to_pandas()
    dt.columns = list(data[eval_var])
    var_mins_arr = np.array(var_mins)
    var_maxs_arr = np.array(var_maxs)
    dt = (dt.subtract(var_mins_arr, axis=0)).divide(var_maxs_arr - var_mins_arr, axis=0)
    if return_as_pandas:
        return dt
    return pl.from_pandas(dt)
