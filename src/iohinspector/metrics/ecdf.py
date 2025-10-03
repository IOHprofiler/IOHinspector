import polars as pl
from typing import Iterable
from .utils import get_sequence
from ..align import align_data
from .normalise_objectives import transform_fval




def get_data_ecdf(
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    free_vars: Iterable[str] = ["algorithm_name"],
    maximization: bool = False,
    x_values: Iterable[int] = None,
    x_min: int = None,
    x_max: int = None,
    scale_xlog: bool = True,
    y_min: int = None,
    y_max: int = None,
    scale_ylog: bool = True,
):
    """Function to plot empirical cumulative distribution function (Based on EAF)

    Args:
        data (pl.DataFrame): The DataFrame which contains the full performance trajectory. Should be generated from a DataManager.
        eval_var (str, optional): Column in 'data' which corresponds to the number of evaluations. Defaults to "evaluations".
        fval_var (str, optional): Column in 'data' which corresponds to the performance measure. Defaults to "raw_y".
        free_vars (Iterable[str], optional): Columns in 'data' which correspond to groups over which data should not be aggregated. Defaults to ["algorithm_name"].
        maximization (bool, optional): Boolean indicating whether the 'fval_var' is being maximized. Defaults to False.
        measures (Iterable[str], optional): List of measures which should be used in the plot. Valid options are 'geometric_mean', 'mean', 'median', 'min', 'max'. Defaults to ['geometric_mean'].
        x_values (Iterable[int], optional): List of x-values at which to get the ECDF data. If not provided, the x_min, x_max and scale_xlog arguments will be used to sample these points.
        scale_xlog (bool, optional): Should the x-samples be log-scaled. Defaults to True.
        x_min (float, optional): Minimum value to use for the 'eval_var', if not present the min of that column will be used. Defaults to None.
        x_max (float, optional): Maximum value to use for the 'eval_var', if not present the max of that column will be used. Defaults to None.
        scale_ylog (bool, optional): Should the y-values be log-scaled before normalization. Defaults to True.
        y_min (float, optional): Minimum value to use for the 'fval_var', if not present the min of that column will be used. Defaults to None.
        y_max (float, optional): Maximum value to use for the 'fval_var', if not present the max of that column will be used. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe of the ECDF data.
    """
    if x_values is None:
        if x_min is None:
            x_min = data[eval_var].min()
        if x_max is None:
            x_max = data[eval_var].max()
        x_values = get_sequence(
            x_min, x_max, 50, scale_log=scale_xlog, cast_to_int=True
        )
    data_aligned = align_data(
        data.cast({eval_var: pl.Int64}),
        x_values,
        group_cols=["data_id"],
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
    )
    dt_ecdf = (
        transform_fval(
            data_aligned,
            fval_col=fval_var,
            maximization=maximization,
            lb=y_min,
            ub=y_max,
            scale_log=scale_ylog,
        )
        .group_by([eval_var] + free_vars)
        .mean()
        .sort(eval_var)
    ).to_pandas()
    return dt_ecdf