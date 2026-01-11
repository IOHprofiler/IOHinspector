import polars as pl
import pandas as pd
from typing import Iterable
from .utils import get_sequence
from ..align import align_data, turbo_align
from .utils import transform_fval




def get_data_ecdf(
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    free_vars: Iterable[str] = ["algorithm_name"],
    f_min: int = None,
    f_max: int = None,
    scale_f_log: bool = True,
    eval_values: Iterable[int] = None,
    eval_min: int = None,
    eval_max: int = None,
    scale_eval_log: bool = True,
    maximization: bool = False,
    turbo: bool = False,
    return_as_pandas: bool = True,
) -> pd.DataFrame | pl.DataFrame:
    """Generate empirical cumulative distribution function (ECDF) data based on EAF calculations.

    Args:
        data (pl.DataFrame): The DataFrame containing the full performance trajectory data.
        fval_var (str, optional): Which column contains the performance measure values. Defaults to "raw_y".
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        free_vars (Iterable[str], optional): Which columns to NOT aggregate over. Defaults to ["algorithm_name"].
        f_min (int, optional): Minimum value for function value scaling. If None, uses minimum from data. Defaults to None.
        f_max (int, optional): Maximum value for function value scaling. If None, uses maximum from data. Defaults to None.
        scale_f_log (bool, optional): Whether to use logarithmic scaling for function values. Defaults to True.
        eval_values (Iterable[int], optional): Specific evaluation values to use. If None, generated from eval_min/max. Defaults to None.
        eval_min (int, optional): Minimum evaluation value. If None, uses minimum from data. Defaults to None.
        eval_max (int, optional): Maximum evaluation value. If None, uses maximum from data. Defaults to None.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        maximization (bool, optional): Whether the performance measure is being maximized. Defaults to False.
        turbo (bool, optional): Whether to use turbo alignment for faster processing. Defaults to False.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pd.DataFrame or pl.DataFrame: A DataFrame containing the ECDF data with aligned evaluation points.
    """
    if eval_values is None:
        if eval_min is None:
            eval_min = data[eval_var].min()
        if eval_max is None:
            eval_max = data[eval_var].max()
        eval_values = get_sequence(
            eval_min, eval_max, 50, scale_log=scale_eval_log, cast_to_int=True
        )
    if turbo:
        data_aligned = turbo_align(
            data.cast({eval_var: pl.Int64}),
            eval_values,
            x_col=eval_var,
            y_col=fval_var,
            maximization=maximization,
        )
    else:
        data_aligned = align_data(
            data.cast({eval_var: pl.Int64}),
            eval_values,
            group_cols=["data_id"],
            x_col=eval_var,
            y_col=fval_var,
            maximization=maximization,
            silence_warning=True
        )
    dt_ecdf = (
        transform_fval(
            data_aligned,
            fval_var=fval_var,
            maximization=maximization,
            lb=f_min,
            ub=f_max,
            scale_log=scale_f_log,
        )
        .group_by([eval_var] + free_vars)
        .mean()
        .sort(eval_var)
    )

    if return_as_pandas:
        return dt_ecdf.to_pandas()
    return dt_ecdf