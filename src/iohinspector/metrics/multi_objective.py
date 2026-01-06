
from typing import Iterable
import polars as pl
import pandas as pd
from iohinspector.indicators import final, add_indicator
from iohinspector.metrics import get_sequence


def get_pareto_front_2d(
    data: pl.DataFrame,
    obj1_var: str = "raw_y",
    obj2_var: str = "F2",
    return_as_pandas: bool = True,
) -> pl.DataFrame | pd.DataFrame:
    """Extract the Pareto front from a 2D multi-objective optimization dataset.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization data.
        obj1_var (str, optional): Which column contains the first objective values. Defaults to "raw_y".
        obj2_var (str, optional): Which column contains the second objective values. Defaults to "F2".
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pl.DataFrame or pd.DataFrame: A DataFrame containing only the non-dominated Pareto front points.
    """
    df = add_indicator(data, final.NonDominated(), [obj1_var, obj2_var])
    df = df.filter(pl.col("final_nondominated") == True)
    if return_as_pandas:
        return df.to_pandas()
    return df



def get_indicator_over_time_data(
    data: pl.DataFrame,
    indicator: object = None,
    obj_vars: Iterable[str] =  ["raw_y", "F2"],
    eval_min: int = 1,
    eval_max: int = 50_000,
    scale_eval_log: bool = True,
    eval_steps: int = 50,
    return_as_pandas: bool = True,
) -> pl.DataFrame | pd.DataFrame:
    """Calculate multi-objective indicator values over time for performance analysis.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization trajectory data.
        indicator (object, optional): The indicator object to calculate over time. Defaults to None.
        obj_vars (Iterable[str], optional): Which columns contain the objective values. Defaults to ["raw_y", "F2"].
        eval_min (int, optional): Minimum evaluation value to consider. Defaults to 1.
        eval_max (int, optional): Maximum evaluation value to consider. Defaults to 50_000.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        eval_steps (int, optional): Number of evaluation steps to generate. Defaults to 50.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pl.DataFrame or pd.DataFrame: A DataFrame with indicator values calculated over the specified evaluation timeline.
    """

    
    evals = get_sequence(
        eval_min, eval_max, eval_steps, cast_to_int=True, scale_log=scale_eval_log
    )
    df = add_indicator(
        data, indicator, obj_vars=obj_vars, evals=evals
    )

    if return_as_pandas:
        return df.to_pandas()
    return df