
from iohinspector.align import align_data, turbo_align
from iohinspector.metrics import transform_fval, get_sequence
import numpy as np
import pandas as pd
import polars as pl
from moocore import eaf, eafdiff


def get_discritized_eaf_single_objective(
    data: pl.DataFrame,
    fval_var: str = "raw_y",
    eval_var: str = "evaluations",
    eval_values = None,
    eval_min = None,
    eval_max = None,
    eval_targets = 10,
    scale_eval_log: bool = True,
    f_min = 1e-8,
    f_max = 1e2,
    scale_f_log: bool = True,  
    f_targets = 101,
    return_as_pandas: bool = True,
) -> pd.DataFrame | pl.DataFrame:
    """Generate discretized EAF data for single-objective optimization problems.

    Args:
        data (pl.DataFrame): The data object containing optimization trajectory data.
        fval_var (str, optional): Which column contains the function values. Defaults to "raw_y".
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        eval_values (array-like, optional): Specific evaluation values to use. If None, generated from eval_min/max.
        eval_min (int, optional): Minimum evaluation value. If None, uses minimum from data.
        eval_max (int, optional): Maximum evaluation value. If None, uses maximum from data.
        eval_targets (int, optional): Number of evaluation targets to generate. Defaults to 10.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        f_min (float, optional): Minimum function value for scaling. Defaults to 1e-8.
        f_max (float, optional): Maximum function value for scaling. Defaults to 1e2.
        scale_f_log (bool, optional): Whether to use logarithmic scaling for function values. Defaults to True.
        f_targets (int, optional): Number of function value targets to generate. Defaults to 101.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with discretized EAF data for single-objective problems.
    """

    if eval_values is None:
        if eval_min is None:
            eval_min = data[eval_var].min()
        if eval_max is None:
            eval_max = data[eval_var].max()
        eval_values = get_sequence(
            eval_min, eval_max, eval_targets, scale_log=scale_eval_log, cast_to_int=True
        )
    
    dt_aligned = turbo_align(
       data,
       eval_values,
       x_col=eval_var,
       y_col=fval_var,
       output="long"
    ) 
    dt_aligned = transform_fval(
        dt_aligned,
        lb=f_min,
        ub=f_max,
        scale_log=scale_f_log,
        fval_var=fval_var,
        )
    targets = np.linspace(0, 1, f_targets) 
    dt_targets = pd.DataFrame(targets, columns=["eaf_target"])

    dt_merged = dt_targets.merge(dt_aligned[[eval_var, 'eaf']].to_pandas(), how='cross')
    dt_merged['ps'] = dt_merged['eaf_target'] <= dt_merged['eaf']
    dt_discr = dt_merged.pivot_table(index='eaf_target', columns=eval_var, values='ps')
    if return_as_pandas:
        return dt_discr
    return pl.from_pandas(dt_discr)



def get_eaf_data( 
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    eval_min: int = None,
    eval_max: int = None,
    scale_eval_log: bool = True,
    return_as_pandas: bool = True,
    )-> pd.DataFrame | pl.DataFrame:
    """Generate aligned EAF data for visualization and analysis.

    Args:
        data (pl.DataFrame): The data object containing optimization trajectory data.
        eval_var (str, optional): Which column contains the evaluation numbers. Defaults to "evaluations".
        eval_min (int, optional): Minimum evaluation value. If None, uses minimum from data.
        eval_max (int, optional): Maximum evaluation value. If None, uses maximum from data.
        scale_eval_log (bool, optional): Whether to use logarithmic scaling for evaluations. Defaults to True.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pd.DataFrame or pl.DataFrame: A DataFrame with aligned EAF data.
    """

    if eval_min is None:
        eval_min = data[eval_var].min()
    if eval_max is None:
        eval_max = data[eval_var].max()

    evals = get_sequence(eval_min, eval_max, 50, scale_eval_log, True).astype("uint64")
    long = turbo_align(data, evals, output='long')

    if return_as_pandas:
        return long.to_pandas()
    return long


def get_eaf_pareto_data(
    data: pl.DataFrame,
    obj1_var: str,
    obj2_var: str,
    return_as_pandas: bool = True,
)-> pd.DataFrame | pl.DataFrame:
    """Generate EAF data for multi-objective optimization problems using Pareto fronts.

    Args:
        data (pl.DataFrame): The data object containing multi-objective optimization data.
        obj1_var (str): Name of the column containing first objective values.
        obj2_var (str): Name of the column containing second objective values.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pd.DataFrame or pl.DataFrame: A DataFrame with EAF data including objective values and EAF percentiles.
    """
    data_to_process = np.array(data[[obj1_var, obj2_var, "data_id"]])
    eaf_data = eaf(data_to_process[:,:-1], data_to_process[:,-1] )
    eaf_data_df = pd.DataFrame(eaf_data)
    eaf_data_df.columns = [obj1_var, obj2_var, "eaf"]
    # scale EAF values from percentages to proportions
    eaf_data_df["eaf"] = eaf_data_df["eaf"].astype(float) / 100.0
    if return_as_pandas:
        return eaf_data_df
    return pl.from_pandas(eaf_data_df)


def get_eaf_diff_data(
    data1: pl.DataFrame,
    data2: pl.DataFrame,
    obj1_var: str,
    obj2_var: str,
    return_as_pandas: bool = True,
)-> pd.DataFrame | pl.DataFrame:
    """Calculate EAF difference data between two multi-objective optimization datasets.

    Args:
        data1 (pl.DataFrame): First dataset containing multi-objective optimization data.
        data2 (pl.DataFrame): Second dataset containing multi-objective optimization data.
        obj1_var (str): Name of the column containing first objective values.
        obj2_var (str): Name of the column containing second objective values.
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pd.DataFrame or pl.DataFrame: A DataFrame with EAF difference rectangles and difference values.
    """
    x = np.array(data1[[obj1_var, obj2_var, "data_id"]])
    y = np.array(data2[[obj1_var, obj2_var, "data_id"]])
    if np.array_equal(np.sort(x.view(np.void), axis=0), np.sort(y.view(np.void), axis=0)):
        cols = ["x_min", "y_min", "x_max", "y_max", "eaf_diff"]
        empty_df = pl.DataFrame({c: [] for c in cols})
        if return_as_pandas:
            return empty_df.to_pandas()
        return empty_df
    eaf_diff_rect = eafdiff(x, y, rectangles=True)
    eaf_diff_df = pl.DataFrame(eaf_diff_rect, schema=["x_min", "y_min", "x_max", "y_max", "eaf_diff"])
   
    if return_as_pandas:
        return eaf_diff_df.to_pandas()
    return eaf_diff_df
