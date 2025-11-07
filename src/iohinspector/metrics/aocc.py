import polars as pl
import pandas as pd
from typing import Iterable, Callable
from functools import partial

def _aocc(
    group: pl.DataFrame, 
    eval_max: int, 
    fval_var: str = "eaf"
) -> pl.DataFrame:
    """Internal helper function to calculate AOCC contribution for a single data group.

    Args:
        group (pl.DataFrame): A single group DataFrame containing evaluation data for one run.
        eval_max (int): Maximum value of evaluations to consider for AOCC calculation.
        fval_var (str, optional): Which data column specifies the performance value. Defaults to "eaf".

    Returns:
        pl.DataFrame: DataFrame with added 'aocc_contribution' column containing normalized area contributions.
    """
    group = group.cast({"evaluations": pl.Int64}).filter(
        pl.col("evaluations") <= eval_max
    )
    new_row = pl.DataFrame(
        {
            "evaluations": [0, eval_max],
            fval_var: [group[fval_var].min(), group[fval_var].max()],
        }
    )
    group = (
        pl.concat([group, new_row], how="diagonal")
        .sort("evaluations")
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
    )
    
    return group.with_columns(
        (
            (
                pl.col("evaluations").diff(n=1, null_behavior="ignore")
                * (pl.col(fval_var).shift(1))
            )
            / eval_max
        ).alias("aocc_contribution")
    ) 
    


def get_aocc(
    data: pl.DataFrame,
    eval_max: int,
    fval_var: str = "eaf",
    free_vars: Iterable[str] = ["function_name", "algorithm_name"],
    return_as_pandas: bool = True,
) -> pl.DataFrame | pd.DataFrame:
    """Calculate Area Over Convergence Curve (AOCC) metric for algorithm performance evaluation.

    Args:
        data (pl.DataFrame): The data object containing performance evaluation data.
        eval_max (int): Maximum value of evaluations to use for AOCC calculation.
        fval_var (str, optional): Which data column specifies the performance value. Defaults to "eaf".
        free_vars (Iterable[str], optional): Which columns to NOT aggregate over. Defaults to ["function_name", "algorithm_name"].
        return_as_pandas (bool, optional): Whether to return results as pandas DataFrame. Defaults to True.

    Returns:
        pl.DataFrame or pd.DataFrame: A dataframe with the area under the EAF (=area over convergence curve).
    """
    aocc_contribs = data.group_by(*["data_id"]).map_groups(
        partial(_aocc, eval_max=eval_max, fval_var=fval_var)
    )
    aoccs = aocc_contribs.group_by(["data_id"] + free_vars).agg(
        pl.col("aocc_contribution").sum()
    )
    final_df = aoccs.group_by(free_vars).agg(
        pl.col("aocc_contribution").mean().alias("AOCC")
    )
    if return_as_pandas:
        return final_df.to_pandas()
    return final_df
