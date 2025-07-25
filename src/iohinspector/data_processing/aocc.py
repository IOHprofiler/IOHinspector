import polars as pl
from typing import Iterable, Callable
from functools import partial

def _aocc(group: pl.DataFrame, max_budget: int, fval_col: str = "eaf"):
    group = group.cast({"evaluations": pl.Int64}).filter(
        pl.col("evaluations") <= max_budget
    )
    new_row = pl.DataFrame(
        {
            "evaluations": [0, max_budget],
            fval_col: [group[fval_col].min(), group[fval_col].max()],
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
                * (pl.col(fval_col).shift(1))
            )
            / max_budget
        ).alias("aocc_contribution")
    ) 
    


def get_aocc(
    data: pl.DataFrame,
    max_budget: int,
    fval_col: str = "eaf",
    group_cols: Iterable[str] = ["function_name", "algorithm_name"],
):
    """Helper function for AOCC calculations

    Args:
        data (pl.DataFrame): The data object to use for getting the performance.
        max_budget (int): Maxium value of evaluations to use
        fval_col (str, optional): Which data column specifies the performance value. Defaults to "eaf".
        group_cols (Iterable[str], optional): Which columns to NOT aggregate over. Defaults to ["function_name", "algorithm_name"].

    Returns:
        pl.DataFrame: a polars dataframe with the area under the EAF (=area over convergence curve)
    """
    aocc_contribs = data.group_by(*["data_id"]).map_groups(
        partial(_aocc, max_budget=max_budget, fval_col=fval_col)
    )
    aoccs = aocc_contribs.group_by(["data_id"] + group_cols).agg(
        pl.col("aocc_contribution").sum()
    )
    return aoccs.group_by(group_cols).agg(
        pl.col("aocc_contribution").mean().alias("AOCC")
    )
