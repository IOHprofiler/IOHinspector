import warnings
from typing import Iterable

import polars as pl
import numpy as np


def align_data(
    df: pl.DataFrame,
    evals: Iterable[int | float],
    group_cols: Iterable[str] = ("data_id",),
    x_col: str = "evaluations",
    y_col: str = "raw_y",
    output: str = "long",
    maximization: bool = False,
) -> pl.DataFrame:
    """Align data based on function evaluation counts

    Args:
        df (pl.DataFrame): DataFrame containing at minimum x, y and group columns specified in further parameters
        evals (Iterable[int]): list containing the function evaluation values at which to align
        group_cols (Iterable[str]): columns to use for grouping
        x_col (str, optional): function evaluation column Defaults to 'evaluations'.
        y_col (str, optional): function value column. Defaults to 'raw_y'.
        output (str, optional): whether to return a long or wide dataframe as output. Defaults to 'long'.
        maximization (bool, optional): whether the data comes from maximization or minimization. Defaults to False (minimization).

    Returns:
        pl.DataFrame: Alligned DataFrame
    """

    warnings.warn( "Turbo align is favoured over this function", DeprecationWarning)
    evals_df = pl.DataFrame({x_col: evals})

    def merge_asof_group(group):
        fval_col = x_col
        if x_col == "evaluations":
            fval_col = y_col

        if maximization:
            group = group.with_columns(group[fval_col].cum_max().alias(fval_col))
        else:
            group = group.with_columns(group[fval_col].cum_min().alias(fval_col))

        if x_col != "evaluations" and maximization:
            merged = evals_df.join_asof(
                group.sort(x_col), on=x_col, strategy="forward"
            ).fill_null(np.inf)
        else:
            merged = evals_df.join_asof(
                group.sort(x_col), on=x_col, strategy="backward"
            ).fill_null(np.inf)

        for col in group_cols:
            merged = merged.with_columns(pl.lit(group[col][0]).alias(col))
        return merged

    result_df = df.group_by(*group_cols).map_groups(merge_asof_group)

    if output == "long":
        return result_df

    pivot_df = result_df.pivot(index=x_col, columns=group_cols, values=y_col)
    return pivot_df


def turbo_align(
    df: pl.DataFrame,
    x_values: Iterable[int | float],
    x_col: str = "evaluations",
    y_col: str = "raw_y",
    output: str = "long",
    maximization: bool = False,
):
    """Align data based on function evaluation counts (fast

    Note:
        Assumes the data is monotonic!
        Assumes data_id is present -> i.e. data comes from manager

    Args:
        df (pl.DataFrame): DataFrame containing at minimum x, y and group columns specified in further parameters
        evals (Iterable[int]): list containing the function evaluation values at which to align
        x_col (str, optional): function evaluation column Defaults to 'evaluations'.
        y_col (str, optional): function value column. Defaults to 'raw_y'.
        output (str, optional): whether to return a long or wide dataframe as output. Defaults to 'long'.
        maximization (bool, optional): whether the data comes from maximization or minimization. Defaults to False (minimization).

    Returns:
        pl.DataFrame: Alligned DataFrame
    """

    data_ids = df["data_id"].unique()

    x_vals = pl.DataFrame(
        { 
            x_col: np.repeat(x_values, len(data_ids)),
            "data_id": np.tile(data_ids, len(x_values)),
        },
        schema={x_col: df[x_col].dtype, "data_id": df['data_id'].dtype},
    )
    df = df.sort([x_col, 'data_id'])
    
    if x_col != "evaluations" and maximization:
        result_df = x_vals.join_asof(
            df, by="data_id", on=x_col, strategy="forward"
        )
    else:
        result_df = x_vals.join_asof(
            df, by="data_id", on=x_col, strategy="backward"
        )
        
              
    if output == "long":
        return result_df

    pivot_df = result_df.pivot(index=x_col, columns=("data_id",), values=y_col)
    return pivot_df
