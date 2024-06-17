from typing import Iterable

import pandas as pd


def align_data(
    df: pd.DataFrame,
    evals: Iterable[int],
    group_cols: Iterable[str],
    x_col: str = "evaluations",
    y_col: str = "raw_y",
    output: str = "long",
) -> pd.DataFrame:
    """Align data based on function evaluation counts

    Args:
        df (pd.DataFrame): DataFrame containing at minimum x, y and group columns specified in further parameters
        evals (Iterable[int]): list containing the function evaluation values at which to align
        group_cols (Iterable[str]): columns to use for grouping
        x_col (str, optional): function evaluation column Defaults to 'evaluations'.
        y_col (str, optional): function value column. Defaults to 'raw_y'.
        output (str, optional): whether to return a long or wide dataframe as output. Defaults to 'long'.

    Returns:
        pd.DataFrame: Alligned DataFrame
    """
    evals_df = pd.DataFrame({x_col: evals})

    def merge_asof_group(group):
        merged = pd.merge_asof(
            evals_df, group, left_on=x_col, right_on=x_col, direction="backward"
        )
        for col in group_cols:
            merged[col] = group[col].iloc[0]
        return merged

    result_df = df.groupby(group_cols).apply(merge_asof_group).reset_index(drop=True)

    if output == "long":
        return result_df

    pivot_df = result_df.pivot(index=x_col, columns=group_cols, values=y_col)
    return pivot_df
