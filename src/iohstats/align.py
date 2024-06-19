from typing import Iterable

import polars as pl


def align_data(
    df: pl.DataFrame,
    evals: Iterable[int],
    group_cols: Iterable[str],
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

    evals_df = pl.DataFrame({x_col: evals})

    def merge_asof_group(group):
        if maximization:
            group = group.with_columns(group[y_col].cum_max().alias(y_col))
        else:
            group = group.with_columns(group[y_col].cum_min().alias(y_col))

        if x_col != "evaluations" and not maximization:
            merged = evals_df.join_asof(group, on=x_col, strategy="forward")
        else:
            merged = evals_df.join_asof(group, on=x_col, strategy="backward")

        for col in group_cols:
            merged = merged.with_columns(pl.lit(group[col][0]).alias(col))
        return merged

    result_df = df.group_by(*group_cols).map_groups(merge_asof_group)

    if output == "long":
        return result_df

    pivot_df = result_df.pivot(index=x_col, columns=group_cols, values=y_col)
    return pivot_df
