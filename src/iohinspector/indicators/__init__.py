from typing import Iterable, Callable
from functools import partial

import polars as pl
import numpy as np

from .anytime import *
from .final import *

def add_indicator(
    df: pl.DataFrame, indicator: Callable, objective_columns: Iterable, **kwargs
) -> pl.DataFrame:
    """Adds an indicator to a Polars DataFrame.

    This function applies a specified indicator function to groups of data
    within a Polars DataFrame, grouped by the "data_id" column. The indicator
    is applied to the specified objective columns, along with any additional
    keyword arguments.

    Args:
        df (pl.DataFrame):
            The Polars DataFrame containing the data to which the indicator
            will be added. It must contain a column named "data_id" for
            grouping purposes.
        indicator (object):
            A callable object (e.g., a function) that computes the desired
            indicator. It must accept `objective_columns` and any additional
            arguments passed via `kwargs`.
        objective_columns (Iterable):
            An iterable of column names in the DataFrame that the indicator
            function will process.
        **kwargs:
            Additional keyword arguments to be passed to the `indicator`
            callable.

    Returns:
        pl.DataFrame:
            A new Polars DataFrame with the computed indicator applied to each
            group of data.
    """
    indicator_callable = partial(
        indicator, objective_columns=objective_columns, **kwargs
    )
    return df.group_by("data_id").map_groups(indicator_callable)
