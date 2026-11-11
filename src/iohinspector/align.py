"""Alignment of runs on a grid of evaluations or of function values."""

from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

from ._upgrade import removed_names, upgrade_hints

__all__ = ["align_data"]
__getattr__ = removed_names(__name__)


@upgrade_hints
def align_data(  # pylint: disable=too-many-arguments
    df: pl.DataFrame,
    x_values: Iterable[int | float],
    group_cols: Sequence[str] = ("run_id",),
    x_col: str = "evaluations",
    y_col: str = "raw_y",
    output: str = "long",
    maximization: bool = False,
) -> pl.DataFrame:
    """Align every run (a group of ``group_cols``) on the values ``x_values``.

    Fixed budget (``x_col`` is ``"evaluations"``): ``y_col`` becomes the
    best-so-far value of each run, and each run gets the value it had at each
    evaluation in ``x_values``. Fixed target (``x_col`` is a function value
    column): ``x_col`` becomes the best-so-far value, and each run gets the
    ``y_col`` (evaluations) at which it first reached each target. A value that
    does not exist yet (before the first evaluation, or a target never reached) is
    ``inf``. Works in one join for all runs.

    Args:
        df: at least ``x_col``, ``y_col`` and ``group_cols``, in evaluation order.
        x_values: the evaluations, or function values, to align on.
        group_cols: the columns that identify a run.
        x_col: the column to align on.
        y_col: the column to read at each aligned value.
        output: ``"long"``, or ``"wide"`` for one column per run.
        maximization: whether larger function values are better.

    Returns:
        pl.DataFrame: one row per run and value of ``x_values`` (long), or one row
            per value of ``x_values`` and one column per run (wide).
    """
    group_cols = list(dict.fromkeys(group_cols))
    fval = y_col if x_col == "evaluations" else x_col
    best = pl.col(fval).cum_max() if maximization else pl.col(fval).cum_min()
    df = df.with_columns(best.over(group_cols))
    grid = (
        df.select(group_cols)
        .unique()
        .join(
            pl.DataFrame({x_col: x_values}).cast({x_col: df.schema[x_col]}), how="cross"
        )
    )
    forward = x_col != "evaluations" and maximization
    if x_col == "evaluations":  # on a tie, the last row in data order
        df = df.sort(x_col, maintain_order=True)
    else:  # on a tie, the first evaluation that reached the target
        df = df.sort([x_col, y_col], descending=[False, not forward])
    result = (
        grid.sort(x_col)
        .join_asof(
            df,
            on=x_col,
            by=group_cols,
            strategy="forward" if forward else "backward",
            check_sortedness=False,
        )
        .with_columns(pl.col(y_col).fill_null(np.inf))
    )
    if output == "long":
        return result
    return result.pivot(index=x_col, on=group_cols, values=y_col)
