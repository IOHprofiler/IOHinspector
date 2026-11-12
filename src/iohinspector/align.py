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
    *,
    eval_var: str = "evaluations",
) -> pl.DataFrame:
    """Align every run (a group of ``group_cols``) on the values ``x_values``.

    Fixed budget (``x_col`` is ``eval_var``): ``y_col`` becomes the
    best-so-far value of each run, and each run gets the value it had at each
    evaluation in ``x_values``. Fixed target (``x_col`` is a function value
    column): ``x_col`` becomes the best-so-far value, and each run gets the
    ``y_col`` (evaluations) at which it first reached each target, ``inf`` for a
    target it never reached. Before a run's first logged evaluation its value is
    null (``docs/definitions/alignment.md``, AL-5): metrics leave it out of their
    statistics but count the run. Works in one join for all runs.

    Fixed budget with an ``evals_used`` column (the run's B_i from the run
    metadata, ``DataManager.load(include_columns=["evals_used"])``): each row
    also gets ``stopped``, whether the budget exceeds B_i (AL-9, FB-6); the value
    there is the run's final best value, carried forward. Without ``evals_used``
    there is no ``stopped`` column: the last logged evaluation is not B_i for
    sources that do not log the final evaluation.

    Args:
        df: at least ``x_col``, ``y_col`` and ``group_cols``, in evaluation order;
            optionally ``evals_used``.
        x_values: the evaluations, or function values, to align on; a value given
            twice counts once.
        group_cols: the columns that identify a run.
        x_col: the column to align on.
        y_col: the column to read at each aligned value.
        output: ``"long"``, or ``"wide"`` for one column per run.
        maximization: whether larger function values are better.
        eval_var: the evaluations column; aligning on it is the fixed-budget view.

    Returns:
        pl.DataFrame: one row per run and value of ``x_values`` (long), or one row
            per value of ``x_values`` and one column per run (wide).
    """
    group_cols = list(dict.fromkeys(group_cols))
    budget = x_col == eval_var
    fval = y_col if budget else x_col
    best = pl.col(fval).cum_max() if maximization else pl.col(fval).cum_min()
    df = df.with_columns(best.over(group_cols))
    stopped = budget and "evals_used" in df.columns
    if stopped:  # B_i per run, from the run metadata (AL-9)
        runs = df.group_by(group_cols).agg(pl.col("evals_used").max())
        df = df.drop("evals_used")
    else:
        runs = df.select(group_cols).unique()
    x = pl.DataFrame({x_col: x_values}).cast({x_col: df.schema[x_col]})
    grid = runs.join(x.unique(maintain_order=True), how="cross")
    if not budget:  # the first row with each best-so-far value reaches it (AL-3)
        df = df.filter(pl.col(fval).is_first_distinct().over(group_cols))
    result = grid.sort(x_col).join_asof(
        df.sort(x_col, maintain_order=True),  # budget: on a tie, the last row
        on=x_col,
        by=group_cols,
        strategy="forward" if maximization and not budget else "backward",
        check_sortedness=False,
    )
    if not budget:
        result = result.with_columns(pl.col(y_col).fill_null(np.inf))
    if stopped:
        result = result.with_columns(stopped=pl.col(x_col) > pl.col("evals_used"))
    if output == "long":
        return result
    return result.pivot(index=x_col, on=group_cols, values=y_col)
