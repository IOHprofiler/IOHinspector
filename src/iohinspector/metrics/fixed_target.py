"""Runtime summary at fixed targets (``docs/definitions/runtime.md``, RT-1..RT-6)."""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import polars as pl

from .._upgrade import upgrade_hints
from ..align import align_data
from ..targets import seq_fv
from .utils import QUANTILES, name_run_sets, runs_of


def _index(t: pl.Expr, i: pl.Expr) -> pl.Expr:
    """``t[i]``, or ``t[0]`` for an index below 0."""
    return t.get(pl.max_horizontal(i, 0))


def _type3(t: pl.Expr, n: pl.Expr, pct: int) -> pl.Expr:
    """The Hyndman-Fan type 3 ``pct`` % quantile of the first ``n`` values of ``t``,
    which are sorted: order statistic round-half-even(n p), at least the first."""
    q, r = n * pct // 100, n * pct % 100
    rank = q + (r > 50).cast(pl.Int64) + ((r == 50) & (q % 2 == 1)).cast(pl.Int64)
    return _index(t, pl.min_horizontal(rank, n) - 1)


@upgrade_hints
def aggregate_running_time(  # pylint: disable=too-many-arguments,too-many-locals
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    by: Sequence[str] = ("algorithm_name",),
    f_min: float | None = None,
    f_max: float | None = None,
    scale_f_log: bool = True,
    eval_max: int | None = None,
    maximization: bool = False,
    custom_op: Callable[[pl.Series], float] | None = None,
    *,
    targets: Iterable[float] | None = None,
    k: float = 10,
    include_unfinished: bool = True,
) -> pl.DataFrame:
    """Runtime summary of each run set (a group of ``by``) at each target.

    Follows ``docs/definitions/runtime.md``. Each run's budget B_i is its
    ``evals_used`` (``DataManager.load(include_columns=["evals_used"])``), or else
    its last logged evaluation. A run succeeds at a target when it reaches it
    within the cap ``eval_max`` (C); a failed run is charged min(B_i, C) (RT-1).

    Args:
        data: the runs, with ``run_id``, ``eval_var``, ``fval_var`` and ``by``.
        eval_var: the column of evaluation numbers.
        fval_var: the column of function values.
        by: the columns that make a run set: runs columns or descriptors.
        f_min: the smallest default target; the data's under log scale its
            smallest positive value (TG-3).
        f_max: the largest default target; the data's maximum when None.
        scale_f_log: whether the default targets are log-spaced.
        eval_max: the budget cap C; None for no cap.
        maximization: whether larger function values are better.
        custom_op: a further statistic of each target's runtimes (``inf`` for a
            failure), as a column named after the function.
        targets: the targets; 50 between ``f_min`` and ``f_max`` when None.
        k: the PAR-k penalty factor (RT-3).
        include_unfinished: quantiles over all runs, a failure counting as ``inf``
            (RT-5); False for the successful runs only (IOHanalyzer).

    Returns:
        pl.DataFrame: per run set and target (``fval_var``): ``n_runs``,
            ``n_success``, ``success_ratio``, ``ERT`` (``inf`` without a success,
            RT-2), ``par_{k}`` (cutoff C, or else the largest B_i), ``mean``,
            ``median`` and ``sd`` of the successful runtimes (RT-4), and the type 3
            quantiles ``q2`` ... ``q98`` (RT-5).
    """
    if targets is None:
        targets = seq_fv(data[fval_var], 50, scale_f_log, f_min, f_max)
    hits = align_data(
        data.select("run_id", eval_var, fval_var),
        targets,
        x_col=fval_var,
        y_col=eval_var,
        maximization=maximization,
    )
    runs = runs_of(data, by, eval_var)
    t, budget = pl.col(eval_var), pl.col("evals_used").cast(pl.Float64)
    cap = np.inf if eval_max is None else eval_max
    # a failure is inf, so sorting puts the successes first in every group
    hits = (
        hits.lazy()
        .with_columns(pl.when(t <= cap).then(t).otherwise(np.inf).alias(eval_var))
        .join(runs.lazy().select("run_id", "run_set", "evals_used"), on="run_id")
        .with_columns(
            charge=pl.min_horizontal(budget, cap),
            cutoff=budget if eval_max is None else pl.lit(float(cap)),
        )
        .sort(eval_var)
    )
    ok = t.is_finite()
    n_runs, n_success = pl.len().cast(pl.Int64), ok.sum().cast(pl.Int64)
    n_quantile = n_runs if include_unfinished else n_success
    middle = [_index(t, (n_success - 1) // 2), _index(t, n_success // 2)]
    aggregations = [
        n_runs.alias("n_runs"),
        n_success.alias("n_success"),
        t.filter(ok).sum().alias("solved"),  # by the sum of hitting times and
        pl.col("charge").filter(~ok).sum().alias("charged"),  # charged budgets
        pl.col("cutoff").max(),
        ((middle[0] + middle[1]) / 2).alias("median"),
        t.filter(ok).std().alias("sd"),
        *(_type3(t, n_quantile, pct).alias(f"q{pct}") for pct in QUANTILES),
    ]
    if custom_op is not None:
        aggregations.append(
            t.map_batches(
                custom_op, return_dtype=pl.Float64, returns_scalar=True
            ).alias(custom_op.__name__)
        )
    n, n_s, solved = pl.col("n_runs"), pl.col("n_success"), pl.col("solved")
    n_q = n if include_unfinished else n_s
    summary = hits.group_by("run_set", fval_var).agg(aggregations).collect()
    summary = (
        summary.with_columns(
            success_ratio=n_s / n,
            ERT=(solved + pl.col("charged")) / n_s,
            **{f"par_{k:g}": (solved + (n - n_s) * k * pl.col("cutoff")) / n},
            mean=pl.when(n_s > 0).then(solved / n_s),
            median=pl.when(n_s > 0).then("median"),
            **{f"q{pct}": pl.when(n_q > 0).then(f"q{pct}") for pct in QUANTILES},
        )
    ).drop("solved", "charged", "cutoff")
    return name_run_sets(summary, runs, by, fval_var)
