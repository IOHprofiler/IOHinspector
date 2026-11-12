"""Fixed-budget summary and overview tables (``docs/definitions/fixed-budget.md``)."""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import polars as pl

from .._upgrade import upgrade_hints
from ..align import align_data
from ..targets import seq_rt
from .utils import custom, describe, name_run_sets, runs_of


@upgrade_hints
def aggregate_convergence(  # pylint: disable=R0913,R0914,R0917
    data: pl.DataFrame,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    by: Sequence[str] = ("algorithm_name",),
    eval_min: int | None = None,
    eval_max: int | None = None,
    custom_op: Callable[[pl.Series], float] | None = None,
    maximization: bool = False,
    *,
    budgets: Iterable[int] | None = None,
    stopped: str = "carry_forward",
    seed: int | None = None,
) -> pl.DataFrame:
    """Fixed-budget summary of each run set (a group of ``by``) at each budget.

    Follows ``docs/definitions/fixed-budget.md``. The statistics are over the runs
    that have a value at the budget: a run has none before its first logged
    evaluation (AL-5, FB-1). A run that stopped before the budget keeps its final
    value (``stopped="carry_forward"``), or is continued by simulated restarts
    (``"restart"``, FB-6), which need the runs' ``evals_used``
    (``DataManager.load(include_columns=["evals_used"])``).

    Args:
        data: the runs, with ``run_id``, ``eval_var``, ``fval_var`` and ``by``.
        eval_var: the column of evaluation numbers.
        fval_var: the column of function values.
        by: the columns that make a run set: runs columns or descriptors.
        eval_min: the smallest default budget; the data's when None.
        eval_max: the largest default budget; the data's when None.
        custom_op: a further statistic of each budget's values, as a column named
            after the function.
        maximization: whether larger function values are better.
        budgets: the budgets; 50 log-spaced from ``eval_min`` to ``eval_max`` when
            None.
        stopped: ``"carry_forward"`` or ``"restart"`` (FB-6).
        seed: the seed of the restarts.

    Returns:
        pl.DataFrame: per run set and budget (``eval_var``): ``mean``, ``median``,
            ``sd``, the type 7 quantiles ``q2`` ... ``q98`` (FB-2), ``best``,
            ``worst``, ``geometric_mean`` (null unless every value is > 0, FB-3),
            ``n_runs`` and ``n_defined``. With ``evals_used`` in ``data`` also
            ``n_active``, the runs that had not stopped (B_i >= b), and with
            restarts ``n_restarted``, the runs that had.
    """
    # #lizard forgives  (the public signature of 0.0.x plus the FB options)
    if data.is_empty():
        raise ValueError("Data is empty, cannot aggregate convergence.")
    if stopped not in ("carry_forward", "restart"):
        raise ValueError(f"stopped is 'carry_forward' or 'restart', not {stopped!r}")
    used = ["evals_used"] if "evals_used" in data.columns else []
    if stopped == "restart" and not used:
        raise ValueError(
            "stopped='restart' needs the runs' evals_used: "
            'DataManager.load(include_columns=["evals_used"])'
        )
    if budgets is None:
        budgets = seq_rt(data[eval_var], 50, lower=eval_min, upper=eval_max)
    data = data.cast({eval_var: pl.Int64})
    runs = runs_of(data, by, eval_var)
    aligned = align_data(
        data.select("run_id", eval_var, fval_var, *used),
        budgets,
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
        eval_var=eval_var,
    )
    if stopped == "restart":
        aligned = _restart(data, aligned, runs, eval_var, fval_var, maximization, seed)
    aligned = aligned.join(runs.select("run_id", "run_set"), on="run_id")
    y = pl.col(fval_var)
    best, worst = (pl.max, pl.min) if maximization else (pl.min, pl.max)
    aggregations = [
        *describe(y),
        best(fval_var).alias("best"),
        worst(fval_var).alias("worst"),
        pl.when((y > 0).all()).then(y.log().mean().exp()).alias("geometric_mean"),
    ]
    if "stopped" in aligned.columns:  # FB-6: runs with B_i >= b
        aggregations.append(pl.col("stopped").not_().sum().alias("n_active"))
        if stopped == "restart":
            aggregations.append(pl.col("stopped").sum().alias("n_restarted"))
    if custom_op is not None:
        aggregations.append(custom(custom_op, y))
    summary = aligned.group_by("run_set", eval_var).agg(aggregations)
    return name_run_sets(summary, runs, by, eval_var)


def _restart(  # pylint: disable=R0913,R0914,R0917
    data: pl.DataFrame,
    aligned: pl.DataFrame,
    runs: pl.DataFrame,
    eval_var: str,
    fval_var: str,
    maximization: bool,
    seed: int | None,
) -> pl.DataFrame:
    """``aligned`` with the value of every stopped run continued by simulated
    restarts (FB-6): after its B_i evaluations, run i is followed by runs drawn
    uniformly with replacement from its run set (one seeded chain per run), each
    for its own B_j evaluations, until the largest budget. The value at b is the
    best over the chain up to b, from ``align_data`` on the chain's joined trace.
    ``runs`` is ``runs_of(data, by)``."""
    # #lizard forgives  (one procedure, FB-6, kept in one place)
    runs = runs.sort("run_set", "run_id").with_columns(
        position=pl.int_range(pl.len()).over("run_set"),
        size=pl.len().over("run_set"),
    )
    ids = runs["run_id"].to_numpy()
    used = np.maximum(runs["evals_used"].to_numpy(), 1)
    size = runs["size"].to_numpy()
    first = np.arange(len(runs)) - runs["position"].to_numpy()
    rng = np.random.default_rng(seed)
    last = aligned[eval_var].max()
    owner = np.flatnonzero(used < last)  # the runs that stop before the last budget
    if not owner.size:
        return aligned
    chain = [(owner, owner, np.zeros_like(owner))]
    end = used[owner]
    while owner.size:  # one restart more for every chain that ends before it
        drawn = first[owner] + rng.integers(size[owner])
        chain.append((owner, drawn, end))
        end = end + used[drawn]
        owner, end = owner[end < last], end[end < last]
    owner, drawn, offset = (np.concatenate(c) for c in zip(*chain))
    segments = pl.DataFrame(
        {"owner": ids[owner], "run_id": ids[drawn], "offset": offset.astype(np.int64)}
    )
    trace = (
        segments.join(data.select("run_id", eval_var, fval_var), on="run_id")
        .select(
            run_id="owner",
            **{eval_var: pl.col(eval_var) + pl.col("offset"), fval_var: fval_var},
        )
        .sort("run_id", eval_var, maintain_order=True)
    )
    restarted = align_data(
        trace,
        aligned[eval_var].unique(),
        x_col=eval_var,
        y_col=fval_var,
        maximization=maximization,
        eval_var=eval_var,
    ).rename({fval_var: "restarted"})
    value = pl.when("stopped").then("restarted").otherwise(fval_var)
    return (
        aligned.join(restarted, on=["run_id", eval_var], how="left")
        .with_columns(value.alias(fval_var))
        .drop("restarted")
    )


def get_overview(
    data: pl.DataFrame,
    by: Sequence[str] = ("algorithm_name",),
    *,
    eval_var: str = "evaluations",
    fval_var: str = "raw_y",
    maximization: bool = False,
) -> pl.DataFrame:
    """Overview table of each run set (a group of ``by``), FB-4.

    A run's final value is its best value. B_i is ``evals_used``
    (``DataManager.load(include_columns=["evals_used"])``), or else the run's last
    logged evaluation.

    Returns:
        pl.DataFrame: per run set: ``n_runs``, ``budget`` (the largest B_i),
            ``best_reached``, ``worst_reached``, ``mean_reached`` and
            ``median_reached`` (of the final values), ``n_best`` (the runs whose
            final value is ``best_reached``), ``worst_recorded`` (the worst first
            logged value), ``min_runtime`` (the smallest first logged evaluation)
            and ``max_runtime`` (the last evaluation at which a run improved).
    """
    y, final = pl.col(fval_var), pl.col("final")
    best, worst = (pl.max, pl.min) if maximization else (pl.min, pl.max)
    runs = runs_of(data, by, eval_var)
    summary = (
        data.group_by("run_id")
        .agg(
            best(fval_var).alias("final"),
            y.first().alias("first"),
            pl.col(eval_var).min().alias("first_evaluation"),
            pl.col(eval_var)
            .filter(y == best(fval_var))
            .min()
            .alias("last_improvement"),
        )
        .join(runs, on="run_id")
        .group_by("run_set")
        .agg(
            pl.len().alias("n_runs"),
            pl.col("evals_used").max().alias("budget"),
            best("final").alias("best_reached"),
            worst("final").alias("worst_reached"),
            final.mean().alias("mean_reached"),
            final.median().alias("median_reached"),
            (final == best("final")).sum().alias("n_best"),
            worst("first").alias("worst_recorded"),
            pl.col("first_evaluation").min().alias("min_runtime"),
            pl.col("last_improvement").max().alias("max_runtime"),
        )
    )
    return name_run_sets(summary, runs, by)
