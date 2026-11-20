"""Line figures of a performance summary: convergence, fixed budget, ERT over the
dimension and the fixed-budget ECDF. Each frame is the metric's own result, drawn
as it is; ``Data.at`` is the target or budget they are at (``spec.Data``)."""

import polars as pl

from ..metrics import (
    aggregate_convergence,
    aggregate_running_time,
    get_fixed_budget_ecdf,
)
from .base import (
    Plot,
    categories,
    curves,
    default_budget,
    default_target,
    finite,
    library_kind,
    log_if_positive,
    one_pair,
    pairs,
)
from .spec import Annotation


def _with_pair(frame: pl.DataFrame, runs: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        function_id=pl.lit(runs["function_id"][0]),
        dimension=pl.lit(runs["dimension"][0]),
    )


def convergence_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``aggregate_convergence`` of one (function, dimension) pair: the quantiles
    of the best-so-far value at each budget, per group."""
    d = panel.data
    return aggregate_convergence(
        one_pair(runs, "convergence"), by=d.by, maximization=d.maximization, **d.options
    )


def convergence_plot(panel, frame: pl.DataFrame) -> Plot:
    """The median of each group with its quartile band over the evaluations, and
    the target ``Data.at`` as a dotted line."""
    d = panel.data
    frame = frame.drop_nulls(["median", "q25", "q75"])
    lines = curves(panel, frame, "evaluations", "median", "q25", "q75")
    scale = log_if_positive(frame["q25"], frame["q75"], maximization=d.maximization)
    target = [Annotation("line", "target", y=d.at)] if d.at is not None else []
    return Plot(
        lines,
        x={"label": "# f-evals", "scale": "log", "ticks": "exponent"},
        y={"label": "best-so-far value", "scale": scale},
        annotations=target,
    )


def fixed_budget_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``aggregate_convergence`` at one budget (``Data.at``, else each pair's
    largest) for every (function, dimension) pair, with their columns."""
    d = panel.data
    return pl.concat(
        _with_pair(
            aggregate_convergence(
                run,
                by=d.by,
                maximization=d.maximization,
                budgets=[d.at or default_budget(run)],
                **d.options,
            ),
            run,
        )
        for run in pairs(runs)
    )


def fixed_budget_plot(panel, frame: pl.DataFrame) -> Plot:
    """The median value of each group, one point per pair; a pair a group has no
    value for is a missing point."""
    frame = frame.drop_nulls("median")
    keys = (
        frame.select("function_id", "dimension")
        .unique()
        .sort("function_id", "dimension")
    )
    position = {k: i for i, k in enumerate(keys.iter_rows())}
    frame = frame.with_columns(
        pair=pl.Series(
            [position[k] for k in frame.select("function_id", "dimension").iter_rows()]
        )
    )
    ticks = [f"f{f} d{d}" for f, d in position]
    return Plot(
        curves(panel, frame, "pair", "median"),
        x=categories(ticks),
        y={
            "label": "median value",
            "scale": log_if_positive(
                frame["median"], maximization=panel.data.maximization
            ),
        },
        mark="points",
    )


def ert_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``aggregate_running_time`` of each dimension of one function at the target
    ``Data.at``, by default the loosest all runs reach."""
    d = panel.data
    if runs["function_id"].n_unique() > 1:
        raise ValueError("ert: one function; select it in data.where")
    target = d.at if d.at is not None else default_target(runs, d.maximization)
    return pl.concat(
        _with_pair(
            aggregate_running_time(
                run, by=d.by, targets=[target], maximization=d.maximization, **d.options
            ),
            run,
        )
        for run in pairs(runs)
    )


def ert_plot(panel, frame: pl.DataFrame) -> Plot:
    """The ERT of each group over the dimension; a dimension without a success
    has no ERT and no point."""
    frame = finite(frame, "ERT")
    return Plot(
        curves(panel, frame, "dimension", "ERT"),
        x={"label": "dimension", "scale": "log", "ticks": "plain"},
        y={"label": "ERT (# f-evals)", "scale": "log"},
    )


def fixed_budget_ecdf_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_fixed_budget_ecdf``: the share of (run, budget) pairs at each value."""
    d = panel.data
    return get_fixed_budget_ecdf(
        runs, by=d.by, maximization=d.maximization, **d.options
    )


def fixed_budget_ecdf_plot(panel, frame: pl.DataFrame) -> Plot:
    """One step line per group over the value reached (its first row, at -inf,
    left out)."""
    frame = finite(frame, "raw_y")
    return Plot(
        curves(panel, frame, "raw_y", "ecdf"),
        x={
            "label": "value",
            "scale": log_if_positive(
                frame["raw_y"], maximization=panel.data.maximization
            ),
        },
        y={"label": "Fraction of (run, budget) pairs", "lim": [0, 1]},
        step=True,
    )


KINDS = {
    "convergence": library_kind(
        convergence_frame, convergence_plot, ("evaluations", "median", "q25", "q75")
    ),
    "fixed_budget": library_kind(
        fixed_budget_frame, fixed_budget_plot, ("function_id", "dimension", "median")
    ),
    "ert": library_kind(ert_frame, ert_plot, ("dimension", "ERT")),
    "fixed_budget_ecdf": library_kind(
        fixed_budget_ecdf_frame, fixed_budget_ecdf_plot, ("raw_y", "ecdf")
    ),
}
