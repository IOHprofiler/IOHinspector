"""Figures that compare groups: the pairwise-test and win-fraction heatmaps, the
target-by-budget map, the cumulative difference plot and the Shapley bars."""

import numpy as np
import polars as pl

from ..metrics import (
    get_cumulative_difference,
    get_ecdf,
    get_ks_tests,
    get_shapley_values,
    get_win_fractions,
    group_label,
)
from ..targets import get_ecdf_targets
from .base import (
    Curve,
    Plot,
    budget_at,
    categories,
    curves,
    edges,
    evals_axis,
    library_kind,
    matrix,
    one_pair,
    per_pair,
    sample_at,
    target_at,
    with_labels,
)
from .spec import Annotation

TARGET_ROWS, BUDGET_COLUMNS = 21, 31  # the cells of the target-by-budget map
EMPTY = 0.02  # cells where a probability, or a difference, is below this are empty
TIE = 0.01  # probabilities this close tie in "best of"


def _heatmap(panel, frame: pl.DataFrame, value: str, label: str, fmt: str) -> Plot:
    """The matrix ``value`` of ``frame``, one row per group (``by``) and one
    column per ``by`` + ``_vs`` group, as cells: the mean over the rows of a cell."""
    by, template = panel.data.by, panel.data.label
    rows = frame.select(group_label(by, template).alias("row"))["row"]
    vs = frame.select(pl.col(f"{c}_vs").alias(c) for c in by)
    cols = vs.select(group_label(by, template).alias("col"))["col"]
    names = sorted({*rows, *cols})
    cells = frame.select(row=rows, col=cols, v=pl.col(value))
    z = matrix(cells, "row", "col", "v", names, names)
    e = np.arange(len(names) + 1) - 0.5
    axis = categories(names)
    return Plot(
        [Curve(label, e, e, z=z)],
        x=axis,
        y={**axis, "lim": axis["lim"][::-1]},  # the first group on top
        mark="cells",
        fmt=fmt,
        legend=False,
    )


def tests_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_ks_tests`` (Holm-corrected, one-sided: the row is better than the
    column) of the value (``Data.measure`` ``value``: at the budget ``Data.at``)
    or runtime (at the target ``Data.at``) of each (function, dimension) pair,
    the default being the pair's largest budget or the loosest target all runs
    reach."""
    d = panel.data

    def tests(run: pl.DataFrame) -> pl.DataFrame:
        sample, value, view = sample_at(d, run)
        values = {"higher_is_better": d.maximization} if value == "value" else {}
        return get_ks_tests(sample, value, d.by, view=[view], **values, **d.options)

    return per_pair(runs, tests)


def tests_plot(panel, frame: pl.DataFrame) -> Plot:
    """Over several pairs, the share of pairs on which the row is significantly
    better than the column; on one pair, the adjusted p-value."""
    if frame.select("function_id", "dimension").n_unique() > 1:
        return _heatmap(panel, frame, "better", "share of pairs: row better", "{:.2f}")
    return _heatmap(panel, frame, "p_adjusted", "adjusted p-value", "{:.3f}")


def wins_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_win_fractions``: how often the row beats the column, over the pairs."""
    d = panel.data
    return get_win_fractions(runs, by=d.by, maximization=d.maximization, **d.options)


def wins_plot(panel, frame: pl.DataFrame) -> Plot:
    return _heatmap(panel, frame, "win_fraction", "win fraction", "{:.2f}")


def target_budget_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_ecdf`` of each group at 21 targets, log-spaced over the values the
    runs reach and 31 budgets log-spaced from 1 (easiest target first), the same
    targets on every pair."""
    d = panel.data
    y = runs["raw_y"]
    lo, hi = float(y.filter(y > 0).min()), float(y.max())  # type: ignore[arg-type]
    targets = np.geomspace(lo, hi, TARGET_ROWS)
    targets = targets if d.maximization else targets[::-1]
    unit = pl.col("dimension") if d.per_dimension else 1
    top = runs.select((pl.col("evaluations") / unit).max()).item()
    budgets = np.geomspace(1, top, BUDGET_COLUMNS)
    table = runs.select("function_id", "dimension").unique()
    return pl.concat(
        get_ecdf(
            runs,
            table.with_columns(target=pl.lit(t)),
            d.by,
            x_values=budgets,
            maximization=d.maximization,
            divide_by_dimension=d.per_dimension,
            **d.options,
        ).with_columns(target=pl.lit(t))
        for t in targets
    )


def target_budget_plot(panel, frame: pl.DataFrame) -> Plot:
    """The map by ``Data.mode``: ``best`` (default): each cell in the colour of
    the group most likely to have reached the target (ties within 1 % to the one
    with the larger mean), cells below 2 % empty; ``one``: the probability of the
    ``Data.focus`` group (default the first); ``vs_best``: its probability less
    the best of the others, diverging, empty within 0.02."""
    d = panel.data
    frame = with_labels(panel, frame)
    targets = frame["target"].unique(maintain_order=True)  # easiest first
    budgets = frame["evaluations"].unique().sort()
    p = {
        str(label): matrix(rows, "target", "evaluations", "ecdf", targets, budgets)
        for (label,), rows in frame.group_by("label", maintain_order=True)
    }
    mode, focus = d.mode or "best", d.focus or min(p)
    if mode not in SHOW:
        raise ValueError(f"target_budget mode {mode!r}: best, one or vs_best")
    if mode != "best" and focus not in p:
        raise ValueError(f"target_budget focus {focus!r}: one of {sorted(p)}")
    x, y = edges(budgets, log=True), edges(targets, log=True)
    return Plot(
        [Curve(k, x, y, z=z, family=k) for k, z in SHOW[mode](p, focus).items()],
        x=evals_axis(d.per_dimension, lim=[x[0], x[-1]]),
        y={
            "label": "target",
            "scale": "log",
            "ticks": "exponent",
            "lim": [y[0], y[-1]],
        },
        mark="cells",
    )


def _best(p: dict, focus) -> dict:
    """Each cell in the colour of the group most likely to have reached the
    target, ties within ``TIE`` to the better ranked (larger mean); cells below
    ``EMPTY`` empty."""
    ranked = sorted(p, key=lambda k: -np.nanmean(p[k]))
    top = np.max([p[k] for k in ranked], axis=0)
    won = np.full(top.shape, "", dtype=object)
    for k in ranked[::-1]:  # the better ranked group last: it wins ties
        won[p[k] >= top - TIE] = k
    won[top < EMPTY] = ""
    shown = {k: np.where(won == k, p[k], np.nan) for k in ranked}
    return {k: z for k, z in shown.items() if np.isfinite(z).any()}


def _one(p: dict, focus) -> dict:
    """The probability of the ``focus`` group."""
    return {focus: p[focus]}


def _vs_best(p: dict, focus) -> dict:
    """The ``focus`` group's probability less the best of the others, empty
    within ``EMPTY``."""
    others = [z for k, z in p.items() if k != focus]
    if not others:
        raise ValueError("target_budget vs_best: needs a second group")
    diff = p[focus] - np.max(others, axis=0)
    return {focus: np.where(abs(diff) < EMPTY, np.nan, diff)}


#: the looks of the target-by-budget map (``Data.mode``)
SHOW = {"best": _best, "one": _one, "vs_best": _vs_best}


def cumulative_difference_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_cumulative_difference`` of the two groups of one pair, with its
    ``label`` ("A − B"): at the target ``Data.at`` for ``measure="runtime"``
    (default: the loosest all runs reach), else at the budget ``Data.at``."""
    d = panel.data
    run = one_pair(runs, "cumulative_difference")
    if d.measure == "runtime":
        at = {"target": target_at(d, run)}
    else:
        at = {"budget": budget_at(d, run)}
    options = {"seed": 0} | d.options  # the same band every time
    result = get_cumulative_difference(
        run, by=d.by, maximization=d.maximization, **at, **options
    )
    groups = run.select(d.by).unique().sort(d.by).select(group_label(d.by, d.label))
    a, b = groups.to_series()[:2]
    return result.curves.with_columns(label=pl.lit(f"{a} − {b}"))


def cumulative_difference_plot(panel, frame: pl.DataFrame) -> Plot:
    """Y_A − Y_B over the cumulative probability p, with its bootstrap band;
    above zero A is the better."""
    return Plot(
        curves(panel, frame, "p", "diff", "diff_lower", "diff_upper"),
        x={"label": "$p$", "lim": [0, 1]},
        y={"label": "$Y_A - Y_B$"},
        annotations=[Annotation("line", y=0)],
    )


def shapley_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_shapley_values``: each group's share of the portfolio's value."""
    d = panel.data
    return get_shapley_values(
        runs,
        get_ecdf_targets(runs, d.targets),
        d.by,
        maximization=d.maximization,
        divide_by_dimension=d.per_dimension,
        **d.options,
    )


def shapley_plot(panel, frame: pl.DataFrame) -> Plot:
    """One bar per group, the largest first, with its standard error when the
    values were sampled."""
    frame = with_labels(panel, frame).sort("shapley", descending=True)
    bars = [
        Curve(
            label,
            np.array([i]),
            np.array([v]),
            *(np.array([v - se, v + se]) if se else (None,) * 2),  # type: ignore[arg-type]
            family=family,
        )
        for i, (label, v, se, family) in enumerate(
            frame.select("label", "shapley", "se", "family").iter_rows()
        )
    ]
    return Plot(
        bars,
        x=categories([c.label for c in bars]),
        y={"label": "Shapley value", "lim": [0, None]},
        mark="bars",
        legend=False,
    )


KINDS = {
    "pairwise_tests": library_kind(tests_frame, tests_plot, ("p_adjusted", "better")),
    "win_fractions": library_kind(wins_frame, wins_plot, ("win_fraction",)),
    "target_budget": library_kind(
        target_budget_frame, target_budget_plot, ("target", "evaluations", "ecdf")
    ),
    "cumulative_difference": library_kind(
        cumulative_difference_frame,
        cumulative_difference_plot,
        ("label", "p", "diff", "diff_lower", "diff_upper"),
    ),
    "shapley": library_kind(shapley_frame, shapley_plot, ("shapley", "se")),
}
