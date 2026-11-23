"""The empirical attainment function of single-objective runs
(``docs/definitions/eaf.md``): the attainment surfaces of each group and the
difference of two groups."""

import polars as pl

from ..metrics import get_eaf, get_eaf_difference
from .base import (
    Curve,
    Plot,
    edges,
    evals_axis,
    library_kind,
    log_if_positive,
    matrix,
    with_labels,
)

PERCENTILES = (10, 50, 90)


def _value(frame: pl.DataFrame) -> str:
    """The value column: ``quality`` when the groups span several pairs."""
    return "quality" if "quality" in frame.columns else "raw_y"


def eaf_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_eaf`` in precise mode at the percentiles of ``Data.options``
    (default 10, 50 and 90)."""
    d = panel.data
    options = {"percentiles": PERCENTILES} | d.options
    return get_eaf(runs, d.by, maximization=d.maximization, **options)


def eaf_plot(panel, frame: pl.DataFrame) -> Plot:
    """One step line per group and percentile: the value the share of runs given
    by the percentile has reached within the evaluations."""
    value = _value(frame)
    frame = with_labels(panel, frame).sort(
        "label", "percentile", "evaluations", maintain_order=True
    )
    lines = [
        Curve(
            f"{label} ({p:g}%)",
            rows["evaluations"].to_numpy().astype(float),
            rows[value].to_numpy(),
            family=rows["family"][0],
        )
        for (label, p), rows in frame.group_by(
            "label", "percentile", maintain_order=True
        )
    ]
    scale = "linear" if value == "quality" else log_if_positive(frame[value])
    return Plot(
        lines,
        x=evals_axis(),
        y={"label": "quality" if value == "quality" else "value", "scale": scale},
        step=True,
    )


def eaf_difference_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_eaf_difference`` of the two groups on the grid of budgets and values
    (discretised mode)."""
    d = panel.data
    return get_eaf_difference(
        runs, d.by, mode="discretised", maximization=d.maximization, **d.options
    )


def eaf_difference_plot(panel, frame: pl.DataFrame) -> Plot:
    """The difference as cells over the budget and the value: positive (the
    first group's colour) where the first group of the two attains more often,
    negative (``color_below``) where the second does."""
    value = _value(frame)
    frame = with_labels(panel, frame)
    first, second = sorted(frame["label"].unique())[:2]
    rows = frame.filter(pl.col("label") == first)
    xs, ys = rows["evaluations"].unique().sort(), rows[value].unique().sort()
    log_y = value == "raw_y" and log_if_positive(ys) == "log"
    z = matrix(rows, value, "evaluations", "eaf_diff", ys, xs)
    x, y = edges(xs, log=True), edges(ys, log=log_y)
    return Plot(
        [Curve(f"{first} − {second}", x, y, z=z, family=rows["family"][0])],
        x=evals_axis(lim=[x[0], x[-1]]),
        y={
            "label": "quality" if value == "quality" else "value",
            "scale": "log" if log_y else "linear",
            "lim": [y[0], y[-1]],
        },
        mark="cells",
    )


KINDS = {
    "eaf": library_kind(eaf_frame, eaf_plot, ("evaluations", "percentile")),
    "eaf_difference": library_kind(
        eaf_difference_frame, eaf_difference_plot, ("evaluations", "eaf_diff")
    ),
}
