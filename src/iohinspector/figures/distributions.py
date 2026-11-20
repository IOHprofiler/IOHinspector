"""Figures of the distribution of one value over the runs of each group: box
plots, violins and histograms of the runtimes at a target (``Data.measure``
``"runtime"``) or the best values at a budget (``"value"``), on one (function,
dimension) pair. ``Data.at`` is the target or budget, by default the loosest
target all runs reach or the largest budget."""

import numpy as np
import polars as pl

from ..metrics import get_histogram, get_runtime_sample, get_value_sample, get_violin
from .base import (
    Curve,
    Plot,
    categories,
    default_budget,
    default_target,
    finite,
    library_kind,
    log_if_positive,
    one_pair,
    with_labels,
)


def _sample(panel, runs: pl.DataFrame, kind: str) -> tuple[pl.DataFrame, str, str]:
    """The sample, its value column and its view column."""
    d = panel.data
    run = one_pair(runs, kind)
    if d.measure == "runtime":
        target = d.at if d.at is not None else default_target(run, d.maximization)
        sample = get_runtime_sample(run, [target], d.by, maximization=d.maximization)
        return sample, "runtime", "target"
    budget = d.at or default_budget(run)
    return (
        get_value_sample(run, [budget], d.by, maximization=d.maximization),
        "value",
        "budget",
    )


def _value(frame: pl.DataFrame) -> str:
    return "runtime" if "runtime" in frame.columns else "value"


def _label(panel) -> str:
    return "runtime (# f-evals)" if panel.data.measure == "runtime" else "best value"


def box_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """The sample, one row per run (``get_runtime_sample`` or ``get_value_sample``),
    without the runs that never reached the target or have no value yet."""
    sample, value, _ = _sample(panel, runs, "box")
    return finite(sample, value)


def box_plot(panel, frame: pl.DataFrame) -> Plot:
    """A box plot of each group, in the order of the labels."""
    value = _value(frame)
    frame = with_labels(panel, frame).sort("label")
    boxes = [
        Curve(
            str(label), np.array([i]), rows[value].to_numpy(), family=rows["family"][0]
        )
        for i, ((label,), rows) in enumerate(
            frame.group_by("label", maintain_order=True)
        )
    ]
    scale = log_if_positive(frame[value], maximization=panel.data.maximization)
    return Plot(
        boxes,
        x=categories([b.label for b in boxes]),
        y={"label": _label(panel), "scale": scale},
        mark="box",
        legend=False,
    )


def violin_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_violin``: quartiles, and the density on a grid, of each group."""
    d = panel.data
    sample, value, view = _sample(panel, runs, "violin")
    return get_violin(sample, value, d.by, view=[view])


def violin_plot(panel, frame: pl.DataFrame) -> Plot:
    """A violin of each group, as wide as its density relative to its highest
    (the same width for every group), with its median."""
    frame = with_labels(panel, frame).sort("label")
    shapes = []
    for i, row in enumerate(frame.iter_rows(named=True)):
        half = 0.4 * np.array(row["density"]) / max(row["density"])
        grid = np.array(row["x"])
        shapes.append(
            Curve(
                row["label"],
                np.array([i]),
                grid,
                i - half,
                i + half,
                row["family"],
                row["median"],
            )
        )
    return Plot(
        shapes,
        x=categories([s.label for s in shapes]),
        y={
            "label": _label(panel),
            "scale": log_if_positive(
                np.concatenate(frame["x"].to_list()),
                maximization=panel.data.maximization,
            ),
        },
        mark="violin",
        legend=False,
    )


def histogram_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_histogram``: the count of each bin, per group (``Data.options``:
    ``bins``)."""
    d = panel.data
    sample, value, view = _sample(panel, runs, "histogram")
    return get_histogram(sample, value, d.by, view=[view], **d.options)


def histogram_plot(panel, frame: pl.DataFrame) -> Plot:
    """The counts of each group as a step, filled down to zero."""
    frame = with_labels(panel, frame).sort("label", "left", maintain_order=True)
    bars = []
    for (label,), rows in frame.group_by("label", maintain_order=True):
        count = rows["count"].to_numpy().astype(float)
        x = np.r_[rows["left"].to_numpy(), rows["right"][-1]]
        y = np.r_[count, count[-1]]
        bars.append(Curve(str(label), x, y, np.zeros_like(y), y, rows["family"][0]))
    return Plot(
        bars,
        x={"label": _label(panel)},
        y={"label": "runs"},
        step=True,
    )


KINDS = {
    "box": library_kind(box_frame, box_plot, ("run_id",)),
    "violin": library_kind(violin_frame, violin_plot, ("median", "x", "density")),
    "histogram": library_kind(
        histogram_frame, histogram_plot, ("left", "right", "count")
    ),
}
