"""What a figure kind is: a frame from the runs, and curves from the frame.

Every kind computes its numbers as a polars frame (``Kind.frame``) and draws
only from that frame (``Kind.plot``). ``render`` takes the runs or the frames
themselves: the website passes the frames it shows, so a figure has exactly its
numbers."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..metrics.utils import group_label


@dataclass
class Curve:
    """One group's line, in data coordinates; it ends at its last point. ``lo``
    and ``hi`` are an optional band; ``family`` groups curves for the
    ``family`` palette (default: the label)."""

    label: str
    x: np.ndarray
    y: np.ndarray
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    family: str | None = None


@dataclass
class Plot:
    """The curves of a figure kind and its defaults for ``spec.Axis`` fields
    (``label``, ``scale``, ``lim``, ``ticks``). ``step``: draw the curves as
    steps, each value holding until the next x."""

    curves: list[Curve]
    x: dict = field(default_factory=dict)
    y: dict = field(default_factory=dict)
    step: bool = False


@dataclass(frozen=True)
class Kind:
    """A figure kind: ``frame(panel, runs)`` computes its frame from the runs
    (``panel.data.where`` already applied) and ``plot(panel, frame)`` draws one;
    ``columns`` are the frame columns ``plot`` needs besides the groups."""

    frame: Callable[..., pl.DataFrame]
    plot: Callable[..., Plot]
    columns: tuple[str, ...]


def with_labels(panel, frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` with the ``label`` and ``family`` of each row: its own columns,
    else from the ``by`` columns and the label template (D17); the family is the
    first ``by`` column's value."""
    by, template = panel.data.by, panel.data.label
    if "label" not in frame.columns:
        missing = [c for c in by if c not in frame.columns]
        if missing:
            raise ValueError(f"frame: no label column, nor the by columns {missing}")
        frame = frame.with_columns(group_label(by, template))
    if "family" not in frame.columns:
        family = group_label(by[:1]) if by[0] in frame.columns else pl.col("label")
        frame = frame.with_columns(family.alias("family"))
    return frame
