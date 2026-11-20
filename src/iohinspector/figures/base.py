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
    mid: float | None = None
    z: np.ndarray | None = None


@dataclass
class Plot:
    """The curves of a figure kind and its defaults for ``spec.Axis`` fields
    (``label``, ``scale``, ``lim``, ``ticks``, ``grid``, and ``ticklabels``: names
    for the ticks 0, 1, ...). ``mark`` says what a curve draws (``marks.MARKS``):

    - ``line``: ``x`` and ``y`` (``step``: each value holding until the next x),
      with the band ``lo`` to ``hi``;
    - ``points``: the same, as markers only;
    - ``bars``: bars at ``x`` of height ``y``, with error bars ``lo`` to ``hi``;
    - ``box``: a box plot of the sample ``y`` at the position ``x[0]``;
    - ``violin``: the density ``lo`` to ``hi`` (a band at the position) along the
      grid ``y``, with the median ``mid``;
    - ``cells``: the cells ``z[row, column]`` between the edges ``y`` and ``x``,
      in the curve's colour (also ``color_below`` where negative, ``nan``: empty),
      with their values written in them by the format ``fmt``.

    ``annotations`` are added to the panel's own; ``legend`` False leaves out the
    legend and the direct labels (the ticks name the curves)."""

    curves: list[Curve]
    x: dict = field(default_factory=dict)
    y: dict = field(default_factory=dict)
    step: bool = False
    mark: str = "line"
    fmt: str | None = None
    legend: bool = True
    annotations: list = field(default_factory=list)


@dataclass(frozen=True)
class Kind:
    """A figure kind: ``frame(panel, runs)`` computes its frame from the runs
    (``panel.data.where`` already applied) and ``plot(panel, frame)`` draws one;
    ``columns`` are the frame columns ``plot`` needs besides the groups, x and y
    first. For ``to_python``: ``code(panel)`` is the Python that sets ``frame``
    from ``runs``, with ``label`` per row, using ``imports``. ``literal``: the
    curves are not rows of the frame, so ``to_python`` writes them out as numbers
    after a comment with ``code``."""

    frame: Callable[..., pl.DataFrame]
    plot: Callable[..., Plot]
    columns: tuple[str, ...]
    code: Callable[..., str]
    imports: str
    literal: bool = False


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


def library_kind(frame, plot, columns: tuple[str, ...]) -> Kind:
    """A kind whose ``frame`` function is public: its code is a call of it."""

    def code(panel) -> str:
        data = {
            k: v
            for k, v in vars(panel.data).items()
            if v != getattr(type(panel.data)(), k)
        }
        return (
            f"frame = {frame.__name__}(Panel({panel.kind!r}, Data(**{data!r})), runs)"
        )

    imports = (
        "from iohinspector.figures import Data, Panel\n"
        f"from {frame.__module__} import {frame.__name__}"
    )
    return Kind(frame, plot, columns, code, imports, literal=True)


def pairs(runs: pl.DataFrame) -> list[pl.DataFrame]:
    """The runs of each (function, dimension) pair: for the metrics that take
    one pair at a time."""
    return runs.partition_by("function_id", "dimension", maintain_order=True)


def one_pair(runs: pl.DataFrame, kind: str) -> pl.DataFrame:
    """``runs``, which must be of one (function, dimension) pair."""
    if len(pairs(runs)) > 1:
        raise ValueError(f"{kind}: one function and dimension; select it in data.where")
    return runs


def default_budget(runs: pl.DataFrame) -> int:
    """The largest logged evaluation."""
    return int(runs["evaluations"].max())  # type: ignore[arg-type]


def default_target(runs: pl.DataFrame, maximization: bool) -> float:
    """The loosest target every run reaches: the worst of the runs' best values."""
    best = runs.group_by("run_id").agg(
        (pl.col("raw_y").max() if maximization else pl.col("raw_y").min()).alias("b")
    )["b"]
    return float(best.min() if maximization else best.max())  # type: ignore[arg-type]


def finite(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """The rows of ``frame`` where ``column`` is a number (not null, nan or inf)."""
    return frame.filter(pl.col(column).is_finite())


def curves(panel, frame: pl.DataFrame, x: str, y: str, lo=None, hi=None) -> list[Curve]:
    """One curve per group of ``frame`` (sorted by label and ``x``): its ``x``
    and ``y`` columns, and the band columns ``lo`` and ``hi``."""
    frame = with_labels(panel, frame).sort("label", x, maintain_order=True)

    def column(rows, name):
        return None if name is None else rows[name].to_numpy().astype(float)

    return [
        Curve(
            str(label),
            column(rows, x),  # type: ignore[arg-type]
            column(rows, y),  # type: ignore[arg-type]
            column(rows, lo),
            column(rows, hi),
            rows["family"][0],
        )
        for (label,), rows in frame.group_by("label", maintain_order=True)
    ]


def edges(centres, log: bool = False) -> np.ndarray:
    """The edges of cells centred on ``centres`` (sorted either way), halfway
    between them, geometrically on a ``log`` axis."""
    c = np.log10(centres) if log else np.asarray(centres, dtype=float)
    c = np.r_[c[0] - 1, c, c[-1] + 1] if len(c) == 1 else c
    e = np.r_[
        c[0] - (c[1] - c[0]) / 2, (c[1:] + c[:-1]) / 2, c[-1] + (c[-1] - c[-2]) / 2
    ]
    return 10**e if log else e


def categories(names: list[str]) -> dict:
    """The axis defaults for categories at 0, 1, ...: their names as ticks."""
    return {"ticklabels": names, "lim": [-0.5, len(names) - 0.5], "grid": "none"}


def log_if_positive(*values, maximization: bool = False) -> str:
    """``log`` for an axis whose values are all positive (not when maximising)."""
    positive = all(np.all(np.asarray(v) > 0) for v in values)
    return "log" if positive and not maximization else "linear"
