"""The BBOB-style ECDF (``docs/definitions/ecdf.md``): one step line per group of
``by``, from the first logged evaluation to the largest budget of all groups; past
a group's own budget its runs are restarted (EC-7), as in cocopp.

Its frame is ``metrics.get_ecdf``'s result: the ``by`` columns (or a ``label``
column), ``evaluations`` (per dimension when ``data.per_dimension``) and
``ecdf``, one row per step; optional ``lo`` and ``hi`` columns draw a band. The x
axis runs from its smallest positive ``evaluations`` to its largest, where every
line ends at its final value: ``ecdf_frame`` adds a row per group at the largest
budget."""

import numpy as np
import polars as pl

from ..metrics.ecdf import get_ecdf
from ..targets import get_ecdf_targets
from .base import Curve, Kind, Plot, with_labels


def ecdf_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """The ECDF of each group of ``panel.data.by`` over the targets
    ``panel.data.targets``, with x the evaluations (per dimension with
    ``panel.data.per_dimension``), and each group's final value at the largest
    budget."""
    d = panel.data
    x = pl.col("evaluations").cast(pl.Float64)
    if d.per_dimension:
        x = x / pl.col("dimension")
    ecdf = get_ecdf(
        runs,
        get_ecdf_targets(runs, d.targets),
        d.by,
        maximization=d.maximization,
        divide_by_dimension=d.per_dimension,
    )
    end = ecdf.group_by(d.by, maintain_order=True).last()
    end = end.with_columns(evaluations=runs.select(x.max()).item())
    return pl.concat([ecdf, end]).unique(
        [*d.by, "evaluations"], keep="first", maintain_order=True
    )


def ecdf_plot(panel, frame: pl.DataFrame) -> Plot:
    """The step line of each group of an ECDF frame, sorted by label."""
    frame = with_labels(panel, frame).sort("label", "evaluations", maintain_order=True)
    x = frame["evaluations"]
    start, end = x.filter(x > 0).min(), x.max()
    curves = [
        _curve(label, rows, start, end)
        for (label,), rows in frame.group_by("label", maintain_order=True)
    ]
    unit = "# f-evals / dimension" if panel.data.per_dimension else "# f-evals"
    return Plot(
        curves,
        x={"label": unit, "scale": "log", "ticks": "exponent", "lim": [start, end]},
        y={"label": "Fraction of targets reached", "lim": [0, 1]},
        step=True,
    )


def _curve(label: str, rows: pl.DataFrame, start, end) -> Curve:
    """The steps of one group from ``start``, where it has the value it reached
    by then, to ``end``, where it keeps its final value."""
    x = rows["evaluations"].to_numpy()
    i = max(np.searchsorted(x, start, side="right") - 1, 0)
    after = slice(i + 1, None)  # the steps after start

    def column(name):
        if name not in rows.columns:
            return None
        v = rows[name].to_numpy()
        return np.r_[v[i], v[after], v[-1]]

    x = np.r_[start, x[after], end]
    if x[-1] == x[-2]:  # the frame ends at the largest budget already
        x = x[:-1]
    n = len(x)
    lo, hi = column("lo"), column("hi")
    return Curve(
        str(label),
        x,
        column("ecdf")[:n],  # type: ignore[index]
        lo if lo is None else lo[:n],
        hi if hi is None else hi[:n],
        rows["family"][0],
    )


ECDF = Kind(ecdf_frame, ecdf_plot, ("evaluations", "ecdf"))
