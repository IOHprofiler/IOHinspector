"""The BBOB-style ECDF (``docs/definitions/ecdf.md``): one step line per group of
``by``, from the first logged evaluation to the largest budget of all groups; past
a group's own budget its runs are restarted (EC-7), as in cocopp."""

import numpy as np
import polars as pl

from ..metrics.ecdf import get_ecdf
from ..metrics.utils import group_label
from ..targets import get_ecdf_targets
from .base import Curve, Plot
from .spec import FigureSpec


def ecdf_plot(spec: FigureSpec, data: pl.DataFrame) -> Plot:
    """The ECDF of each group of ``spec.data.by`` over the targets
    ``spec.data.targets``, with x the evaluations (per dimension with
    ``spec.data.per_dimension``)."""
    d = spec.data
    x = pl.col("evaluations").cast(pl.Float64)
    if d.per_dimension:
        x = x / pl.col("dimension")
    ecdf = get_ecdf(
        data,
        get_ecdf_targets(data, d.targets),
        d.by,
        maximization=d.maximization,
        divide_by_dimension=d.per_dimension,
    ).with_columns(group_label(d.by, d.label))
    start, end = data.select(x.filter(x > 0).min().alias("start"), x.max()).row(0)
    curves = [
        _curve(label, ecdf.filter(pl.col("label") == label), start, end)
        for label in sorted(ecdf["label"].unique())  # pairs="common"
    ]
    unit = "# f-evals / dimension" if d.per_dimension else "# f-evals"
    return Plot(
        curves,
        x={"label": unit, "scale": "log", "ticks": "exponent", "lim": [start, end]},
        y={"label": "Fraction of targets reached", "lim": [0, 1]},
        step=True,
    )


def _curve(label: str, ecdf: pl.DataFrame, start: float, end: float) -> Curve:
    """The steps of one group from ``start``, where it has the value it reached
    by then, to ``end``, where it keeps its final value."""
    x, y = ecdf["evaluations"].to_numpy(), ecdf["ecdf"].to_numpy()
    i = np.searchsorted(x, start, side="right") - 1
    after = slice(i + 1, None)  # the steps after start
    return Curve(label, np.r_[start, x[after], end], np.r_[y[i], y[after], y[-1]])
