"""The parameter figures. Every runs set (a configuration) is a group of
``Data.by``, the recorded parameters that vary, and its measure is the area under
the ECDF (``get_ecdf_auc``). Parallel coordinates, the marginals and the
sensitivity bars show all parameters; the map two of them."""

import numpy as np
import polars as pl

from ..metrics import get_ecdf_auc
from ..targets import get_ecdf_targets
from .base import Curve, Plot, categories, library_kind, matrix, with_labels

MEASURE = "ECDF AUC"


def parameters_frame(panel, runs: pl.DataFrame) -> pl.DataFrame:
    """``get_ecdf_auc`` of each configuration: its parameters (``Data.by``) and
    ``auc``."""
    d = panel.data
    return get_ecdf_auc(
        runs,
        get_ecdf_targets(runs, d.targets),
        d.by,
        divide_by_dimension=d.per_dimension,
        maximization=d.maximization,
        **d.options,
    )


def _scaled(values: pl.Series) -> np.ndarray:
    """``values`` on [0, 1]: numbers by their range, anything else by the rank of
    its value."""
    if values.dtype.is_numeric():
        v = values.to_numpy().astype(float)
        span = v.max() - v.min()
        return (v - v.min()) / span if span else np.full(len(v), 0.5)
    order = {v: i for i, v in enumerate(sorted(values.unique()))}
    return np.array([order[v] for v in values]) / max(len(order) - 1, 1)


def _marginal(frame: pl.DataFrame, parameter: str) -> pl.DataFrame:
    """The mean measure at each value of ``parameter``, in its order."""
    return frame.group_by(parameter).agg(pl.col("auc").mean()).sort(parameter)


def parallel_plot(panel, frame: pl.DataFrame) -> Plot:
    """One line per configuration through an axis per parameter and the measure,
    each scaled to [0, 1]."""
    by = panel.data.by
    frame = with_labels(panel, frame)
    scaled = np.column_stack([*(_scaled(frame[c]) for c in by), _scaled(frame["auc"])])
    lines = [
        Curve(label, np.arange(len(by) + 1.0), y)
        for label, y in zip(frame["label"], scaled)
    ]
    return Plot(
        lines,
        x=categories([*by, MEASURE]),
        y={"label": "scaled", "lim": [0, 1]},
        legend=False,
    )


def marginals_plot(panel, frame: pl.DataFrame) -> Plot:
    """The mean measure over each parameter's values, one line per parameter, the
    values scaled to [0, 1] (the parameters have no common unit)."""
    lines = []
    for p in panel.data.by:
        m = _marginal(frame, p)
        lines.append(Curve(p, _scaled(m[p]), m["auc"].to_numpy()))
    return Plot(
        lines,
        x={"label": "parameter value (scaled)", "lim": [0, 1]},
        y={"label": f"mean {MEASURE}"},
    )


def sensitivity_plot(panel, frame: pl.DataFrame) -> Plot:
    """One bar per parameter: the range of its marginal mean, the largest first."""
    spread = {
        p: float(np.ptp(_marginal(frame, p)["auc"].to_numpy())) for p in panel.data.by
    }
    order = sorted(spread, key=lambda p: -spread[p])
    bars = [Curve(p, np.array([i]), np.array([spread[p]])) for i, p in enumerate(order)]
    return Plot(
        bars,
        x=categories(order),
        y={"label": f"range of the mean {MEASURE}", "lim": [0, None]},
        mark="bars",
        legend=False,
    )


def map_plot(panel, frame: pl.DataFrame) -> Plot:
    """The mean measure over two parameters (``Data.by``), as cells; a cell no
    configuration is in is empty."""
    if len(panel.data.by) != 2:
        raise ValueError("parameter_map: two parameters in data.by")
    px, py = panel.data.by
    xs, ys = frame[px].unique().sort(), frame[py].unique().sort()
    z = matrix(frame, py, px, "auc", ys, xs)
    ex, ey = (np.arange(len(v) + 1) - 0.5 for v in (xs, ys))
    return Plot(
        [Curve(MEASURE, ex, ey, z=z)],
        x={**categories([str(v) for v in xs]), "label": px},
        y={**categories([str(v) for v in ys]), "label": py},
        mark="cells",
        fmt="{:.2f}",
        legend=False,
    )


KINDS = {
    kind: library_kind(parameters_frame, plot, ("auc",))
    for kind, plot in {
        "parallel_coordinates": parallel_plot,
        "marginals": marginals_plot,
        "parameter_map": map_plot,
        "sensitivity": sensitivity_plot,
    }.items()
}
