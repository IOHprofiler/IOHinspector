"""``render(spec, data) -> bytes``: one publication figure, the same bytes for the
same spec, data and matplotlib version on every machine."""

import io
import math

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib import style, ticker
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .base import Curve, Plot
from .ecdf import ecdf_plot
from .labels import add_direct_labels, place_direct_labels
from .spec import FORMATS, Axis, FigureSpec, Series, _one_of
from .venues import LINESTYLES, MARKERS, PALETTES, VENUES

KINDS = {"ecdf": ecdf_plot}
CM = 1 / 2.54
METADATA = {
    "pdf": {"CreationDate": None},
    "svg": {"Date": None},
    "png": {"Software": None},
}
TICK_FORMATS = {
    "exponent": lambda v, _: f"${round(math.log10(v), 6):g}$",
    "power": lambda v, _: f"$10^{{{round(math.log10(v), 6):g}}}$",
    "plain": lambda v, _: f"${v:.12g}$",
}


def render(spec: FigureSpec, data: pl.DataFrame, *, fmt: str | None = None) -> bytes:
    """The figure ``spec`` describes, drawn from ``data`` (the runs' logged rows,
    ``DataManager.load``), as ``fmt`` (``spec.formats[0]`` by default): PDF, SVG
    or PNG at ``spec.dpi``. matplotlib's own settings do not leak in."""
    fmt = fmt or spec.formats[0]
    _one_of("format", fmt, FORMATS)
    buffer = io.BytesIO()
    with style.context("default"), mpl.rc_context(_rc(spec)):
        fig = _figure(spec, _select(spec, data))
        fig.savefig(buffer, format=fmt, dpi=spec.dpi, metadata=METADATA[fmt])
    return buffer.getvalue()


def figure_size(spec: FigureSpec) -> tuple[float, float]:
    """The width and height of the figure in inches."""
    venue = VENUES[spec.venue]
    width = spec.width_cm * CM if spec.width_cm else venue.width_in(spec.span)
    height = spec.height_cm * CM if spec.height_cm else width * spec.aspect
    return width, height


def series_styles(spec: FigureSpec, labels: list[str]) -> dict[str, dict]:
    """Name, colour, line style, marker, line width and visibility of each label:
    the spec's ``series`` overrides, else the palette, line styles and markers in
    the order of ``labels``."""
    palette = PALETTES.get(spec.palette, spec.palette)  # type: ignore[arg-type]
    styles = {}
    for i, label in enumerate(labels):
        own = spec.series.get(label, Series())
        styles[label] = {
            "name": own.name or label,
            "color": own.color or palette[i % len(palette)],
            "linestyle": own.linestyle or LINESTYLES[i % len(LINESTYLES)],
            "marker": own.marker or MARKERS[i % len(MARKERS)],
            "linewidth": own.linewidth or mpl.rcParams["lines.linewidth"],
            "visible": own.visible,
        }
    return styles


def _rc(spec: FigureSpec) -> dict:
    venue, text = VENUES[spec.venue], spec.text
    size = text.size or venue.caption_size
    return {
        "font.family": text.family or venue.mpl_font,
        "mathtext.fontset": venue.mathtext,
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": text.title_size or venue.size,
        "xtick.labelsize": text.tick_size or size,
        "ytick.labelsize": text.tick_size or size,
        "legend.fontsize": text.legend_size or size,
        "axes.formatter.use_mathtext": True,  # cmr10 has no minus sign
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "grid.linewidth": 0.4,
        "grid.color": "#d0d0d0",
        "legend.fancybox": False,
        "legend.edgecolor": "#808080",
        "pdf.fonttype": 42,
        "svg.hashsalt": "iohinspector",
    }


def _select(spec: FigureSpec, data: pl.DataFrame) -> pl.DataFrame:
    for column, values in spec.data.where.items():
        data = data.filter(pl.col(column).is_in(values))
    if data.is_empty():
        raise ValueError(f"data.where {spec.data.where!r} selects no rows")
    return data


def _figure(spec: FigureSpec, data: pl.DataFrame) -> Figure:
    plot: Plot = KINDS[spec.kind](spec, data)
    styles = series_styles(spec, [c.label for c in plot.curves])
    shown = [c for c in plot.curves if styles[c.label]["visible"]]
    fig = Figure(figsize=figure_size(spec), layout="constrained")
    ax = fig.add_subplot()
    for curve in shown:
        _draw(ax, curve, styles[curve.label], plot.step)
    _axis(ax, "x", spec.x, plot.x)  # after the curves, for automatic limits
    _axis(ax, "y", spec.y, plot.y)
    if spec.title:
        ax.set_title(spec.title)
    for i, curve in enumerate(shown):  # spread over the final x limits
        phase = (i + 1) / (len(shown) + 1)
        style = styles[curve.label]
        _sparse_markers(ax, curve, style, n=spec.markers, phase=phase, step=plot.step)
    _legend(fig, ax, spec, shown, styles)
    return fig


def _axis(ax, which: str, own: Axis, defaults: dict) -> None:
    scale = own.scale or defaults.get("scale", "linear")
    ticks = defaults.get("ticks", "auto") if own.ticks == "auto" else own.ticks
    label = defaults.get("label", "") if own.label is None else own.label
    if own.label is None and label and scale == "log" and ticks == "exponent":
        label = f"$\\log_{{10}}$({label})"
    getattr(ax, f"set_{which}scale")(scale)
    getattr(ax, f"set_{which}label")(label)
    lim = own.lim or defaults.get("lim")
    if lim:
        getattr(ax, f"set_{which}lim")(*lim)
    axis = getattr(ax, f"{which}axis")
    if scale == "log" and ticks != "auto":
        axis.set_major_locator(ticker.LogLocator(10))
        axis.set_major_formatter(ticker.FuncFormatter(TICK_FORMATS[ticks]))
        axis.set_minor_formatter(ticker.NullFormatter())
    if own.grid != "none":
        ax.grid(True, which=own.grid, axis=which)


def _draw(ax, curve: Curve, style: dict, step: bool) -> None:
    line = {k: style[k] for k in ("color", "linestyle", "linewidth")}
    ax.plot(curve.x, curve.y, drawstyle="steps-post" if step else "default", **line)
    ax.plot(  # the final value, at the curve's end
        curve.x[-1:],
        curve.y[-1:],
        marker=style["marker"],
        color=style["color"],
        linestyle="none",
        clip_on=False,
    )


def _sparse_markers(ax, curve: Curve, style, *, n: int, phase: float, step: bool):
    """``n`` markers spread evenly over the x axis (on its scale),
    shifted by ``phase`` of the spacing per curve so curves do not share them."""
    if not n:
        return
    log = ax.get_xscale() == "log"
    scale = np.log10 if log else (lambda v: np.asarray(v, dtype=float))
    lo, hi = scale(ax.get_xlim())
    t = lo + (np.arange(n) + phase) * (hi - lo) / n
    x = 10**t if log else t
    x = x[(x >= curve.x[0]) & (x < curve.x[-1])]
    if step:
        y = curve.y[np.searchsorted(curve.x, x, side="right") - 1]
    else:
        y = np.interp(x, curve.x, curve.y)
    ax.plot(x, y, marker=style["marker"], color=style["color"], linestyle="none")


def _legend(fig, ax, spec: FigureSpec, curves: list[Curve], styles) -> None:
    """Direct labels or a legend, both ordered by final value, highest first."""
    order = sorted(curves, key=lambda c: -c.y[-1])
    size = mpl.rcParams["legend.fontsize"]
    if spec.legend.mode == "direct":
        ends = [
            (styles[c.label]["name"], c.x[-1], c.y[-1], styles[c.label]["color"])
            for c in order
        ]
        texts = add_direct_labels(ax, ends, size)
        fig.draw_without_rendering()  # lay out, so the axes' height is known
        place_direct_labels(ax, texts, size)
        return
    if spec.legend.mode == "none":
        return
    handles = [_handle(styles[c.label]) for c in order]
    if spec.legend.mode == "box":
        ax.legend(handles=handles, loc=spec.legend.loc, ncols=spec.legend.columns)
    else:
        fig.legend(
            handles=handles, loc="outside right center", ncols=spec.legend.columns
        )


def _handle(style: dict) -> Line2D:
    keys = ("color", "linestyle", "linewidth", "marker")
    return Line2D([], [], label=style["name"], **{k: style[k] for k in keys})
