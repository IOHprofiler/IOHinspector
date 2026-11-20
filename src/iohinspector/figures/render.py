"""``render(spec, data) -> bytes``: one publication figure, the same bytes for the
same spec, data and matplotlib version on every machine (``pgf``: and TeX
installation, which measures the text)."""

import contextlib
import dataclasses
import io
import math
import re

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.style import context as style_context
from matplotlib.text import Text

from .base import Curve, Plot
from .ecdf import ecdf_plot
from .labels import add_direct_labels, place_direct_labels
from .spec import FORMATS, Axis, FigureSpec, Series, _one_of
from .venues import LINESTYLES, MARKERS, PALETTES, VENUES

KINDS = {"ecdf": ecdf_plot}
CM = 1 / 2.54
METADATA = {  # without dates; the PGF backend takes none
    "pdf": {"metadata": {"CreationDate": None}},
    "svg": {"metadata": {"Date": None}},
    "png": {"metadata": {}},
}
TICK_FORMATS = {
    "exponent": lambda v, _: f"${round(math.log10(v), 6):g}$",
    "power": lambda v, _: f"$10^{{{round(math.log10(v), 6):g}}}$",
    "plain": lambda v, _: f"${v:.12g}$",
}


TEX_SPECIAL = {
    "#": r"\#",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(text: str) -> str:
    """``text`` for LaTeX: its special characters made literal, except in
    ``$...$`` math, which stays as it is."""
    plain = re.compile("|".join(map(re.escape, TEX_SPECIAL)))
    parts = text.split("$")
    parts[::2] = [plain.sub(lambda m: TEX_SPECIAL[m[0]], t) for t in parts[::2]]
    return "$".join(parts)


def render(spec: FigureSpec, data: pl.DataFrame, *, fmt: str | None = None) -> bytes:
    """The figure ``spec`` describes, drawn from ``data`` (the runs' logged rows,
    ``DataManager.load``), as ``fmt`` (``spec.formats[0]`` by default): PDF, SVG
    or PNG at ``spec.dpi``, or ``pgf``: a PGF picture for ``\\input`` in a LaTeX
    document, set in that document's own font (it needs a TeX installation to
    measure text; ``pgfplots.export`` does not). matplotlib's own settings do
    not leak in."""
    fmt = fmt or spec.formats[0]
    _one_of("format", fmt, FORMATS)
    buffer = io.BytesIO()
    with style(spec):
        fig = _figure(spec, *_prepare(spec, data))
        if fmt == "pgf":  # TeX reads the text
            for text in fig.findobj(Text):
                text.set_text(tex_escape(text.get_text()))
        fig.savefig(buffer, format=fmt, dpi=spec.dpi, **METADATA.get(fmt, {}))
    out = buffer.getvalue()
    # the tick labels use matplotlib's \mathdefault, which the document lacks
    return b"\\providecommand{\\mathdefault}[1]{#1}\n" + out if fmt == "pgf" else out


@contextlib.contextmanager
def style(spec: FigureSpec):
    """matplotlib's settings for ``spec``, and nothing of the user's."""
    with style_context("default"), mpl.rc_context(_rc(spec)):
        yield


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
        "pgf.rcfonts": False,  # the document's font, not font.family
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "",
        "svg.hashsalt": "iohinspector",
    }


def _select(spec: FigureSpec, data: pl.DataFrame) -> pl.DataFrame:
    for column, values in spec.data.where.items():
        data = data.filter(pl.col(column).is_in(values))
    if data.is_empty():
        raise ValueError(f"data.where {spec.data.where!r} selects no rows")
    return data


def _prepare(spec: FigureSpec, data: pl.DataFrame) -> tuple[Plot, dict]:
    """The visible curves of the selected ``data`` and the style of each label."""
    plot: Plot = KINDS[spec.kind](spec, _select(spec, data))
    styles = series_styles(spec, [c.label for c in plot.curves])
    shown = [c for c in plot.curves if styles[c.label]["visible"]]
    return dataclasses.replace(plot, curves=shown), styles


def _figure(spec: FigureSpec, plot: Plot, styles: dict) -> Figure:
    fig = Figure(figsize=figure_size(spec), layout="constrained")
    ax = fig.add_subplot()
    for curve in plot.curves:
        _draw(ax, curve, styles[curve.label], plot.step)
    _axis(ax, "x", spec.x, plot.x)  # after the curves, for automatic limits
    _axis(ax, "y", spec.y, plot.y)
    if spec.title:
        ax.set_title(spec.title)
    for curve, x, y in marker_points(ax, spec.markers, plot):  # on final limits
        ax.plot(
            x,
            y,
            marker=styles[curve.label]["marker"],
            linestyle="none",
            color=styles[curve.label]["color"],
        )
    _legend(fig, ax, spec, plot.curves, styles)
    return fig


def axis_settings(own: Axis, defaults: dict) -> dict:
    """The label, scale, limits and tick format of an axis: the spec's, else the
    figure kind's."""
    scale = own.scale or defaults.get("scale", "linear")
    ticks = defaults.get("ticks", "auto") if own.ticks == "auto" else own.ticks
    label = defaults.get("label", "") if own.label is None else own.label
    if own.label is None and label and scale == "log" and ticks == "exponent":
        label = f"$\\log_{{10}}$({label})"
    lim = own.lim or defaults.get("lim")
    return {"label": label, "scale": scale, "lim": lim, "ticks": ticks}


def _axis(ax, which: str, own: Axis, defaults: dict) -> None:
    a = axis_settings(own, defaults)
    getattr(ax, f"set_{which}scale")(a["scale"])
    getattr(ax, f"set_{which}label")(a["label"])
    if a["lim"]:
        getattr(ax, f"set_{which}lim")(*a["lim"])
    axis = getattr(ax, f"{which}axis")
    if a["scale"] == "log" and a["ticks"] != "auto":
        axis.set_major_locator(ticker.LogLocator(10))
        axis.set_major_formatter(ticker.FuncFormatter(TICK_FORMATS[a["ticks"]]))
        axis.set_minor_formatter(ticker.NullFormatter())
    if own.grid != "none":
        ax.grid(True, which=own.grid, axis=which)


def _draw(ax, curve: Curve, look: dict, step: bool) -> None:
    line = {k: look[k] for k in ("color", "linestyle", "linewidth")}
    ax.plot(curve.x, curve.y, drawstyle="steps-post" if step else "default", **line)
    ax.plot(  # the final value, at the curve's end
        curve.x[-1:],
        curve.y[-1:],
        marker=look["marker"],
        color=look["color"],
        linestyle="none",
        clip_on=False,
    )


def marker_points(ax, n: int, plot: Plot):
    """The sparse markers: for each curve, ``n`` points on it spread evenly over
    the x axis (on its scale), each curve shifted by its own fraction of the
    spacing so that curves do not share positions. Yields ``(curve, x, y)``."""
    log = ax.get_xscale() == "log"
    lo, hi = (np.log10 if log else np.asarray)(ax.get_xlim())
    for i, curve in enumerate(plot.curves if n else []):
        phase = (i + 1) / (len(plot.curves) + 1)
        t = lo + (np.arange(n) + phase) * (hi - lo) / n
        x = 10**t if log else t
        x = x[(x >= curve.x[0]) & (x < curve.x[-1])]  # within the curve's range
        if plot.step:
            y = curve.y[np.searchsorted(curve.x, x, side="right") - 1]
        else:
            y = np.interp(x, curve.x, curve.y)
        yield curve, x, y


def _legend(fig, ax, spec: FigureSpec, curves: list[Curve], styles) -> None:
    """Direct labels or a legend, both ordered by final value, highest first."""
    order = [
        styles[c.label] | {"end": c} for c in sorted(curves, key=lambda c: -c.y[-1])
    ]
    mode, columns = spec.legend.mode, spec.legend.columns
    if mode == "direct":
        _direct_labels(fig, ax, order)
    elif mode == "box":
        ax.legend(handles=list(map(_handle, order)), loc=spec.legend.loc, ncols=columns)
    elif mode == "outside":
        fig.legend(
            handles=list(map(_handle, order)), loc="outside right center", ncols=columns
        )


def _direct_labels(fig, ax, order: list[dict]) -> None:
    size = mpl.rcParams["legend.fontsize"]
    ends = [(s["name"], s["end"].x[-1], s["end"].y[-1], s["color"]) for s in order]
    texts = add_direct_labels(ax, ends, size)
    fig.draw_without_rendering()  # lay out, so the axes' height is known
    place_direct_labels(ax, texts, size)


def _handle(look: dict) -> Line2D:
    keys = ("color", "linestyle", "linewidth", "marker")
    return Line2D([], [], label=look["name"], **{k: look[k] for k in keys})
