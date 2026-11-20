"""``render(spec, data) -> bytes``: one publication figure, the same bytes for the
same spec, data and matplotlib version on every machine (``pgf``: and TeX
installation, which measures the text).

Every element the figure draws carries a stable id (its SVG ``id``); see
``figures.elements`` for the tree of them."""

import contextlib
import io
import math
import re
from collections import Counter
from collections.abc import Callable

import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.style import context as style_context
from matplotlib.text import Text

from .base import Curve, Kind, Plot
from .ecdf import ECDF
from .labels import add_direct_labels, place_direct_labels
from .spec import FORMATS, UNITS, Axis, FigureSpec, Panel, Series, _one_of
from .venues import DARK_PALETTES, HIGHLIGHT, LINESTYLES, MARKERS, PALETTES, VENUES

KINDS: dict[str, Kind] = {"ecdf": ECDF}
METADATA = {  # without dates; the PGF backend takes none
    "pdf": {"metadata": {"CreationDate": None}},
    "svg": {"metadata": {"Date": None}},
    "png": {"metadata": {}},
}
TICK_FORMATS = {
    "exponent": lambda v, _: f"${round(math.log10(v), 6):g}$",
    "power": lambda v, _: f"$10^{{{round(math.log10(v), 6):g}}}$",
    "sci": lambda v, _: f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-"),
    "plain": lambda v, _: f"${v:.12g}$",
}
DARK = {  # a dark slide: Graphite dark's well, text and edges
    "figure.facecolor": "#121316",
    "axes.facecolor": "#121316",
    "text.color": "#eceef1",
    "axes.labelcolor": "#eceef1",
    "axes.edgecolor": "#c3c8d0",
    "xtick.color": "#c3c8d0",
    "ytick.color": "#c3c8d0",
    "grid.color": "#2e323a",
    "legend.facecolor": "#121316",
    "legend.edgecolor": "#5c626b",
}
TEX_SPECIAL = {
    "#": r"\#",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
# add(id, kind, parent, artists, **meta): an element, its id set on its artists
Add = Callable[..., None]


def tex_escape(text: str) -> str:
    """``text`` for LaTeX: its special characters made literal, except in
    ``$...$`` math, which stays as it is."""
    plain = re.compile("|".join(map(re.escape, TEX_SPECIAL)))
    parts = text.split("$")
    parts[::2] = [plain.sub(lambda m: TEX_SPECIAL[m[0]], t) for t in parts[::2]]
    return "$".join(parts)


def render(
    spec: FigureSpec,
    data: pl.DataFrame | list[pl.DataFrame] | None = None,
    *,
    frames: pl.DataFrame | list[pl.DataFrame | None] | None = None,
    fmt: str | None = None,
) -> bytes:
    """The figure ``spec`` describes, as ``fmt`` (``spec.formats[0]`` by default):
    PDF, SVG or PNG at ``spec.dpi``, or ``pgf``: a PGF picture for ``\\input`` in
    a LaTeX document, set in that document's own font (it needs a TeX
    installation to measure text; ``pgfplots`` does not). matplotlib's own
    settings do not leak in.

    Each panel is drawn from its frame in ``frames`` (``panel_frames``, or the
    numbers the website shows), else from the runs ``data`` (their logged rows,
    ``DataManager.load``). Either is one frame for every panel or a list with one
    per panel."""
    fmt = fmt or spec.formats[0]
    _one_of("format", fmt, FORMATS)
    buffer = io.BytesIO()
    with style(spec):
        fig = _figure(spec, *_prepare(spec, data, frames))
        if fmt == "pgf":  # TeX reads the text
            for text in fig.findobj(Text):
                text.set_text(tex_escape(text.get_text()))
        fig.savefig(buffer, format=fmt, dpi=spec.dpi, **METADATA.get(fmt, {}))
    out = buffer.getvalue()
    # the tick labels use matplotlib's \mathdefault, which the document lacks
    return b"\\providecommand{\\mathdefault}[1]{#1}\n" + out if fmt == "pgf" else out


def panel_frames(spec: FigureSpec, data) -> list[pl.DataFrame]:
    """The frame of each panel (``base.Kind``), computed from the runs ``data``:
    one frame for every panel or a list with one per panel."""
    return [
        KINDS[panel.kind].frame(panel, _select(panel, runs))
        for panel, runs in zip(spec.panels, _per_panel(spec, data, "data"))
    ]


@contextlib.contextmanager
def style(spec: FigureSpec):
    """matplotlib's settings for ``spec``, and nothing of the user's."""
    with style_context("default"), mpl.rc_context(rc_params(spec)):
        yield


def figure_size(spec: FigureSpec) -> tuple[float, float]:
    """The width and height of the figure in inches."""
    unit = UNITS[spec.unit]
    width = spec.width * unit if spec.width else VENUES[spec.venue].width_in(spec.span)
    height = spec.height * unit if spec.height else width * spec.aspect
    return width, height


def series_styles(
    spec: FigureSpec, labels: list[str], families: list | None = None
) -> dict[str, dict]:
    """The look of each label: the spec's ``series`` overrides, else the palette,
    line styles and markers in the order of ``labels``, and the series' ``id``
    (a slug of its label, unique among ``labels``). The ``family`` palette
    colours by ``families`` (default: the labels); ``highlight`` colours the
    series highlighted yours (wider) and other (thinner) by its rules."""
    name = spec.palette if isinstance(spec.palette, str) else None
    dark = spec.background == "dark"
    palette = (dark and DARK_PALETTES.get(name or "")) or PALETTES.get(name or "")
    palette = palette or spec.palette
    roles = HIGHLIGHT["dark" if dark else "white"]
    families = families or labels
    family_index = {f: i for i, f in enumerate(dict.fromkeys(families))}
    in_family, ids, styles = Counter(), set(), {}
    width = mpl.rcParams["lines.linewidth"]
    index = 0  # among the series that take a palette colour
    for label, family in zip(labels, families):
        own = spec.series.get(label, Series())
        role = own.highlight if name == "highlight" else None
        if name == "family":
            color = palette[family_index[family] % len(palette)]
            dash = in_family[family]
            in_family[family] += 1
        elif role:
            color, dash = roles[role], index
        else:
            color, dash = palette[index % len(palette)], index
            index += 1
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "series"
        while slug in ids:
            slug += "_"
        ids.add(slug)
        styles[label] = {
            "id": slug,
            "name": own.name or label,
            "color": own.color or color,
            "color_below": own.color_below,
            "linestyle": own.linestyle or LINESTYLES[dash % len(LINESTYLES)],
            "marker": own.marker or MARKERS[dash % len(MARKERS)],
            "marker_size": own.marker_size or mpl.rcParams["lines.markersize"],
            "linewidth": own.linewidth
            or width * {"yours": 1.5, "other": 0.75}.get(role or "", 1),
            "every": own.every,
            "visible": own.visible,
            "order": own.order,
            "band": own.band,
            "band_opacity": own.band_opacity,
            "highlight": own.highlight,
        }
    return styles


def rc_params(spec: FigureSpec) -> dict:
    """matplotlib's ``rcParams`` for ``spec``: fonts, sizes, line widths and the
    background."""
    venue, text = VENUES[spec.venue], spec.text
    size = text.size or venue.caption_size
    rc = {
        "font.family": text.family or venue.mpl_font,
        "mathtext.fontset": venue.mathtext,
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": text.title_size or venue.size,
        "figure.titlesize": text.title_size or venue.size,
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
    if spec.background == "dark":
        rc |= DARK
    return rc | {"savefig.transparent": spec.background == "none"}


def _per_panel(spec: FigureSpec, value, name: str) -> list:
    n = len(spec.panels)
    if value is None or isinstance(value, pl.DataFrame):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{name}: one per panel, {n}; got {len(value)}")
    return list(value)


def _select(panel: Panel, data: pl.DataFrame | None) -> pl.DataFrame:
    if data is None:
        raise ValueError("give the runs (data) or the frames of every panel")
    for column, values in panel.data.where.items():
        data = data.filter(pl.col(column).is_in(values))
    if data.is_empty():
        raise ValueError(f"data.where {panel.data.where!r} selects no rows")
    return data


def _plot(panel: Panel, runs, frame) -> Plot:
    kind = KINDS[panel.kind]
    if frame is None:
        frame = kind.frame(panel, _select(panel, runs))
    missing = [c for c in kind.columns if c not in frame.columns]
    if missing:
        raise ValueError(f"the {panel.kind} frame has no columns {missing}")
    return kind.plot(panel, frame)


def _prepare(spec: FigureSpec, data, frames=None) -> tuple[list[Plot], dict]:
    """Each panel's visible curves in drawing order (by ``order``; others behind,
    yours in front), and the look of every label, the same in every panel."""
    plots = [
        _plot(*args)
        for args in zip(
            spec.panels,
            _per_panel(spec, data, "data"),
            _per_panel(spec, frames, "frames"),
        )
    ]
    first = {c.label: c.family or c.label for p in plots for c in p.curves}
    styles = series_styles(spec, list(first), list(first.values()))
    rank = {"other": -1, "yours": 1}

    def key(curve: Curve):
        look = styles[curve.label]
        return look["order"], rank.get(look["highlight"], 0)

    for plot in plots:
        shown = [c for c in plot.curves if styles[c.label]["visible"]]
        plot.curves = sorted(shown, key=key)
    return plots, styles


def _figure(
    spec: FigureSpec, plots: list[Plot], styles: dict, record: Add | None = None
):
    """The figure, laid out, with an id on every element; ``record`` gets each
    element as ``add`` does, with only its drawn artists (``elements``)."""

    def add(id: str, kind: str, parent: str | None, artists, **meta) -> None:
        if artists is not None:
            artists = [a for a in artists if _drawn(a)]
            if not artists:
                return
            for k, artist in enumerate(artists):
                artist.set_gid(id if len(artists) == 1 else f"{id}-{k}")
        if record:
            record(id, kind, parent, artists, **meta)

    fig = Figure(figsize=figure_size(spec), layout="constrained")
    add("figure", "figure", None, [fig])
    layout, n = spec.layout, len(plots)
    columns = min(layout.columns, n)
    grid = fig.subplots(
        math.ceil(n / columns),
        columns,
        sharex=layout.share_x,
        sharey=layout.share_y,
        squeeze=False,
    ).ravel()
    for ax in grid[n:]:
        ax.remove()
    axes = list(grid[:n])
    if n > 1 and layout.gap is not None:
        pad = layout.gap * UNITS[spec.unit] / 2
        fig.get_layout_engine().set(w_pad=pad, h_pad=pad, wspace=0, hspace=0)  # type: ignore[union-attr]
    if spec.title:
        add("title", "title", "figure", [fig.suptitle(spec.title)])
    for i, (ax, panel, plot) in enumerate(zip(axes, spec.panels, plots)):
        _panel(ax, i, spec, panel, plot, styles, add)
    for i, (ax, plot) in enumerate(zip(axes, plots)):  # on final limits
        for curve, x, y in marker_positions(ax, spec, plot, styles):
            look = styles[curve.label]
            line = ax.plot(
                x,
                y,
                marker=look["marker"],
                markersize=look["marker_size"],
                linestyle="none",
                color=look["color"],
            )
            add(f"p{i}-s-{look['id']}-markers", "markers", f"p{i}-s-{look['id']}", line)
    direct = _legends(fig, axes, spec, plots, styles, add)
    fig.draw_without_rendering()  # lay out, so the axes' sizes and ticks are known
    for ax, texts in direct:
        place_direct_labels(ax, texts, mpl.rcParams["legend.fontsize"])
    for i, ax in enumerate(axes):
        lines = []
        for which in "xy":
            ticks = getattr(ax, f"{which}axis")._update_ticks()
            labels = [t.label1 for t in ticks]
            add(f"p{i}-{which}ticks", "ticks", f"p{i}-{which}axis", labels)
            lines += [t.gridline for t in ticks]
        add(f"p{i}-grid", "grid", f"p{i}", lines)
    return fig


def _drawn(artist) -> bool:
    if isinstance(artist, Figure):
        return True
    if isinstance(artist, Text) and not artist.get_text():
        return False
    return artist.get_visible()


def panel_title(spec: FigureSpec, panel: Panel, i: int) -> str:
    """The title of panel ``i``: its name, or its letter, (a), with its name."""
    labels = spec.layout.labels
    if labels == "letters":
        return " ".join(filter(None, [f"({chr(ord('a') + i)})", panel.title]))
    return (panel.title or "") if labels == "names" else ""


def _panel(ax, i: int, spec, panel: Panel, plot: Plot, styles, add: Add) -> None:
    p = f"p{i}"
    add(p, "panel", "figure", [ax], panel=i)
    add(f"{p}-plot", "plot", p, [ax.patch])
    for curve in reversed(plot.curves):  # front first
        look = styles[curve.label]
        add(
            f"{p}-s-{look['id']}",
            "series",
            p,
            None,
            label=curve.label,
            name=look["name"],
        )
    for curve in plot.curves:
        _draw(
            ax,
            curve,
            styles[curve.label],
            plot.step,
            f"{p}-s-{styles[curve.label]['id']}",
            add,
        )
    _axis(ax, "x", panel.x, plot.x)  # after the curves, for automatic limits
    _axis(ax, "y", panel.y, plot.y)
    for side, spine in ax.spines.items():
        shown = {"box": True, "spines": side in ("left", "bottom")}
        spine.set_visible(shown.get(spec.frame, False))
    add(f"{p}-frame", "frame", p, list(ax.spines.values()))
    title = panel_title(spec, panel, i)
    if title:
        loc = "left" if spec.layout.labels == "letters" else "center"
        add(f"{p}-title", "title", p, [ax.set_title(title, loc=loc)])
    for which in "xy":
        axis = getattr(ax, f"{which}axis")
        add(f"{p}-{which}axis", "axis", p, [axis])
        add(f"{p}-{which}label", "label", f"{p}-{which}axis", [axis.label])
    size = spec.text.annotation_size or mpl.rcParams["font.size"]
    for j, note in enumerate(panel.annotations):
        add(f"{p}-a{j}", "annotation", p, _annotate(ax, note, size), annotation=j)


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
    log = a["scale"] == "log"
    if own.tick_count:
        n = own.tick_count
        axis.set_major_locator(
            ticker.LogLocator(10, numticks=n) if log else ticker.MaxNLocator(n - 1 or 1)
        )
    elif log and a["ticks"] != "auto":
        axis.set_major_locator(ticker.LogLocator(10))
    if log and a["ticks"] != "auto":
        axis.set_major_formatter(ticker.FuncFormatter(TICK_FORMATS[a["ticks"]]))
        axis.set_minor_formatter(ticker.NullFormatter())
    if own.grid != "none":
        ax.grid(True, which=own.grid, axis=which)


def _draw(ax, curve: Curve, look: dict, step: bool, sid: str, add: Add) -> None:
    if curve.lo is not None and curve.hi is not None and look["band"]:
        band = ax.fill_between(
            curve.x,
            curve.lo,
            curve.hi,
            step="post" if step else None,
            color=look["color"],
            alpha=look["band_opacity"],
            linewidth=0,
        )
        add(f"{sid}-band", "band", sid, [band])
    line = {k: look[k] for k in ("color", "linestyle", "linewidth")}
    drawn = ax.plot(
        curve.x, curve.y, drawstyle="steps-post" if step else "default", **line
    )
    add(f"{sid}-line", "line", sid, drawn)
    end = ax.plot(  # the final value, at the curve's end
        curve.x[-1:],
        curve.y[-1:],
        marker=look["marker"],
        markersize=look["marker_size"],
        color=look["color"],
        linestyle="none",
        clip_on=False,
    )
    add(f"{sid}-end", "end", sid, end)


def _annotate(ax, note, size: float) -> list:
    """The artists of one annotation (``spec.Annotation``)."""
    color = note.color or mpl.rcParams["text.color"]
    text = {"fontsize": size, "color": color}
    if note.kind == "text":
        return [ax.text(note.x, note.y, note.text, **text)]
    if note.kind == "arrow":
        arrow = {"arrowstyle": "->", "color": color, "linewidth": 0.6}
        return [
            ax.annotate(
                note.text,
                (note.x, note.y),
                (note.x2, note.y2),
                arrowprops=arrow,
                **text,
            )
        ]
    vertical = note.x is not None
    lo, hi = (note.x, note.x2) if vertical else (note.y, note.y2)
    if note.kind == "line":
        draw = ax.axvline if vertical else ax.axhline
        out = [draw(lo, color=color, linewidth=0.8, linestyle="--")]
    else:
        span = ax.axvspan if vertical else ax.axhspan
        out = [span(lo, hi, color=color, alpha=0.12, linewidth=0)]
    if note.text:
        at = (lo, 0.98) if vertical else (0.02, lo)
        where = ax.get_xaxis_transform() if vertical else ax.get_yaxis_transform()
        va = "top" if vertical else "bottom"
        out.append(ax.text(*at, note.text, transform=where, va=va, ha="left", **text))
    return out


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


def marker_positions(ax, spec: FigureSpec, plot: Plot, styles: dict):
    """The markers of each curve: on every n-th point with ``Series.every``,
    else the sparse ones. Yields ``(curve, x, y)``."""
    sparse = {c.label: (x, y) for c, x, y in marker_points(ax, spec.markers, plot)}
    for curve in plot.curves:
        every = styles[curve.label]["every"]
        if every:
            yield curve, curve.x[::every], curve.y[::every]
        elif curve.label in sparse:
            yield curve, *sparse[curve.label]


def legend_entries(spec: FigureSpec, curves: list[Curve], styles: dict) -> list[dict]:
    """The looks of ``curves`` (in drawing order) in the legend's order, each
    with its curve as ``end``; with ``legend.others`` the series highlighted
    other become one "n others" entry, last, with no ``end``."""
    looks = [styles[c.label] | {"end": c} for c in curves]
    order = spec.legend.order
    if order == "drawing":
        looks.reverse()  # front first
    elif order == "name":
        looks.sort(key=lambda s: s["name"])
    else:
        yours = order == "yours"
        looks.sort(key=lambda s: (yours and s["highlight"] != "yours", -s["end"].y[-1]))
    if spec.legend.others:
        others = [s for s in looks if s["highlight"] == "other"]
        looks = [s for s in looks if s["highlight"] != "other"]
        if others:
            looks.append(others[0] | {"name": f"{len(others)} others", "end": None})
    return looks


def _legends(fig, axes, spec: FigureSpec, plots, styles, add: Add) -> list:
    """Direct labels or legends; returns the direct labels of each axes, to
    place once the layout is known."""
    legend, direct = spec.legend, []
    if legend.mode == "none":
        return direct
    if legend.mode == "outside" or (
        legend.mode == "box" and legend.shared and len(axes) > 1
    ):
        union = {}
        for plot in plots:
            for curve in plot.curves:
                union.setdefault(curve.label, curve)
        handles = [
            _handle(s) for s in legend_entries(spec, list(union.values()), styles)
        ]
        loc = (
            "outside right center"
            if legend.mode == "outside"
            else "outside lower center"
        )
        add(
            "legend",
            "legend",
            "figure",
            [fig.legend(handles=handles, loc=loc, ncols=legend.columns)],
        )
        return direct
    for i, (ax, plot) in enumerate(zip(axes, plots)):
        entries = legend_entries(spec, plot.curves, styles)
        if legend.mode == "box":
            box = ax.legend(
                handles=list(map(_handle, entries)),
                loc=legend.loc,
                ncols=legend.columns,
            )
            add(f"p{i}-legend", "legend", f"p{i}", [box])
            continue
        ends = [s for s in entries if s["end"] is not None]
        size = mpl.rcParams["legend.fontsize"]
        points = [(s["name"], s["end"].x[-1], s["end"].y[-1], s["color"]) for s in ends]
        texts = add_direct_labels(ax, points, size)
        for look, text in zip(ends, texts):
            sid = f"p{i}-s-{look['id']}"
            add(f"{sid}-label", "label", sid, [text])
        direct.append((ax, texts))
    return direct


def _handle(look: dict) -> Line2D:
    keys = ("color", "linestyle", "linewidth", "marker")
    return Line2D(
        [],
        [],
        label=look["name"],
        markersize=look["marker_size"],
        **{k: look[k] for k in keys},
    )
