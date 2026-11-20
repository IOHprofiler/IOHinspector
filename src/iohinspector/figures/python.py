"""``to_python(spec, data) -> str``: the matplotlib code of a figure, to restyle by
hand. It sets ``rcParams``, a ``STYLE`` table with each series' look, and per
panel computes the frame from ``data`` (the runs, from the data code that
Reproduce writes before it) and draws it. What ``render`` takes from the layout,
such as the axis limits and the heights of the direct labels, is written in as
numbers."""

import math

from matplotlib import colors

from .render import (
    KINDS,
    TICK_CODE,
    _annotate,
    _figure,
    _prepare,
    axis_settings,
    figure_size,
    legend_entries,
    rc_params,
    style,
)
from .spec import UNITS, FigureSpec

DRAW = '''

def draw(ax, frame, labels, x, y, end, step):
    """The series ``labels`` of ``frame`` (back first), from its columns ``x``
    and ``y``, each line held to ``end``."""
    for label in labels:
        rows = frame.filter(pl.col("label") == label).sort(x)
        look = STYLE[label]
        xs, ys = [*rows[x], end], [*rows[y], rows[y][-1]]
        if "lo" in rows.columns and look["band"]:
            ax.fill_between(
                rows[x],
                rows["lo"],
                rows["hi"],
                step="post" if step else None,
                color=look["color"],
                alpha=look["band_opacity"],
                linewidth=0,
            )
        line = {k: look[k] for k in ("color", "marker", "markersize")}
        ax.plot(
            xs,
            ys,
            drawstyle="steps-post" if step else "default",
            linestyle=look["linestyle"],
            linewidth=look["linewidth"],
            markevery=look["markevery"],
            **line,
        )
        ax.plot(xs[-1:], ys[-1:], linestyle="none", clip_on=False, **line)


def handle(label, name=None):
    look = STYLE[label]
    keys = ("color", "linestyle", "linewidth", "marker", "markersize")
    return Line2D([], [], label=name or look["name"], **{k: look[k] for k in keys})
'''


class _Calls:
    """Writes the method calls made on it as Python source, such as
    ``ax.text(1.0, 0.5, 'a')``; a call used as an argument of another call is
    written inside it."""

    def __init__(self, name: str, lines: list):
        self.name, self.lines = name, lines

    def __repr__(self) -> str:
        return self.name

    def __getattr__(self, method: str):
        def call(*args, **kwargs):
            for value in [*args, *kwargs.values()]:
                if isinstance(value, _Calls) and value in self.lines:
                    self.lines.remove(value)  # an argument, not a statement
            text = [*map(repr, args), *(f"{k}={v!r}" for k, v in kwargs.items())]
            out = _Calls(f"{self.name}.{method}({', '.join(text)})", self.lines)
            self.lines.append(out)
            return out

        return call


def to_python(spec: FigureSpec, data=None, *, frames=None) -> str:
    """Python that draws the figure ``render(spec, data, frames=frames)`` draws,
    with matplotlib and the frames' functions, from a variable ``data``."""
    with style(spec):
        plots, styles = _prepare(spec, data, frames)
        fig = _figure(spec, plots, styles)
        size = spec.text.annotation_size or rc_params(spec)["font.size"]
        n, layout = len(plots), spec.layout
        columns = min(layout.columns, n)
        kinds = [KINDS[p.kind] for p in spec.panels]
        out = [
            f'"""{spec.title or "Figure"}: an iohinspector figure as matplotlib code."""',
            "",
            "import math",
            "",
            "import matplotlib as mpl",
            "import matplotlib.pyplot as plt",
            "import polars as pl",
            "from matplotlib import ticker",
            "from matplotlib.lines import Line2D",
            *dict.fromkeys(k.imports for k in kinds),
            "",
            "# data: the runs, from the data code (Reproduce)",
            "",
            "mpl.rcParams.update(",
            "    {",
            *(f"        {k!r}: {v!r}," for k, v in rc_params(spec).items()),
            "    }",
            ")",
            "STYLE = {  # each series: its name and look",
            *(
                f"    {label!r}: {_look(spec, look)!r},"
                for label, look in styles.items()
            ),
            "}",
            DRAW,
            "fig, axes = plt.subplots(",
            f"    {math.ceil(n / columns)},",
            f"    {columns},",
            f"    figsize=({', '.join(f'{v:.4f}' for v in figure_size(spec))}),",
            '    layout="constrained",',
            f"    sharex={layout.share_x},",
            f"    sharey={layout.share_y},",
            "    squeeze=False,",
            ")",
            "axes = axes.ravel()",
            f"for ax in axes[{n}:]:",
            "    ax.remove()",
        ]
        if n > 1 and layout.gap is not None:
            pad = layout.gap * UNITS[spec.unit] / 2
            out.append(
                f"fig.get_layout_engine().set(w_pad={pad!r}, h_pad={pad!r}, wspace=0, hspace=0)"
            )
        if spec.title:
            out.append(f"fig.suptitle({spec.title!r})")
        for i, (ax, panel, plot, kind) in enumerate(
            zip(fig.axes, spec.panels, plots, kinds)
        ):
            out += ["", f"# panel {i + 1}", "runs = data"]
            out += [
                f"runs = runs.filter(pl.col({k!r}).is_in({v!r}))"
                for k, v in panel.data.where.items()
            ]
            labels = [c.label for c in plot.curves]
            x, y = kind.columns[:2]
            out += [
                kind.code(panel),
                f"ax = axes[{i}]",
                f"draw(ax, frame, {labels!r}, {x!r}, {y!r}, {float(ax.get_xlim()[1])!r}, {plot.step})",
            ]
            out += _axes(ax, panel, plot, spec)
            notes: list = []
            for note in panel.annotations:
                _annotate(_Calls("ax", notes), note, size)
            out += [repr(c) for c in list(notes)]
            out += _legend(ax, i, spec, plot, styles)
        if fig.legends:  # outside, or one box for all panels
            union = {}
            for plot in plots:
                for curve in plot.curves:
                    union.setdefault(curve.label, curve)
            entries = legend_entries(spec, list(union.values()), styles)
            loc = (
                "outside right center"
                if spec.legend.mode == "outside"
                else "outside lower center"
            )
            out += [
                "",
                f"fig.legend(handles={_handles(entries)}, loc={loc!r}, ncols={spec.legend.columns})",
            ]
    fmt = spec.formats[0] if spec.formats[0] != "pgf" else "pdf"
    out += ["", f'fig.savefig("figure.{fmt}", dpi={spec.dpi})', ""]
    return "\n".join(out)


def _look(spec: FigureSpec, look: dict) -> dict:
    keys = ("name", "color", "linestyle", "linewidth", "marker", "band", "band_opacity")
    every = look["every"] or (1 / spec.markers if spec.markers else None)
    return {k: look[k] for k in keys} | {
        "markersize": look["marker_size"],
        "markevery": every,
    }


def _axes(ax, panel, plot, spec: FigureSpec) -> list[str]:
    out = []
    for which, own, defaults in (("x", panel.x, plot.x), ("y", panel.y, plot.y)):
        lim = tuple(float(v) for v in getattr(ax, f"get_{which}lim")())
        out += [
            f"ax.set_{which}scale({getattr(ax, f'get_{which}scale')()!r})",
            f"ax.set_{which}lim{lim!r}",
            f"ax.set_{which}label({getattr(ax, f'get_{which}label')()!r})",
        ]
        log = getattr(ax, f"get_{which}scale")() == "log"
        ticks = axis_settings(own, defaults)["ticks"]
        if own.tick_count:
            n = own.tick_count
            locator = (
                f"LogLocator(10, numticks={n})" if log else f"MaxNLocator({n - 1 or 1})"
            )
            out.append(f"ax.{which}axis.set_major_locator(ticker.{locator})")
        elif log and ticks != "auto":
            out.append(f"ax.{which}axis.set_major_locator(ticker.LogLocator(10))")
        if log and ticks != "auto":
            out += [
                f"ax.{which}axis.set_major_formatter(ticker.FuncFormatter({TICK_CODE[ticks]}))",
                f"ax.{which}axis.set_minor_formatter(ticker.NullFormatter())",
            ]
        if own.grid != "none":
            out.append(f"ax.grid(True, which={own.grid!r}, axis={which!r})")
    hidden = [side for side, spine in ax.spines.items() if not spine.get_visible()]
    out += [f"ax.spines[{side!r}].set_visible(False)" for side in hidden]
    for loc in ("left", "center"):
        if title := ax.get_title(loc=loc):
            out.append(f"ax.set_title({title!r}, loc={loc!r})")
    return out


def _handles(entries: list[dict]) -> str:
    """The legend handles of ``entries``; "n others" keeps its own name."""
    calls = [
        f"handle({s['label']!r}{'' if s['end'] else ', ' + repr(s['name'])})"
        for s in entries
    ]
    return "[" + ", ".join(calls) + "]"


def _legend(ax, i: int, spec: FigureSpec, plot, styles) -> list[str]:
    """The panel's box legend or direct labels, where ``render`` put them."""
    legend = spec.legend
    if legend.mode == "box" and ax.get_legend():
        entries = legend_entries(spec, plot.curves, styles)
        return [
            f"ax.legend(handles={_handles(entries)}, loc={legend.loc!r}, ncols={legend.columns})"
        ]
    out = []
    for text in ax.texts:
        if not (text.get_gid() or "").endswith("-label"):
            continue
        x, y = (float(v) for v in text.xy)
        at = tuple(round(float(v), 4) for v in text.get_position())
        arrow = {
            "arrowstyle": "-",
            "color": colors.to_hex(text.arrow_patch.get_edgecolor()),
            "linewidth": 0.6,
            "shrinkA": 1,
            "shrinkB": 0,
        }
        out.append(
            f"ax.annotate({text.get_text()!r}, ({x!r}, {y!r}), xytext={at!r}, "
            f'textcoords="axes fraction", va="center", ha="left", '
            f"fontsize={text.get_fontsize()!r}, annotation_clip=False, arrowprops={arrow!r})"
        )
    return out
