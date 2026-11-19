"""``pgfplots(spec, data) -> {name.tex, name.csv}``: the figure as a pgfplots
``tikzpicture`` plus the curves as a CSV table, to edit in LaTeX.

The picture sets no font, so it takes the document's. Colours, line widths and
dashes, markers, text sizes, axis size and limits, and the positions of the
direct labels are those ``render`` draws. The axis is as large as in the
matplotlib figure, whose tick labels and direct labels sit outside it: set
``width`` and ``height`` in the ``.tex`` to fit a column. Needs the LaTeX
packages ``pgfplots`` (compat 1.18 or later) and ``xcolor``.
"""

import io

import matplotlib as mpl
import polars as pl
from matplotlib import colors

from .render import _figure, _prepare, axis_settings, marker_points, style, tex_escape
from .spec import FigureSpec

LINESTYLES = {"-": "solid", "--": "dashed", "-.": "dashdot", ":": "dotted"}
# matplotlib marker: pgfplots mark with its options
MARKS = {
    "o": "*",
    "s": "square*",
    "^": "triangle*",
    "D": "diamond*",
    "v": "triangle*, rotate=180",
    "p": "pentagon*",
    "X": "x",
    "*": "star",
}
LEGEND_POS = {
    "upper": "north",
    "lower": "south",
    "center": "",
    "left": "west",
    "right": "east",
}
TICKS = {  # log axis tick labels other than the default 10^k; \tick is the natural log
    "exponent": r"$\pgfmathparse{int(round(\tick/ln(10)))}\pgfmathresult$",
    "plain": r"$\pgfmathparse{10^int(round(\tick/ln(10)))}\pgfmathprintnumber[fixed,precision=4]{\pgfmathresult}$",
}


def pgfplots(
    spec: FigureSpec, data: pl.DataFrame, name: str = "figure"
) -> dict[str, bytes]:
    """The pgfplots picture of ``spec`` as ``{name}.tex``, reading ``{name}.csv``
    (columns ``series``, ``x``, ``y``; series 0 is the highest final value) from
    the document's folder, and that table."""
    with style(spec):
        plot, styles = _prepare(spec, data)
        fig = _figure(spec, plot, styles)
        fig.draw_without_rendering()  # the layout: axes size, limits, labels
        ax = fig.axes[0]
        curves = sorted(plot.curves, key=lambda c: -c.y[-1])  # legend order
        lines = [
            "\\begin{tikzpicture}",
            f"\\begin{{axis}}[{_axis_options(spec, plot, ax)}]",
        ]
        for i, curve in enumerate(curves):
            look = styles[curve.label]
            options = _line(look) + (", const plot" if plot.step else "")
            keep = f"y filter/.expression={{\\thisrow{{series}}=={i} ? y : nan}}"
            lines.append(
                f"\\addplot[{options}, {keep}] table[col sep=comma, x=x, y=y] {{{name}.csv}};"
            )
            if spec.legend.mode in ("box", "outside"):
                lines.append(f"\\addlegendentry{{{tex_escape(look['name'])}}}")
        lines += _markers(ax, spec, plot, styles, curves)
        lines += _direct_labels(ax)
        lines += ["\\end{axis}", "\\end{tikzpicture}", ""]
    csv = io.StringIO()
    csv.write("series,x,y\n")
    for i, curve in enumerate(curves):
        csv.writelines(
            f"{i},{x!r},{y!r}\n" for x, y in zip(curve.x.tolist(), curve.y.tolist())
        )
    return {
        f"{name}.tex": "\n".join(lines).encode(),
        f"{name}.csv": csv.getvalue().encode(),
    }


def _bp(points: float) -> str:
    return f"{points:.4g}bp"  # matplotlib's points are 1/72 inch: TeX's big points


def _font(size: float) -> str:
    return f"font=\\fontsize{{{size:g}}}{{{1.2 * size:g}}}\\selectfont"


def _color(value) -> str:
    r, g, b = (round(255 * c) for c in colors.to_rgb(value))
    return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


def _axis_options(spec: FigureSpec, plot, ax) -> str:
    rc = mpl.rcParams
    width, height = (v * 72 / ax.figure.dpi for v in (ax.bbox.width, ax.bbox.height))
    opts = [
        "scale only axis",
        f"width={_bp(width)}",
        f"height={_bp(height)}",
        "tick align=inside",
        "clip mode=individual",  # the labels lie outside the axes
        f"axis line style={{line width={_bp(rc['axes.linewidth'])}}}",
        f"tick style={{line width={_bp(rc['xtick.major.width'])}}}",
        f"minor tick style={{line width={_bp(rc['xtick.minor.width'])}}}",
        f"grid style={{line width={_bp(rc['grid.linewidth'])}, draw={_color(rc['grid.color'])}}}",
        f"label style={{{_font(rc['axes.labelsize'])}}}",
        f"tick label style={{{_font(rc['xtick.labelsize'])}}}",
        f"title style={{{_font(rc['axes.titlesize'])}}}",
    ]
    for which, own, defaults in (("x", spec.x, plot.x), ("y", spec.y, plot.y)):
        lo, hi = getattr(ax, f"get_{which}lim")()
        opts += [
            f"{which}min={lo:.10g}",
            f"{which}max={hi:.10g}",
            f"{which}label={{{tex_escape(getattr(ax, f'get_{which}label')())}}}",
        ]
        if getattr(ax, f"get_{which}scale")() == "log":
            opts.append(f"{which}mode=log")
            ticks = axis_settings(own, defaults)["ticks"]
            if ticks in TICKS:
                opts.append(f"{which}ticklabel={{{TICKS[ticks]}}}")
        if own.grid != "none":
            opts.append(f"{which}majorgrids")
        if own.grid == "both":
            opts.append(f"{which}minorgrids")
    if spec.title:
        opts.append(f"title={{{tex_escape(spec.title)}}}")
    mode, loc = spec.legend.mode, spec.legend.loc.split()
    legend = f"{_font(rc['legend.fontsize'])}, draw={_color(rc['legend.edgecolor'])}, legend cell align=left"
    if mode == "box":
        pos = (
            " ".join(filter(None, (LEGEND_POS.get(w, "") for w in loc))) or "south east"
        )  # best
        opts.append(f"legend pos={pos}")
    elif mode == "outside":
        opts.append("legend pos=outer north east")
    if mode in ("box", "outside"):
        opts.append(f"legend style={{{legend}}}")
    return ",\n    ".join(opts)


def _line(look: dict) -> str:
    """The pgfplots options of a line: colour, width, and the dashes
    matplotlib draws for its style (scaled by the width)."""
    width = look["linewidth"]
    opts = [
        f"color={_color(look['color'])}",
        f"line width={_bp(width)}",
        "line cap=butt",
    ]
    name = LINESTYLES.get(look["linestyle"])
    if name is None:
        opts.append("draw=none")  # matplotlib's "None": no line
    elif name != "solid":
        scale = width if mpl.rcParams["lines.scale_dashes"] else 1
        pattern = mpl.rcParams[f"lines.{name}_pattern"]
        on_off = " ".join(
            f"{'on' if i % 2 == 0 else 'off'} {_bp(v * scale)}"
            for i, v in enumerate(pattern)
        )
        opts.append(f"dash pattern={on_off}")
    return ", ".join(opts)


def _markers(ax, spec: FigureSpec, plot, styles, curves) -> list[str]:
    """The sparse markers and the final value of each curve, as inline points."""
    sparse = {c.label: (x, y) for c, x, y in marker_points(ax, spec.markers, plot)}
    out = []
    for curve in curves:
        look = styles[curve.label]
        x, y = sparse.get(curve.label, ([], []))
        pairs = [*zip(list(x), list(y)), (curve.x[-1], curve.y[-1])]
        points = " ".join(f"({float(a)!r},{float(b)!r})" for a, b in pairs)
        mark, _, extra = MARKS.get(look["marker"], "*").partition(", ")
        c = _color(look["color"])
        out.append(
            f"\\addplot[only marks, forget plot, color={c}, mark={mark}, mark size={_bp(mpl.rcParams['lines.markersize'] / 2)}, "
            f"mark options={{solid, fill={c}{', ' + extra if extra else ''}}}] coordinates {{{points}}};"
        )
    return out


def _direct_labels(ax) -> list[str]:
    """Each label of ``render`` at the right of the axes, joined to the end of
    its curve by a leader."""
    out = []
    for text in ax.texts:
        color, (x, y) = _color(text.arrow_patch.get_edgecolor()), text.xy
        above = text.get_position()
        at = f"(rel axis cs:{above[0]:.4f},{above[1]:.4f})"
        out.append(
            f"\\draw[color={color}, line width={_bp(text.arrow_patch.get_linewidth())}] "
            f"(axis cs:{float(x)!r},{float(y)!r}) -- {at};\n"
            f"\\node[anchor=west, inner sep=1bp, text={color}, {_font(text.get_fontsize())}] at {at} "
            f"{{{tex_escape(text.get_text())}}};"
        )
    return out
