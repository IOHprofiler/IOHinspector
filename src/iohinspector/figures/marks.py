"""What a curve draws besides a line (``base.Plot.mark``): points, bars, box plots,
violins and cells. Each function draws one curve ``c`` (``base.Curve``) with its
``look`` (the series' ``color``, ``color_below``, ``linewidth``, ``linestyle``,
``marker`` and ``marker_size``) and returns its artists by part, which become the
elements ``<series>-<part>``.

``to_python`` writes these functions' source into the code it generates, so they
use only matplotlib and numpy and take what they draw as attributes."""

import matplotlib as mpl
import numpy as np


def points(ax, c, look, fmt=None):
    marker = {"marker": look["marker"], "markersize": look["marker_size"]}
    return {
        "points": ax.plot(c.x, c.y, linestyle="none", color=look["color"], **marker)
    }


def bars(ax, c, look, fmt=None):
    out = {"bars": list(ax.bar(c.x, c.y, width=0.8, color=look["color"]))}
    if c.lo is not None:
        err = ax.errorbar(
            c.x,
            c.y,
            yerr=[c.y - c.lo, c.hi - c.y],
            fmt="none",
            capsize=2,
            linewidth=0.6,
        )
        out["errors"] = [*err.lines[1], *err.lines[2]]
    return out


def box(ax, c, look, fmt=None):
    color = look["color"]
    line = {"color": color, "linewidth": look["linewidth"]}
    parts = ax.boxplot(
        c.y,
        positions=c.x[:1],
        widths=0.6,
        patch_artist=True,
        manage_ticks=False,
        boxprops={**line, "facecolor": color, "alpha": 0.4},
        medianprops=line,
        whiskerprops=line,
        capprops=line,
        flierprops={
            "marker": look["marker"],
            "markeredgecolor": color,
            "markersize": 2.5,
        },
    )
    return {"box": [a for artists in parts.values() for a in artists]}


def violin(ax, c, look, fmt=None):
    fill = ax.fill_betweenx(
        c.y, c.lo, c.hi, color=look["color"], alpha=0.4, linewidth=0
    )
    out = {"violin": [fill]}
    if c.mid is not None:
        at = [np.interp(c.mid, c.y, edge) for edge in (c.lo, c.hi)]
        out["median"] = [
            ax.hlines(c.mid, *at, color=look["color"], linewidth=look["linewidth"])
        ]
    return out


def cells(ax, c, look, fmt=None):
    diverging = bool(np.nanmin(c.z) < 0)
    ground = mpl.rcParams["axes.facecolor"]
    ramp = [look["color_below"] or "#0072B2", ground, look["color"]]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(  # type: ignore[attr-defined]
        "cells", ramp[not diverging :]
    )
    ink = lambda m, v: (
        "white" if np.dot(m.cmap(m.norm(v))[:3], [0.3, 0.6, 0.1]) < 0.5 else "black"
    )
    mesh = ax.pcolormesh(
        c.x,
        c.y,
        np.ma.masked_invalid(c.z),
        cmap=cmap,
        vmin=-1 if diverging else 0,
        vmax=1,
        zorder=2,
    )
    out = {"cells": [mesh]}
    if fmt:
        size = mpl.rcParams["font.size"] * 0.75
        mid = lambda e: (np.asarray(e[1:]) + np.asarray(e[:-1])) / 2
        out["values"] = [
            ax.text(
                x,
                y,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=size,
                zorder=3,
                color=ink(mesh, v),
            )
            for y, row in zip(mid(c.y), c.z)
            for x, v in zip(mid(c.x), row)
            if np.isfinite(v)
        ]
    return out


MARKS = {"points": points, "bars": bars, "box": box, "violin": violin, "cells": cells}
