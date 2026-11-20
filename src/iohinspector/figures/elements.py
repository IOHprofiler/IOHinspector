"""``elements(spec, data) -> tree``: every element of a figure, as Studio selects
them, with its box.

Each node is ``{"id", "kind", "box", "svg", "children"}`` and, for some kinds,
more keys. ``box`` is ``[x0, y0, x1, y1]`` in points from the top left of the
figure: the SVG's own coordinates. ``svg`` lists the ids the node's parts carry
in the SVG ``render`` writes (its own id when it is one part, ``<id>-<k>`` for
several); a node without parts (a series) is the union of its children.

The ids are stable: the same for the same spec and labels. ``figure`` holds
``title``, the panels ``p<i>`` and a shared ``legend``. Panel ``p<i>`` (key
``panel``: i) holds ``p<i>-plot`` (the plot area), ``-frame``, ``-grid``,
``-title``, ``-xaxis`` and ``-yaxis`` (each with its ``-xlabel`` and tick labels
``-xticks``), the series ``p<i>-s-<slug>`` front first (keys ``label``, the key
of ``FigureSpec.series``, and ``name``), each with ``-line``, ``-band``,
``-end`` (its final value), ``-markers`` and ``-label`` (its direct label), its
``-legend`` and the annotations ``p<i>-a<j>`` (key ``annotation``: j).
"""

import polars as pl
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox, BboxBase

from .render import _figure, _prepare, style
from .spec import FigureSpec


def elements(
    spec: FigureSpec,
    data: pl.DataFrame | list[pl.DataFrame] | None = None,
    *,
    frames: pl.DataFrame | list[pl.DataFrame | None] | None = None,
) -> dict:
    """The element tree of the figure ``render(spec, data, frames=frames)``
    draws."""
    nodes: dict[str, dict] = {}
    parts: dict[str, list] = {}

    def record(id, kind, parent, artists, **meta):
        parts[id] = artists or []
        nodes[id] = {"id": id, "kind": kind, **meta, "children": []}
        nodes[id]["svg"] = [a.get_gid() for a in parts[id]]
        if parent is not None:
            nodes[parent]["children"].append(nodes[id])

    with style(spec):
        fig = _figure(spec, *_prepare(spec, data, frames), record)
        height = fig.bbox.height
        scale = 72 / fig.dpi

        def box(node) -> Bbox | None:
            boxes = [_extent(a) for a in parts.get(node["id"], [])]
            boxes += [b for b in map(box, node["children"]) if b is not None]
            if not boxes:
                return None
            b = Bbox.union(boxes)
            node["box"] = [
                round(float(v * scale), 2)
                for v in (b.x0, height - b.y1, b.x1, height - b.y0)
            ]
            return b

        box(nodes["figure"])
    return nodes["figure"]


def _extent(artist) -> BboxBase:
    if isinstance(artist, Figure):
        return artist.bbox
    if isinstance(artist, (Axes, Axis)):
        return artist.get_tightbbox()  # type: ignore[return-value]
    return artist.get_window_extent()
