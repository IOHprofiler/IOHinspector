"""Publication figures (D11): a ``FigureSpec`` and ``render(spec, data) -> bytes``.

The website and Python users get the same figure from the same spec. A spec has
one or more panels; each panel's kind computes a frame from the runs
(``panel_frames``) and draws only from it, so ``render(spec, frames=...)`` draws
exactly the numbers the website shows. Venue presets set the width and fonts
(``VENUES``); ``PALETTES`` are colour-blind safe and every series also has its
own line style and marker; ``STYLES`` are named looks (``apply_style``). Every
element has a stable id in the SVG; ``elements`` gives their tree with boxes,
and ``to_python`` the matplotlib code of a figure. For LaTeX, ``render`` writes
``fmt="pgf"`` and ``pgfplots`` a pgfplots picture with its CSV, both set in the
document's own font.
"""

from .elements import elements
from .python import to_python
from .render import figure_size, panel_frames, render, series_styles
from .spec import (
    STYLES,
    Annotation,
    Axis,
    Data,
    FigureSpec,
    Layout,
    Legend,
    Panel,
    Series,
    Text,
    add_panels,
    apply_style,
)
from .tex import pgfplots
from .venues import PALETTES, VENUES, Venue

__all__ = [
    "PALETTES",
    "STYLES",
    "VENUES",
    "Annotation",
    "Axis",
    "Data",
    "FigureSpec",
    "Layout",
    "Legend",
    "Panel",
    "Series",
    "Text",
    "Venue",
    "add_panels",
    "apply_style",
    "elements",
    "figure_size",
    "panel_frames",
    "pgfplots",
    "render",
    "series_styles",
    "to_python",
]
